# 修改loss保存逻辑，使多次训练的loss都能保存到同一CSV文件

import matplotlib
import argparse
matplotlib.use('Agg')  # 非交互式后端
import os
import csv
import math
import time
from pathlib import Path
from typing import Tuple
import torch
import torch.utils.data
import matplotlib.pyplot as plt
from augment import AugmentPipe3D
from network import Discriminator, Generator, MappingNetwork
from utils import cycle_dataloader, PorosityDataset, DiscriminatorLoss, GeneratorLoss, GradientPenalty, TransitionDistanceLoss
from MetricChecker import MetricChecker

# 创建保存结果的文件夹

# 定义损失CSV路径（移除初始化写入表头的逻辑，改为在save_losses中处理）

class Configs:
    def __init__(self, args: argparse.Namespace):
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 模型组件
        self.discriminator = None
        self.generator = None
        self.mapping_network = None

        ### 添加ADA增强管道
        self.augment_pipe = None

        # 损失函数
        self.discriminator_loss = None
        self.generator_loss = None

        # 正则化
        self.gradient_penalty = GradientPenalty()
        self.gradient_penalty_coefficient: float = 10.

        ## 过渡值区域惩罚
        self.Transition_loss = TransitionDistanceLoss()
        self.Transition_initial_weight = 1.0  # 初始权重
        self.Transition_max_weight = 10.0  # 最大权重
        self.Transition_start_step = 5000  # 开始计算过渡损失的步数
        self.Transition_rampup_steps = 10000  # 权重从初始到最大的递增步数

        # 优化器
        self.generator_optimizer = None
        self.discriminator_optimizer = None
        self.mapping_network_optimizer = None

        # 数据加载器
        self.loader = None

        # 3D训练超参数
        self.batch_size: int = args.batch_size
        self.d_latent: int = 128
        self.volume_size: int = args.volume_size
        self.mapping_network_layers: int = 4
        self.learning_rate: float = 1e-3
        self.mapping_network_learning_rate: float = 1e-5
        self.gradient_accumulate_steps: int = 1  # 增加梯度累积 可减轻显存压力
        # beta1=0 表示不累积历史梯度的一阶矩信息，beta2=0.99表示二阶矩估计会保留 99% 的历史信息
        self.adam_betas: Tuple[float, float] = (0, 0.99)
        self.style_mixing_prob: float = 0
        self.training_steps: int = args.training_steps
        self.n_gen_blocks: int = 0

        ### ADA相关参数
        self.ada_target: float = 0.6  # 目标信号值
        self.ada_adjust_interval: int = 5  # 调整间隔（步）
        self.ada_adjust_rate: float = 0.03  # 调整速率

        # 延迟正则化参数
        self.lazy_gradient_penalty_interval: int = 100

        # 日志与保存参数
        self.log_generated_interval: int = 50
        self.save_checkpoint_interval: int = 50

        ##  指标检查 参数
        self.metric_check_start_step: int = 13000  # 1.8w步开始检查
        self.metric_check_interval: int = 50  # 每100步检查一次
        self.metric_check_sample_num: int = 200  # 每次检查生成200个样本

        ## 数据集路径
        self.dataset_path: str = args.dataset_path
        self.val_dataset_path: str = args.val_dataset_path
        self.num_workers: int = args.num_workers
        self.pin_memory: bool = args.pin_memory

        ## 检查点路径（用于续训）
        self.resume: bool = args.resume
        self.resume_checkpoint_path: str = args.resume_checkpoint_path

        ## 输出路径
        self.results_dir: str = args.results_dir
        self.losses_dir = os.path.join(self.results_dir, 'losses')
        self.slices_dir = os.path.join(self.results_dir, 'slices')
        self.checkpoints_dir = os.path.join(self.results_dir, 'checkpoints')
        self.loss_csv_path = os.path.join(self.losses_dir, 'training_losses.csv')
        self._prepare_output_dirs()

        # 指标检查器
        self.metric_checker = None

    def _prepare_output_dirs(self):
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.losses_dir, exist_ok=True)
        os.makedirs(self.slices_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

    def _find_latest_checkpoint(self):
        checkpoint_paths = list(Path(self.checkpoints_dir).glob('ckpt_step_*.pth'))
        if not checkpoint_paths:
            return None

        def _extract_step(path: Path):
            try:
                return int(path.stem.rsplit('_', 1)[-1])
            except ValueError:
                return -1

        return str(max(checkpoint_paths, key=_extract_step))

    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点，用于续训"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.mapping_network.load_state_dict(checkpoint['mapping_network'])
        self.generator_optimizer.load_state_dict(checkpoint['g_optim'])
        self.discriminator_optimizer.load_state_dict(checkpoint['d_optim'])
        self.mapping_network_optimizer.load_state_dict(checkpoint['m_optim'])

        ### 加载ADA相关状态
        if 'augment_pipe_state' in checkpoint:
            self.augment_pipe.load_state_dict(checkpoint['augment_pipe_state'])

        start_step = checkpoint['step']
        print(f"Loaded checkpoint from {checkpoint_path}, step {start_step}. Resuming training...")
        return start_step

    def get_w(self, batch_size: int):
        """获取潜在向量w"""
        if torch.rand(()).item() < self.style_mixing_prob:
            cross_over_point = int(torch.rand(()).item() * self.n_gen_blocks)
            z2 = torch.randn(batch_size, self.d_latent).to(self.device)
            z1 = torch.randn(batch_size, self.d_latent).to(self.device)

            w1 = self.mapping_network(z1)
            w2 = self.mapping_network(z2)

            w1 = w1[None, :, :].expand(cross_over_point, -1, -1)
            w2 = w2[None, :, :].expand(self.n_gen_blocks - cross_over_point, -1, -1)
            return torch.cat((w1, w2), dim=0)
        else:
            z = torch.randn(batch_size, self.d_latent).to(self.device)
            w = self.mapping_network(z)
            return w[None, :, :].expand(self.n_gen_blocks, -1, -1)

    def get_3d_noise(self, batch_size: int, zero_noise=False):
        """生成噪声（支持零噪声选项）"""
        noise = []
        resolution = 4

        for i in range(self.n_gen_blocks):
            if i == 0:
                n1 = None
            else:
                if zero_noise:
                    n1 = torch.zeros(batch_size, 1, resolution, resolution, resolution, device=self.device)
                else:
                    n1 = torch.randn(batch_size, 1, resolution, resolution, resolution, device=self.device)

            if zero_noise:
                n2 = torch.zeros(batch_size, 1, resolution, resolution, resolution, device=self.device)
            else:
                n2 = torch.randn(batch_size, 1, resolution, resolution, resolution, device=self.device)

            noise.append((n1, n2))
            resolution *= 2

        return noise

    def generate_volumes(self, batch_size: int, zero_noise=False):
        """生成3D二值化孔隙结构（支持零噪声选项）"""
        w = self.get_w(batch_size)
        noise = self.get_3d_noise(batch_size, zero_noise)
        volumes = self.generator(w, noise)
        return volumes, w

    def save_losses(self, step, disc_loss, gen_loss, gp=None, transition_penalty=None, augment_p=None, ada_signal=None, porosity_mean=None):
        """保存损失值到CSV文件（支持续训/多次运行追加写入）"""
        # 检查文件是否存在且非空
        file_exists = os.path.isfile(self.loss_csv_path)
        file_is_empty = file_exists and os.path.getsize(self.loss_csv_path) == 0

        with open(self.loss_csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 仅当文件不存在或为空时写入表头
            if not file_exists or file_is_empty:
                writer.writerow(
                    ['step', 'discriminator_loss', 'generator_loss', 'gradient_penalty',
                     'transition_penalty', 'augment_p', 'ada_signal', 'porosity_mean'])

            # 写入损失值（处理空值为空白字符串）
            writer.writerow([
                step,
                disc_loss.item() if hasattr(disc_loss, 'item') else disc_loss,
                gen_loss.item() if hasattr(gen_loss, 'item') else gen_loss,
                gp.item() if gp is not None else '',
                transition_penalty.item() if transition_penalty is not None else '',
                augment_p if augment_p is not None else '',
                ada_signal if ada_signal is not None else '',
                round(porosity_mean, 4) if porosity_mean is not None else ''
            ])

    def save_images(self, step, generated_volumes, real_volumes):
        """保存2D切片可视化"""
        # 生成切片可视化（沿z轴中间位置）
        z_mid = self.volume_size // 2
        # 3D体积形状: [batch, channel, depth, height, width]
        real_slices = real_volumes[:, 0, z_mid, :, :].detach().cpu()  # 取中间深度的切片
        gen_slices = generated_volumes[:, 0, z_mid, :, :].detach().cpu()

        # 二值化转换：将 [-1, 1] 转换为 [0, 1]
        gen_slices = (gen_slices + 1) / 2  # [-1, 1] -> [0, 1]

        # 绘制对比图
        plt.figure(figsize=(15, 10))
        # 绘制6张生成模型的切片（上两行，每行3张）
        for i in range(6):
            plt.subplot(3, 3, i + 1)
            plt.imshow(gen_slices[i], cmap='gray', vmin=0, vmax=1)
            plt.title(f'Generated Slice {i + 1} (z={z_mid})')
            plt.axis('off')
        # 绘制3张真实模型的切片（下一行）
        for i in range(3):
            plt.subplot(3, 3, i + 7)
            plt.imshow(real_slices[i], cmap='gray', vmin=0, vmax=1)
            plt.title(f'Real Slice {i + 1} (z={z_mid})')
            plt.axis('off')

        plt.tight_layout()
        slice_path = os.path.join(self.slices_dir, f'slice_step_{step}.png')
        plt.savefig(slice_path, dpi=300)
        plt.close()

    def save_checkpoint(self, step: int):
        """保存模型检查点"""
        checkpoint = {
            'step': step,
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'mapping_network': self.mapping_network.state_dict(),
            'g_optim': self.generator_optimizer.state_dict(),
            'd_optim': self.discriminator_optimizer.state_dict(),
            'm_optim': self.mapping_network_optimizer.state_dict(),
            'augment_pipe_state': self.augment_pipe.state_dict(),  ### 保存ADA状态
        }
        torch.save(checkpoint, os.path.join(self.checkpoints_dir, f'ckpt_step_{step}.pth'))

    def init(self):
        # 初始化数据集和数据加载器
        dataset = PorosityDataset(self.dataset_path, self.volume_size)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=self.pin_memory
        )
        self.loader = cycle_dataloader(dataloader)

        # 计算分辨率的log2值
        log_resolution = int(math.log2(self.volume_size))   # 6

        # 初始化模型
        self.discriminator = Discriminator(log_resolution).to(self.device)
        self.generator = Generator(log_resolution, self.d_latent).to(self.device)
        self.n_gen_blocks = self.generator.n_blocks
        self.mapping_network = MappingNetwork(self.d_latent, self.mapping_network_layers).to(self.device)

        ### 初始化ADA增强管道（仅像素操作）
        self.augment_pipe = AugmentPipe3D(
            ## 几何变化参数
            xflip=0.6,  # X轴翻转概率乘数
            yflip=0.6,  # Y轴翻转概率乘数
            zflip=0.6,  # Z轴翻转概率乘数
            rotate90=0.6,  # 90度旋转概率乘数
            ## 形态学参数
            erode_prob=0.3,  # 腐蚀概率乘数
            dilate_prob=0.3,  # 膨胀概率乘数
            morph_kernel_size=3,  # 形态学核大小
            # 差异化高斯扰动：适中概率，边界扰动稍大
            numeric_perturb_prob=0.5,  # 数值扰动概率乘数
            sigma_boundary=0.09,  # 边界扰动标准差
            sigma_interior=0.04  # 内部扰动标准差
        ).to(self.device)

        # 初始化损失函数
        self.discriminator_loss = DiscriminatorLoss()
        self.generator_loss = GeneratorLoss()

        # 初始化优化器
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.learning_rate,
            betas=self.adam_betas
        )
        self.generator_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.learning_rate,
            betas=self.adam_betas
        )
        self.mapping_network_optimizer = torch.optim.Adam(
            self.mapping_network.parameters(),
            lr=self.mapping_network_learning_rate,
            betas=self.adam_betas
        )

        # 初始化指标检查器
        self.metric_checker = MetricChecker(
            real_dir=self.dataset_path,
            shape=(self.volume_size, self.volume_size, self.volume_size),
            results_dir=self.results_dir
        )

        # 检查是否存在检查点，存在则加载
        start_step = 0
        if self.resume:
            checkpoint_path = self.resume_checkpoint_path or self._find_latest_checkpoint()
            if checkpoint_path and os.path.exists(checkpoint_path):
                start_step = self.load_checkpoint(checkpoint_path)
            else:
                print("Resume requested, but no checkpoint was found. Training will start from step 0.")

        return start_step

    def step(self, idx: int):
        """训练步骤"""
        # 记录当前批次的损失
        current_gp = None
        current_transition_penalty = None
        current_ada_signal = None
        current_augment_p = self.augment_pipe.p.item()

        ############################################ 训练判别器 ############################################
        self.discriminator_optimizer.zero_grad()

        for i in range(self.gradient_accumulate_steps):
            # 生成3D体积
            generated_volumes, _ = self.generate_volumes(self.batch_size)
            ### 对生成图像应用增强（与真实图像相同的增强管道）
            augmented_fake = self.augment_pipe(generated_volumes.detach())
            fake_output = self.discriminator(augmented_fake)

            # 加载真实体积
            real_volumes = next(self.loader).to(self.device)
            ### 对真实图像应用像素增强
            augmented_real = self.augment_pipe(real_volumes)

            if (idx + 1) % self.lazy_gradient_penalty_interval == 0:
                augmented_real.requires_grad_()
            real_output = self.discriminator(augmented_real)

            ### 更新ADA统计信息（使用真实图像的输出）
            self.augment_pipe.update_ada_stats(real_output)

            # 计算判别器损失
            real_loss, fake_loss = self.discriminator_loss(real_output, fake_output)
            disc_loss = real_loss + fake_loss

            # 添加R1梯度惩罚（延迟计算）
            gp = None
            if (idx + 1) % self.lazy_gradient_penalty_interval == 0:
                gp = self.gradient_penalty(augmented_real, real_output)
                current_gp = gp  # 保存梯度惩罚值用于记录
                disc_loss = disc_loss + 0.5 * self.gradient_penalty_coefficient * gp * self.lazy_gradient_penalty_interval

            # 反向传播
            disc_loss.backward()

        self.discriminator_optimizer.step()

        ############################################# 训练生成器 ############################################
        self.generator_optimizer.zero_grad()
        self.mapping_network_optimizer.zero_grad()

        for i in range(self.gradient_accumulate_steps):
            # 生成3D体积
            generated_volumes, w = self.generate_volumes(self.batch_size)
            ### 应用增强
            augmented_generated = self.augment_pipe(generated_volumes)
            fake_output = self.discriminator(augmented_generated)

            # 计算生成器损失
            gen_loss = self.generator_loss(fake_output)

            ## 过渡惩罚值
            if idx >= self.Transition_start_step:
                # 计算原始过渡惩罚值（无权重）
                current_transition_penalty = self.Transition_loss(generated_volumes)

                # 调整权重：10000→20000步线性递增至max_weight，之后保持max_weight
                if idx <= self.Transition_start_step + self.Transition_rampup_steps:
                    # 递增阶段：(当前步-起始步)/递增步数 → 0→1的因子
                    rampup_factor = (idx - self.Transition_start_step) / self.Transition_rampup_steps
                    current_transition_weight = self.Transition_initial_weight + rampup_factor * (
                                self.Transition_max_weight - self.Transition_initial_weight)
                else:
                    # 稳定阶段：保持最大权重
                    current_transition_weight = self.Transition_max_weight

                # 叠加到生成器总损失
                gen_loss += current_transition_penalty * current_transition_weight

            # 反向传播
            gen_loss.backward()

        self.generator_optimizer.step()
        self.mapping_network_optimizer.step()

        ### 调整ADA增强概率
        if (idx + 1) % self.ada_adjust_interval == 0:
            current_ada_signal, current_augment_p = self.augment_pipe.adjust_p(
                target=self.ada_target,
                rate=self.ada_adjust_rate
            )

        # 定期记录和保存
        if (idx + 1) % self.log_generated_interval == 0:
            # 针对预热阶段
            ada_signal_str = f"{current_ada_signal:.4f}" if current_ada_signal is not None else "N/A"
            augment_p_str = f"{current_augment_p:.4f}" if current_augment_p is not None else "0.00"
            print(
                f"Step {idx + 1}: Discriminator Loss = {disc_loss.item():.4f}, Generator Loss = {gen_loss.item():.4f}, "
                f"Ada Signal = {ada_signal_str}, Augment P = {augment_p_str}")

            # 保存损失值
            self.save_losses(idx + 1, disc_loss, gen_loss, current_gp, current_transition_penalty,
                             current_augment_p, current_ada_signal)
            # 保存生成的图像
            self.save_images(idx + 1, generated_volumes, real_volumes)

        # 保存检查点
        if (idx + 1) % self.save_checkpoint_interval == 0:
            self.save_checkpoint(idx + 1)

        # 检查指标（从1.8w步开始，每100步检查一次）
        current_step = idx + 1
        if (current_step >= self.metric_check_start_step and
                (current_step - self.metric_check_start_step) % self.metric_check_interval == 0):
            self.metric_checker.check(
                generator=self,  # 传递当前Configs实例作为生成器
                step=current_step,
                num_samples=self.metric_check_sample_num
            )

    def train(self):
        """训练主循环"""
        print(f"开始训练，设备: {self.device}")
        print(f"Training steps: {self.training_steps}")
        print(f"体积尺寸: {self.volume_size}x{self.volume_size}x{self.volume_size}")
        print(f"批次大小: {self.batch_size}, 总步数: {self.training_steps}")
        print(f"Dataset path: {self.dataset_path}")
        print(f"Results dir: {self.results_dir}")
        print(f"ADA Target: {self.ada_target}, Adjust Interval: {self.ada_adjust_interval}")
        print(f"指标检查: 从 {self.metric_check_start_step} 步开始，每 {self.metric_check_interval} 步一次")

        start_step = self.init()

        for i in range(start_step, self.training_steps):
            start_time = time.time()
            self.step(i)

            # 每25步打印进度
            if (i + 1) % 25 == 0:
                elapsed = time.time() - start_time
                time_consuming = elapsed / 60
                print(f"进度: {i + 1}/{self.training_steps} | "
                      f"速度: {time_consuming:.2f} min/步"
                      )


def parse_args():
    parser = argparse.ArgumentParser(description="Train CY-PNMGAN on 3D porous volumes.")
    parser.add_argument('--dataset-path', required=True, help='Path to training .raw files.')
    parser.add_argument('--val-dataset-path', default='', help='Optional validation dataset path.')
    parser.add_argument('--results-dir', default='results', help='Directory for checkpoints, logs, and figures.')
    parser.add_argument('--resume', action='store_true', help='Resume training from a checkpoint.')
    parser.add_argument(
        '--resume-checkpoint-path',
        default='',
        help='Checkpoint path for resume. If empty and --resume is set, latest checkpoint in results dir is used.'
    )
    parser.add_argument('--training-steps', type=int, default=500_000, help='Total training steps.')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for training.')
    parser.add_argument('--num-workers', type=int, default=8, help='DataLoader worker count.')
    parser.add_argument('--volume-size', type=int, default=64, help='Input volume edge length.')
    parser.set_defaults(pin_memory=torch.cuda.is_available())
    parser.add_argument('--pin-memory', dest='pin_memory', action='store_true', help='Enable DataLoader pin_memory.')
    parser.add_argument('--no-pin-memory', dest='pin_memory', action='store_false', help='Disable pin_memory.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = Configs(args)
    config.train()
