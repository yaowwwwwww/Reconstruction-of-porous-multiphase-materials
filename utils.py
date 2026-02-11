import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import numpy as np
from pathlib import Path
import torch
import torch.utils.data
from torch import nn

def cycle_dataloader(data_loader):
    """
    Infinite loader that recycles the data loader after each epoch
    """
    while True:
        for batch in data_loader:
            yield batch

class PorosityDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, volume_size: int = 64):
        super().__init__()
        self.volume_size = volume_size
        # 收集所有.raw文件路径
        self.paths = list(Path(path).glob('*.raw'))

        if not self.paths:
            raise ValueError(f"未在路径 {path} 找到任何.raw文件！")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        file_path = self.paths[index]

        ## 判断数据来源：真实数据路径还是生成的FID样本路径
        ## 真实数据使用uint8解析，生成数据使用float32解析
        if 'fid_samples' in str(file_path):  # 生成的FID样本
            dtype = np.float32
        else:  # 真实数据
            dtype = np.uint8

        # 根据数据类型解析文件
        with open(file_path, 'rb') as f:
            data = np.fromfile(f, dtype=dtype)

        # 重塑为[1,64,64,64]
        volume = data.reshape(1, self.volume_size, self.volume_size, self.volume_size)
        volume = torch.from_numpy(volume).float()

        # 仅对真实数据进行归一化（0→-1，1→1）
        if dtype == np.uint8:
            volume = (volume - 0.5) / 0.5  # 真实数据uint8需要归一化

        # 生成数据已经是[-1,1]范围，无需再次归一化
        return volume


class DiscriminatorLoss(nn.Module):
    """判别器的Logistic损失实现"""
    def __init__(self):
        super().__init__()

    def forward(self, real_output, fake_output):
        # 真实样本损失：-log(sigmoid(real_output))
        real_loss = -torch.log(torch.sigmoid(real_output) + 1e-8).mean()
        # 生成样本损失：-log(1 - sigmoid(fake_output))
        fake_loss = -torch.log(1 - torch.sigmoid(fake_output) + 1e-8).mean()
        return real_loss, fake_loss


class GeneratorLoss(nn.Module):
    """生成器的Logistic损失实现"""
    def __init__(self):
        super().__init__()

    def forward(self, fake_output):
        # 生成器损失：-log(sigmoid(fake_output))，希望生成样本被判别为真实
        gen_loss = -torch.log(torch.sigmoid(fake_output) + 1e-8).mean()
        return gen_loss


# 过渡区域损失
class TransitionDistanceLoss(nn.Module):
    def __init__(self):
        """仅计算基于距离的原始惩罚值（不含权重），权重在外部控制"""
        super().__init__()

    def forward(self, generated_volumes):
        """
        基于距离[-1,1]的惩罚逻辑：距离越远（越接近0）惩罚越重
        公式：1 - x²（x为生成样本体素值，范围[-1,1]）
        - x=±1时惩罚为0，x=0时惩罚为1（最大）
        """
        # 计算每个体素的惩罚值，取均值作为原始损失
        transition_penalty = torch.mean(1 - generated_volumes ** 2)
        return transition_penalty


class GradientPenalty(nn.Module):
    def forward(self, x: torch.Tensor, d: torch.Tensor):
        batch_size = x.shape[0]
        gradients, *_ = torch.autograd.grad(outputs=d, inputs=x,
                                            grad_outputs=d.new_ones(d.shape),
                                            create_graph=True)
        gradients = gradients.reshape(batch_size, -1)
        norm = gradients.norm(2, dim=-1)
        return torch.mean(norm ** 2)

