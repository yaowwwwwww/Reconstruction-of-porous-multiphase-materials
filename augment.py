# 3D版本：删除 整数平移

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------- 1..基础几何变换核心函数 --------------------------
# 辅助函数：构建变换矩阵  将输入的多行元素转换为一个 PyTorch 张量矩阵
def matrix(*rows, device=None):
    # 检查所有行的长度是否相同
    assert all(len(row) == len(rows[0]) for row in rows)
    # 将所有行的元素 “平铺” 为一个一维列表   外层循环 for row in rows   内层循环 for x in row
    elems = [x for row in rows for x in row]
    # 筛选出参考Tensor张量
    ref = [x for x in elems if isinstance(x, torch.Tensor)]
    if len(ref) == 0:
        return torch.tensor(rows, device=device, dtype=torch.float32)
    # 确定参考张量的设备和形状（批次化形状，如 [16,1,1]）
    ref_tensor = ref[0]
    device = ref_tensor.device if device is None else device
    ref_shape = ref_tensor.shape  # 关键：获取批次化形状
    # 统一所有元素的形状：非张量→张量，标量张量→批次化张量
    processed_elems = []
    for x in elems:
        if isinstance(x, torch.Tensor):
            # 若为张量，扩展到参考形状（保持值不变）
            # expand_as 不复制数据，仅改变视图，效率高
            expanded = x.expand_as(ref_tensor)
        else:
            # 若为常数，先转为张量，再扩展到参考形状
            x_tensor = torch.tensor(x, device=device, dtype=torch.float32)
            expanded = x_tensor.expand(ref_shape)
        processed_elems.append(expanded)


    # 先堆叠为 [B,1,1,16]（假设16个元素），再重塑为 [B,1,1,4,4]
    matrix_tensor = torch.stack(processed_elems, dim=-1).reshape(ref_shape + (len(rows), -1))
    # 用 squeeze 移除所有 size=1 的维度，保留批量维度（若批量维度为1也不影响）
    matrix_tensor = matrix_tensor.squeeze(dim=tuple(range(1, len(ref_shape))))
    # 最终形状：[N, 行数, 列数]（例如 [B,4,4]，B为批量大小）
    return matrix_tensor


# 3D 平移矩阵 (齐次坐标: (x, y, z, 1))
def translate3d(tx, ty, tz, **kwargs):
    return matrix(
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1], **kwargs)

# 3D 缩放矩阵
def scale3d(sx, sy, sz, **kwargs):
    return matrix(
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1], **kwargs)

# 3D X轴旋转矩阵
def rotate_x(theta, **kwargs):
    return matrix(
        [1, 0, 0, 0],
        [0, torch.cos(theta), -torch.sin(theta), 0],
        [0, torch.sin(theta), torch.cos(theta), 0],
        [0, 0, 0, 1], **kwargs)


# 3D Y轴旋转矩阵
def rotate_y(theta, **kwargs):
    return matrix(
        [torch.cos(theta), 0, torch.sin(theta), 0],
        [0, 1, 0, 0],
        [-torch.sin(theta), 0, torch.cos(theta), 0],
        [0, 0, 0, 1], **kwargs)


# 3D Z轴旋转矩阵
def rotate_z(theta, **kwargs):
    return matrix(
        [torch.cos(theta), -torch.sin(theta), 0, 0],
        [torch.sin(theta), torch.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1], **kwargs)


def translate3d_inv(tx, ty, tz, **kwargs):
    return translate3d(-tx, -ty, -tz, **kwargs)


def scale3d_inv(sx, sy, sz, **kwargs):
    return scale3d(1 / sx, 1 / sy, 1 / sz, **kwargs)


def rotate_x_inv(theta, **kwargs):
    return rotate_x(-theta, **kwargs)


def rotate_y_inv(theta, **kwargs):
    return rotate_y(-theta, **kwargs)


def rotate_z_inv(theta, **kwargs):
    return rotate_z(-theta, **kwargs)

# -------------------------- 2.2D形态学操作核心函数 --------------------------
def get_morph_kernel_2d(kernel_size=3, device=None):
    """创建2D十字形结构元素（适配切片处理，与原3D核的切片逻辑一致）"""
    kernel = torch.zeros((kernel_size, kernel_size), device=device)
    center = kernel_size // 2
    # 水平中线（对应3D核的x方向切片）
    kernel[center, :] = 1.0
    # 垂直中线（对应3D核的y方向切片）
    kernel[:, center] = 1.0
    return kernel


def erode_cross_2d(x, kernel):
    """2D十字形腐蚀（单切片处理，复用原3D操作的2D核心逻辑）"""
    padding = kernel.shape[0] // 2
    # 2D反射边界填充（匹配原3D的边界模式）
    x_pad = F.pad(x, (padding, padding, padding, padding), mode='reflect')

    # 提取局部区域（仅处理2D维度）
    unfold = F.unfold(x_pad, kernel_size=kernel.shape, stride=1)
    batch_size, channels, height, width = x.shape
    k = kernel.numel()
    unfold = unfold.view(batch_size, channels, k, height, width)

    # 核筛选与最小值计算（与原3D逻辑完全一致）
    kernel_flat = kernel.flatten()
    valid_mask = (kernel_flat == 1.0).float()
    valid_pixels = unfold * valid_mask[None, None, :, None, None]
    valid_pixels = valid_pixels.masked_fill(valid_mask[None, None, :, None, None] == 0, float('inf'))
    return torch.min(valid_pixels, dim=2)[0]


def dilate_cross_2d(x, kernel):
    """2D十字形膨胀（单切片处理，复用原3D操作的2D核心逻辑）"""
    padding = kernel.shape[0] // 2
    x_pad = F.pad(x, (padding, padding, padding, padding), mode='reflect')

    # 提取局部区域（仅处理2D维度）
    unfold = F.unfold(x_pad, kernel_size=kernel.shape, stride=1)
    batch_size, channels, height, width = x.shape
    k = kernel.numel()
    unfold = unfold.view(batch_size, channels, k, height, width)

    # 核筛选与最大值计算（与原3D逻辑完全一致）
    kernel_flat = kernel.flatten()
    valid_mask = (kernel_flat == 1.0).float()
    valid_pixels = unfold * valid_mask[None, None, :, None, None]
    valid_pixels = valid_pixels.masked_fill(valid_mask[None, None, :, None, None] == 0, float('-inf'))
    return torch.max(valid_pixels, dim=2)[0]


# -------------------------- 3.差异化高斯扰动 --------------------------
# 1.获得孔隙边界
def get_pore_boundary_mask_3d(volumes: torch.Tensor, kernel_size=3) -> torch.Tensor:
    """
    3D孔隙边界体素识别：通过3×3×3邻域检查，标记孔隙-固体接触的边界体素
    输入: volumes [B, C, D, H, W]，体素值∈[-1,1]（孔隙≤0，固体>0）
    输出: boundary_mask [B, C, D, H, W]，1=边界体素，0=内部体素
    """
    batch_size, channels, depth, height, width = volumes.shape
    device = volumes.device
    padding = kernel_size // 2

    # 1. 生成3D十字形核（仅检查中心体素的6个相邻体素：上下/左右/前后）
    kernel = torch.zeros((1, 1, kernel_size, kernel_size, kernel_size), device=device)
    center = kernel_size // 2
    # 沿深度(D)、高度(H)、宽度(W)的中线（对应3D空间的6个相邻方向）
    kernel[0, 0, center, center, :] = 1.0  # W方向
    kernel[0, 0, center, :, center] = 1.0  # H方向
    kernel[0, 0, :, center, center] = 1.0  # D方向

    # 2. 对孔隙体素进行3D邻域展开（仅关注孔隙区域的邻域是否有固体）
    # 先创建孔隙掩码：1=孔隙体素(≤0)，0=固体体素(>0)
    pore_mask = (volumes <= 0).float()
    # 对孔隙掩码进行3D卷积，统计邻域内固体体素的数量（固体=1-pore_mask）
    solid_in_neighbor = F.conv3d(
        1 - pore_mask,  # 输入：1=固体，0=孔隙
        kernel,         # 3D十字形核
        padding=padding,
        groups=channels  # 按通道分组卷积，避免跨通道干扰
    )

    # 3. 边界体素判定：孔隙体素的邻域内存在固体体素（solid_in_neighbor > 0）
    boundary_mask = (pore_mask * (solid_in_neighbor > 0)).float()
    return boundary_mask

# 2.高斯扰动  边界体素扰动大，内部体素扰动小
def gaussian_perturb_3d(
    volumes: torch.Tensor,
    boundary_mask: torch.Tensor,
    sigma_boundary: float = 0.09,  # 孔隙边界高斯标准差（非均匀起伏显著）
    sigma_interior: float = 0.04    # 孔隙内部高斯标准差（起伏平缓）
) -> torch.Tensor:

    device = volumes.device

    # 1. 生成基础高斯噪声（均值=0，标准差=1） 95.4% 的概率落在[-2, 2]
    gaussian_noise = torch.randn_like(volumes, device=device)

    # 2. 差异化标准差赋值：先创建标准差张量：默认用内部标准差，边界位置替换为边界标准差
    sigma_tensor = sigma_interior * torch.ones_like(volumes, device=device)
    sigma_tensor = torch.where(
        boundary_mask == 1,  # 边界体素
        sigma_boundary * torch.ones_like(sigma_tensor),
        sigma_tensor          # 内部体素
    )

    # 3. 缩放噪声到目标标准差，并截断极端值（保留[-2σ, 2σ]，覆盖95.4%有效范围）
    scaled_noise = gaussian_noise * sigma_tensor
    scaled_noise = torch.clamp(scaled_noise, min=-2*sigma_tensor, max=2*sigma_tensor)

    # 4. 施加噪声并执行类别不变约束
    perturbed = volumes + scaled_noise
    # 孔隙体素（原≤0）：裁剪到[-1, 0]，避免变为固体
    # 固体体素（原>0）：裁剪到[0, 1]，避免变为孔隙（天然材料中固体界面扰动需求低，简化处理）
    perturbed = torch.where(
        volumes <= 0,
        torch.clamp(perturbed, min=-1.0, max=0.0),
        torch.clamp(perturbed, min=0.0, max=1.0)
    )

    return perturbed

# 3.数值扰动入口函数
def numeric_perturb_3d(
    volumes: torch.Tensor,
    sigma_boundary: float = 0.09,
    sigma_interior: float = 0.04
) -> torch.Tensor:
    """
    3D数值扰动入口：先识别孔隙边界，再执行差异化高斯扰动
    输入: volumes [B, C, D, H, W]，体素值∈[-1,1]
    输出: perturbed_volumes [B, C, D, H, W]，扰动后体素值
    """
    # 1. 识别孔隙边界体素
    boundary_mask = get_pore_boundary_mask_3d(volumes, kernel_size=3)
    # 2. 执行差异化高斯扰动
    perturbed = gaussian_perturb_3d(
        volumes=volumes,
        boundary_mask=boundary_mask,
        sigma_boundary=sigma_boundary,
        sigma_interior=sigma_interior
    )
    return perturbed


#######################################################################################
class AugmentPipe3D(nn.Module):
    def __init__(self,
                 # 3D几何操作
                 xflip=0.6,  # X轴翻转概率乘数
                 yflip=0.6,  # Y轴翻转概率乘数
                 zflip=0.6,  # Z轴翻转概率乘数
                 rotate90=0.6,  # 90度旋转概率乘数
                 # 形态学
                 erode_prob=0.3,  # 腐蚀概率乘数
                 dilate_prob=0.3,  # 膨胀概率乘数
                 morph_kernel_size=3,  # 形态学核大小
                 # 差异化高斯扰动（新增参数）
                 numeric_perturb_prob: float = 0.4,  # 数值扰动概率乘数
                 sigma_boundary: float = 0.09,       # 孔隙边界高斯标准差
                 sigma_interior: float = 0.04        # 孔隙内部高斯标准差
                 ):
        super().__init__()
        # 增强概率的全局乘数，由ADA动态调整
        self.register_buffer('p', torch.tensor(0.0))

        # 3D操作相关参数
        self.xflip = float(xflip)  # X轴翻转概率乘数
        self.yflip = float(yflip)  # Y轴翻转概率乘数
        self.zflip = float(zflip)  # Z轴翻转概率乘数
        self.rotate90 = float(rotate90)  # 90度旋转概率乘数

        # 用于统计判别器对真实图像的信号
        self.register_buffer('ada_stats', torch.tensor(0.0))
        self.register_buffer('ada_steps', torch.tensor(0))

        # 形态学 相关参数
        self.erode_prob = erode_prob
        self.dilate_prob = dilate_prob
        self.morph_kernel_size = morph_kernel_size
        self.morph_kernel = None  # 延迟初始化核

        # 差异化高斯扰动参数（替换原max_perturb）
        self.numeric_perturb_prob = numeric_perturb_prob
        self.sigma_boundary = sigma_boundary  # 适配天然材料界面非均匀起伏
        self.sigma_interior = sigma_interior  # 适配孔隙内部平缓特性

    def update_ada_stats(self, real_pred):
        """更新ADA统计信息：收集判别器对真实图像的输出信号"""
        # 取判别器输出的符号作为信号 rt sign:[-1/1]   signal = 2P-1   P为 reals中D评分>0 的占比，上限ada_target
        signal = torch.sign(real_pred).mean().detach()
        # 平滑判别器对真实图像的输出信号: 保留 99% 的历史信息 + signal 乘以 0.01
        self.ada_stats.mul_(0.99).add_(signal, alpha=0.01)
        self.ada_steps.add_(1)

    def adjust_p(self, target=0.6, rate=0.001):
        """根据目标值调整增强概率p"""
        if self.ada_steps.item() < 10:  # 预热步骤，不调整
            return None, self.p.item()
        # 计算当前信号与目标的偏差
        current_signal = self.ada_stats.item()
        delta = (target - current_signal) * rate

        # 更新增强概率，限制在[0, 1]范围内
        new_p = self.p.item() - delta
        new_p = max(0.0, min(1.0, new_p))
        self.p.copy_(torch.tensor(new_p))

        return current_signal, new_p

    def forward(self, volumes):
        """
        输入: 3D体数据，形状为 [B, C, D, H, W] (D=深度, H=高度, W=宽度)
        输出: 增强后的3D体数据，保持64x64x64尺寸
        """
        assert isinstance(volumes, torch.Tensor) and volumes.ndim == 5, "输入必须是5D张量 [B, C, D, H, W]"
        batch_size, num_channels, depth, height, width = volumes.shape
        device = volumes.device

        # 初始化逆变换矩阵：4x4单位矩阵
        I_4 = torch.eye(4, device=device)
        G_inv = I_4.expand(batch_size, 4, 4)

        # X轴翻转：概率为 (xflip * p)
        if self.xflip > 0:
            flip_mask = (torch.rand([batch_size], device=device) < self.xflip * self.p).float()
            scale = 1 - 2 * flip_mask[:, None, None]  # 1或-1
            flip_matrix = scale3d_inv(scale, torch.ones_like(scale), torch.ones_like(scale), device=device)
            G_inv = G_inv @ flip_matrix

        # Y轴翻转：概率为 (yflip * p)
        if self.yflip > 0:
            flip_mask = (torch.rand([batch_size], device=device) < self.yflip * self.p).float()
            scale = 1 - 2 * flip_mask[:, None, None]  # 1或-1
            flip_matrix = scale3d_inv(torch.ones_like(scale), scale, torch.ones_like(scale), device=device)
            G_inv = G_inv @ flip_matrix

        # Z轴翻转：概率为 (zflip * p)
        if self.zflip > 0:
            flip_mask = (torch.rand([batch_size], device=device) < self.zflip * self.p).float()
            scale = 1 - 2 * flip_mask[:, None, None]  # 1或-1
            flip_matrix = scale3d_inv(torch.ones_like(scale), torch.ones_like(scale), scale, device=device)
            G_inv = G_inv @ flip_matrix

        # 90度旋转：在三个轴上随机选择一个进行旋转
        if self.rotate90 > 0:
            rotate_mask = (torch.rand([batch_size], device=device) < self.rotate90 * self.p)
            rotations = torch.where(rotate_mask,
                                    torch.randint(1, 4, [batch_size], device=device),
                                    torch.zeros([batch_size], device=device, dtype=torch.int32))

            # 随机选择旋转轴
            axes = torch.randint(0, 3, [batch_size], device=device)  # 0:X, 1:Y, 2:Z

            # 对每个样本应用相应的旋转
            for i in range(batch_size):
                if rotate_mask[i]:
                    theta = rotations[i].float() * (torch.pi / 2)
                    if axes[i] == 0:  # X轴旋转
                        rot_matrix = rotate_x_inv(theta, device=device)
                    elif axes[i] == 1:  # Y轴旋转
                        rot_matrix = rotate_y_inv(theta, device=device)
                    else:  # Z轴旋转
                        rot_matrix = rotate_z_inv(theta, device=device)
                    G_inv[i] = G_inv[i] @ rot_matrix

        ############################# 1.执行基础 几何变换 #############################
        if not torch.allclose(G_inv, I_4.expand_as(G_inv), atol=1e-6):
            # 生成3D坐标映射网格
            grid = F.affine_grid(G_inv[:, :3, :], volumes.shape, align_corners=False)
            # 应用3D变换
            volumes = F.grid_sample(volumes, grid, mode='bilinear', padding_mode='reflection', align_corners=False)

        ############################# 2. 3D→2D切片形态学操作#############################
        if (self.erode_prob > 0 or self.dilate_prob > 0) and self.p > 0:
            # 初始化2D十字形核（适配切片处理）
            if self.morph_kernel is None:
                self.morph_kernel = get_morph_kernel_2d(self.morph_kernel_size, device)

            # 1. 筛选需要应用形态学操作的样本
            total_morph_prob = (self.erode_prob + self.dilate_prob) * self.p
            morph_mask = torch.rand(batch_size, device=device) < total_morph_prob  # [B]，标记需处理的样本

            if morph_mask.any():
                # 计算腐蚀/膨胀的概率权重
                total_prob = self.erode_prob + self.dilate_prob
                erode_weight = self.erode_prob / total_prob if total_prob > 0 else 0.5

                # 创建输出副本
                volumes_out = volumes.clone()

                # 2. 为每个需处理的样本确定操作类型（整个3D体统一操作）
                # 为每个样本生成一次操作类型，而非每个切片
                operation_type = torch.where(
                    torch.rand(batch_size, device=device) < erode_weight,
                    torch.tensor(0, device=device),  # 0表示腐蚀
                    torch.tensor(1, device=device)  # 1表示膨胀
                )

                # 3. 对每个需处理的样本执行操作
                for i in range(batch_size):
                    if morph_mask[i]:
                        # 获取当前样本的操作类型（整个3D体统一）
                        is_erode = operation_type[i] == 0

                        # 提取单个样本的3D数据
                        single_volume = volumes[i:i + 1]
                        batch, ch, d, h, w = single_volume.shape
                        processed_slices = []

                        # 4. 对所有切片执行相同操作
                        for z in range(d):
                            # 提取2D切片
                            slice_2d = single_volume[:, :, z:z + 1, :, :].squeeze(2)

                            # 对当前切片执行与整个3D体相同的操作
                            if is_erode:
                                processed_slice = erode_cross_2d(slice_2d, self.morph_kernel)
                            else:
                                processed_slice = dilate_cross_2d(slice_2d, self.morph_kernel)

                            # 恢复深度维度并收集
                            processed_slices.append(processed_slice.unsqueeze(2))

                        # 5. 合并所有切片为3D数据
                        merged_3d = torch.cat(processed_slices, dim=2)
                        volumes_out[i:i + 1] = merged_3d

                # 更新为处理后的结果
                volumes = volumes_out

        ############################# 3. 核心：差异化高斯扰动 #############################
        if self.numeric_perturb_prob > 0 and self.p > 0:
            current_numeric_prob = self.numeric_perturb_prob * self.p
            numeric_mask = (torch.rand([batch_size], device=device) < current_numeric_prob).float()
            if numeric_mask.any():
                # 扩展掩码到3D维度：[B] → [B, C, D, H, W]
                numeric_mask_3d = numeric_mask[:, None, None, None, None].expand_as(volumes)
                # 执行差异化高斯扰动
                perturbed_volumes = numeric_perturb_3d(
                    volumes=volumes,
                    sigma_boundary=self.sigma_boundary,
                    sigma_interior=self.sigma_interior
                    )
                # 合并结果：需扰动的样本用perturbed_volumes，否则用原volumes
                volumes = volumes * (1 - numeric_mask_3d) + perturbed_volumes * numeric_mask_3d

        return volumes