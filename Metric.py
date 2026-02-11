import os
import numpy as np
import porespy as ps
import openpnm as op
from scipy.interpolate import make_interp_spline
from concurrent.futures import ProcessPoolExecutor, as_completed

# 设置并行进程数
MAX_WORKERS = os.cpu_count() or 4

def process_file(file_path, shape=(64, 64, 64), dtype=np.uint8):
    """处理单个3D .raw文件，提取孔隙网络与属性"""
    # 读取3D体数据并reshape为64×64×64
    vol_3d = np.fromfile(file_path, dtype=dtype).reshape(shape)
    vol_3d_inverted = vol_3d == 0  # 孔隙为True，固体为False

    # 提取3D孔隙网络
    snow = ps.networks.snow2(vol_3d_inverted, boundary_width=0, voxel_size=1)
    pn = op.io.network_from_porespy(snow.network)

    # 统计孔隙连通性
    throat_conns = pn['throat.conns']
    num_pores = pn['pore.coords'].shape[0]
    all_pores = np.arange(num_pores)

    conn_counts = np.zeros(num_pores, dtype=int)
    # 统计 每个孔隙 通过喉道连接到其他孔隙的数量
    for conn in throat_conns:
        # 每个喉道连接的两个孔隙 conn[0] 和 conn[1]
        conn_counts[conn[0]] += 1
        conn_counts[conn[1]] += 1
    # 连通孔隙数
    connected_pores = all_pores[conn_counts > 0]

    # 提取3D孔隙/喉道属性
    pore_volume = pn['pore.volume'][connected_pores]
    pore_surface_area = pn['pore.surface_area'][connected_pores]
    pore_equivalent_diameter = pn['pore.equivalent_diameter'][connected_pores]

    throat_length = pn['throat.total_length']
    throat_surface_area = pn['throat.perimeter'] * throat_length
    throat_equivalent_diameter = pn['throat.equivalent_diameter']

    return {
        'pore_volume': pore_volume,
        'pore_surface_area': pore_surface_area,
        'pore_equivalent_diameter': pore_equivalent_diameter,
        'throat_length': throat_length,
        'throat_surface_area': throat_surface_area,
        'throat_equivalent_diameter': throat_equivalent_diameter,
        'file_name': os.path.basename(file_path),
        'vol_3d_inverted': vol_3d_inverted,
        'vol_3d': vol_3d
    }


def calculate_porosity_and_stats(pn, shape=(64, 64, 64), voxel_size=1):
    """计算3D体数据的孔隙率、喉道数量等统计指标"""
    # 3D体积计算
    total_pore_volume = np.sum(pn['pore.volume']) * (voxel_size ** 3)
    total_volume = np.prod(shape) * (voxel_size ** 3)
    total_porosity = total_pore_volume / total_volume

    # 连通孔隙率计算
    throat_conns = pn['throat.conns']
    num_pores = pn['pore.coords'].shape[0]
    all_pores = np.arange(num_pores)

    conn_counts = np.zeros(num_pores, dtype=int)
    for conn in throat_conns:
        conn_counts[conn[0]] += 1
        conn_counts[conn[1]] += 1

    connected_pores = all_pores[conn_counts > 0]
    connected_pore_volume = np.sum(pn['pore.volume'][connected_pores]) * (voxel_size ** 3)
    connected_porosity = connected_pore_volume / total_volume

    num_throats = pn['throat.conns'].shape[0]

    return total_porosity, connected_porosity, num_pores, len(connected_pores), num_throats


def process_stats_file(args):
    """单个文件的统计计算函数（用于并行）"""
    file_path, shape = args
    try:
        file_name = os.path.basename(file_path)
        # 处理3D体数据
        result = process_file(file_path, shape)
        # 重新构建3D网络
        vol_3d = np.fromfile(file_path, np.uint8).reshape(shape)
        vol_3d_inverted = vol_3d == 0
        snow = ps.networks.snow2(vol_3d_inverted, boundary_width=0, voxel_size=1)
        pn = op.io.network_from_porespy(snow.network)
        stats = calculate_porosity_and_stats(pn, shape)
        return file_name, stats
    except Exception as e:
        print(f"文件 {file_path} 处理出错: {e}")
        return None, None


def calculate_stats_for_folder(folder_path, shape=(64, 64, 64)):
    """计算文件夹内所有3D样本的统计指标"""
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.raw')]
    all_stats = []
    all_file_names = []

    # 并行处理文件
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交任务：传递(file_path, shape)元组作为参数
        futures = {executor.submit(process_stats_file, (path, shape)): path for path in file_paths}
        # 获取结果
        for future in as_completed(futures):
            file_name, stats = future.result()
            if file_name and stats:  # 过滤处理失败的文件
                all_file_names.append(file_name)
                all_stats.append(stats)

    # 计算平均值
    avg_total_porosity = np.mean([s[0] for s in all_stats]) if all_stats else 0
    avg_connected_porosity = np.mean([s[1] for s in all_stats]) if all_stats else 0
    avg_total_pores = np.mean([s[2] for s in all_stats]) if all_stats else 0
    avg_connected_pores = np.mean([s[3] for s in all_stats]) if all_stats else 0
    avg_throat_numbers = np.mean([s[4] for s in all_stats]) if all_stats else 0

    return all_file_names, all_stats, (avg_total_porosity, avg_connected_porosity,
                                       avg_total_pores, avg_connected_pores, avg_throat_numbers)


def process_tpc_file(args):
    """单个文件的两点概率函数计算（用于并行）"""
    file_path, shape, bins, voxel_size = args
    try:
        # 读取3D体数据
        vol_3d = np.fromfile(file_path, dtype=np.uint8).reshape(shape)
        vol_3d_inverted = vol_3d == 0  # 孔隙为True
        tpc_data = ps.metrics.two_point_correlation(
            im=vol_3d_inverted,
            bins=bins,
            voxel_size=voxel_size
        )
        return tpc_data.distance, tpc_data.pdf
    except Exception as e:
        print(f"文件 {file_path} TPC计算出错: {e}")
        return None, None


def get_all_tpc_data(folder_path, shape=(64, 64, 64), bins=120, voxel_size=1):
    """获取文件夹内所有3D样本的两点概率函数数据"""
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.raw')]
    all_distances = []
    all_pdfs = []

    # 并行处理TPC计算
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交任务：传递(file_path, shape, bins, voxel_size)参数
        futures = {
            executor.submit(process_tpc_file, (path, shape, bins, voxel_size)): path
            for path in file_paths
        }

        # 获取结果
        for future in as_completed(futures):
            dist, pdf = future.result()
            if dist is not None and pdf is not None:
                all_distances.append(dist)
                all_pdfs.append(pdf)

    return all_distances, all_pdfs


def calculate_average_tpc(all_distances, all_pdfs):
    """计算所有样本两点概率函数的平均值"""
    if not all_distances or not all_pdfs:
        return None, None

    # 统一距离范围（取所有样本距离的并集）
    min_dist = np.min([d.min() for d in all_distances])
    max_dist = np.max([d.max() for d in all_distances])
    common_dist = np.linspace(min_dist, max_dist, 500)  # 统一插值点

    avg_pdf = np.zeros_like(common_dist, dtype=float)
    count = 0

    for dist, pdf in zip(all_distances, all_pdfs):
        # 对每个样本的PDF进行插值到common_dist
        try:
            spline = make_interp_spline(dist, pdf, k=3)
            interp_pdf = spline(common_dist)
            avg_pdf += interp_pdf
            count += 1
        except Exception as e:
            print(f"插值处理时出错: {e}")
            continue

    if count == 0:
        return None, None

    avg_pdf /= count  # 求均值
    return common_dist, avg_pdf



def align_tpc_data(real_dist, real_pdf, fake_dist, fake_pdf):
    """对齐真假样本的TPC距离尺度（兼容旧版本scipy）"""
    # 取距离范围的并集
    min_dist = min(real_dist.min(), fake_dist.min())
    max_dist = max(real_dist.max(), fake_dist.max())
    common_dist = np.linspace(min_dist, max_dist, 500)  # 生成统一距离点

    # 对两者进行插值（移除extrapolate参数，兼容旧版本scipy）
    try:
        # 旧版本scipy不支持extrapolate，改用默认边界条件
        real_spline = make_interp_spline(real_dist, real_pdf, k=3)
        fake_spline = make_interp_spline(fake_dist, fake_pdf, k=3)

        # 计算插值结果
        aligned_real = real_spline(common_dist)
        aligned_fake = fake_spline(common_dist)

        # 手动处理超出原始范围的点（设置为0，模拟extrapolate='zeros'）
        real_mask = (common_dist < real_dist.min()) | (common_dist > real_dist.max())
        fake_mask = (common_dist < fake_dist.min()) | (common_dist > fake_dist.max())
        aligned_real[real_mask] = 0
        aligned_fake[fake_mask] = 0

        return common_dist, aligned_real, aligned_fake
    except Exception as e:
        print(f"TPC数据对齐失败: {str(e)}")
        return None, None, None

def calculate_tpc_relative_error(real_pdf, fake_pdf, epsilon=1e-8):
    """计算真假样本TPC的相对误差"""
    if len(real_pdf) != len(fake_pdf):
        print("错误：真实样本与输出样本的TPC长度不匹配")
        return None

    # 避免除以零
    mask = real_pdf < epsilon
    relative_error = np.full_like(real_pdf, np.nan, dtype=float)

    # 有效区域计算相对误差（百分比）
    valid_mask = ~mask
    relative_error[valid_mask] = (
            np.abs(fake_pdf[valid_mask] - real_pdf[valid_mask])
            / real_pdf[valid_mask]
            * 100
    )

    return relative_error


# 并行提取孔隙和喉道数据
def extract_pore_throat_data(file_path, shape):
    try:
        result = process_file(file_path, shape)
        return (result['pore_equivalent_diameter'],
                result['throat_equivalent_diameter'])
    except Exception as e:
        print(f"文件 {file_path} 数据提取出错: {e}")
        return None, None

# 并行提取孔隙和喉道数据
def extract_data_parallel(folder_path, shape):
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.raw')]
    pore_data_list = []
    throat_data_list = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(extract_pore_throat_data, path, shape): path for path in file_paths}
        for future in as_completed(futures):
            pore_data, throat_data = future.result()
            if pore_data is not None and len(pore_data) > 0:
                pore_data_list.append(pore_data)
            if throat_data is not None and len(throat_data) > 0:
                throat_data_list.append(throat_data)

    return pore_data_list, throat_data_list