import os
import time
import csv
import numpy as np
from Metric import (
    calculate_stats_for_folder, get_all_tpc_data, calculate_average_tpc,
    align_tpc_data, calculate_tpc_relative_error, extract_data_parallel
)
import porespy as ps
# 禁用进度条输出
ps.settings.tqdm['disable'] = True

class MetricChecker:
    def __init__(self, real_dir, shape=(64, 64, 64), results_dir='results'):
        self.real_dir = real_dir
        self.shape = shape
        self.results_dir = results_dir
        self.metrics_order = [
            "step",
            "总孔隙率",
            "连通孔隙率",
            "总孔隙数",
            "连通孔隙数",
            "喉道数量",
            "两点相关函数",
            "孔隙等效直径区间误差",
            "喉道等效直径区间误差",
            "合计1-6（相对误差）",
            "合计7-8（绝对误差）"
        ]
        self.tolerance = {
            "总孔隙率": 2,
            "连通孔隙率": 2,
            "总孔隙数": 4,
            "连通孔隙数": 4,
            "喉道数量": 4,
            "两点相关函数": 5,
            "孔隙等效直径区间误差": 2,
            "喉道等效直径区间误差": 2
        }

        # 缓存真实样本的统计数据
        self.real_stats_cached = False
        self.real_all_stats = None
        self.real_avg_stats = None
        self.real_avg_tpc_dist = None
        self.real_avg_tpc_pdf = None
        self.real_pore_data = None
        self.real_throat_data = None

        # 确保结果目录存在
        os.makedirs(os.path.join(results_dir, 'metrics'), exist_ok=True)
        self.check_csv_path = os.path.join(results_dir, 'metrics', 'check.csv')

    def _cache_real_samples_stats(self):
        """计算并缓存真实样本的统计指标（只执行一次）"""
        if self.real_stats_cached:
            return
        # 1. 计算真实样本的统计指标
        _, self.real_all_stats, self.real_avg_stats = calculate_stats_for_folder(self.real_dir, self.shape)
        # 2. 计算真实样本的平均TPC
        real_tpc_distances, real_tpc_pdfs = get_all_tpc_data(self.real_dir, shape=self.shape)
        self.real_avg_tpc_dist, self.real_avg_tpc_pdf = calculate_average_tpc(real_tpc_distances, real_tpc_pdfs)
        # 3. 提取真实样本的孔隙和喉道数据
        self.real_pore_data, self.real_throat_data = extract_data_parallel(self.real_dir, self.shape)
        self.real_stats_cached = True

    def generate_samples(self, generator, num_samples=300, temp_dir='temp_fake_samples'):
        """调用生成器生成样本并保存到临时目录"""
        os.makedirs(temp_dir, exist_ok=True)

        # 生成指定数量的样本
        batch_size = min(30, num_samples)  # 避免显存溢出
        total_generated = 0

        while total_generated < num_samples:
            current_batch_size = min(batch_size, num_samples - total_generated)
            volumes, _ = generator.generate_volumes(current_batch_size, zero_noise=False)

            # 转换为uint8并保存为.raw文件
            for i in range(current_batch_size):
                vol = volumes[i, 0].detach().cpu().numpy()
                vol = (vol > 0).astype(np.uint8)   # 二值化：小于0→0，大于0→1
                file_path = os.path.join(temp_dir, f'fake_{total_generated + i}.raw')
                vol.tofile(file_path)

            total_generated += current_batch_size

        return temp_dir

    def check(self, generator, step, num_samples=300):
        """检查当前step的生成器效果"""
        # 确保真实样本统计数据已缓存
        self._cache_real_samples_stats()
        if not self.real_stats_cached:
            return

        start_time = time.time()
        print(f"\n===== 开始检查Step {step} 的生成样本指标 =====")

        # 生成样本
        temp_dir = os.path.join(self.results_dir, 'temp_fake_samples')
        fake_dir = self.generate_samples(generator, num_samples, temp_dir)

        # 初始化结果字典
        error_results = {name: "未计算" for name in self.metrics_order}
        error_results["step"] = step
        rel_errors_1_6 = []
        base_passed = False

        # 1. 计算生成样本的统计指标
        print("正在计算生成样本的统计指标...")
        _, fake_all_stats, fake_avg_stats = calculate_stats_for_folder(fake_dir, self.shape)

        # 定义指标列表
        metrics = [
            ("总孔隙率", self.real_avg_stats[0], True),
            ("连通孔隙率", self.real_avg_stats[1], True),
            ("总孔隙数", self.real_avg_stats[2], False),
            ("连通孔隙数", self.real_avg_stats[3], False),
            ("喉道数量", self.real_avg_stats[4], False)
        ]

        # 检查基础指标
        for i, (name, real_val, is_porosity) in enumerate(metrics):
            fake_val = fake_avg_stats[i]

            # 计算相对误差（单位：%）
            if real_val == 0:
                rel_error = float('inf')
            else:
                rel_error = abs(fake_val - real_val) / real_val * 100

            rounded_error = round(rel_error, 2)
            error_results[name] = rounded_error
            rel_errors_1_6.append(rounded_error)

            # 判断是否合格
            if rel_error > self.tolerance[name]:
                rel_errors_1_6.pop()
                break
        else:
            base_passed = True

        # 2. 计算TPC指标（仅当前5项合格）
        tpc_rel_error = None
        if base_passed:
            # 计算生成样本TPC
            fake_tpc_distances, fake_tpc_pdfs = get_all_tpc_data(fake_dir, shape=self.shape)
            fake_avg_tpc_dist, fake_avg_tpc_pdf = calculate_average_tpc(fake_tpc_distances, fake_tpc_pdfs)

            # 对齐TPC数据并计算相对误差
            if base_passed:
                common_tpc_dist, aligned_real_tpc, aligned_fake_tpc = align_tpc_data(
                    self.real_avg_tpc_dist, self.real_avg_tpc_pdf,
                    fake_avg_tpc_dist, fake_avg_tpc_pdf
                )
            # 计算TPC平均相对误差
            if base_passed:
                tpc_rel_errors = calculate_tpc_relative_error(aligned_real_tpc, aligned_fake_tpc)
                if tpc_rel_errors is None:
                    base_passed = False
                else:
                    valid_tpc_errors = tpc_rel_errors[~np.isnan(tpc_rel_errors)]
                    if len(valid_tpc_errors) == 0:
                        base_passed = False
                    else:
                        tpc_rel_error = np.mean(valid_tpc_errors)
                        tpc_rounded_error = round(tpc_rel_error, 2)
                        error_results["两点相关函数"] = tpc_rounded_error
                        rel_errors_1_6.append(tpc_rounded_error)

                        if tpc_rel_error > self.tolerance["两点相关函数"]:
                            base_passed = False
                            rel_errors_1_6.pop()

        # 3. 计算孔隙和喉道有效直径区间误差（仅当TPC合格）
        abs_errors_7_8 = []
        if base_passed and tpc_rel_error is not None:
            # 提取生成样本的数据
            fake_pore_data, fake_throat_data = extract_data_parallel(fake_dir, self.shape)

            # 检查数据有效性
            if not self.real_pore_data or not fake_pore_data or not self.real_throat_data or not fake_throat_data:
                base_passed = False
            else:
                # 计算孔隙等效直径区间误差
                bins_step_pore = 5
                all_pore_data = np.concatenate([np.concatenate(self.real_pore_data), np.concatenate(fake_pore_data)])
                min_pore = np.min(all_pore_data)
                max_pore = np.max(all_pore_data)
                bins_pore = np.arange(
                    np.floor(min_pore / bins_step_pore) * bins_step_pore,
                    np.ceil(max_pore / bins_step_pore) * bins_step_pore + bins_step_pore,
                    bins_step_pore
                )

                # 计算真实样本各区间平均计数
                real_pore_counts = [np.histogram(data, bins=bins_pore)[0] for data in self.real_pore_data]
                real_pore_avg = np.mean(real_pore_counts, axis=0) if real_pore_counts else np.zeros_like(bins_pore[:-1])
                # 计算生成样本各区间平均计数
                fake_pore_counts = [np.histogram(data, bins=bins_pore)[0] for data in fake_pore_data]
                fake_pore_avg = np.mean(fake_pore_counts, axis=0) if fake_pore_counts else np.zeros_like(bins_pore[:-1])

                # 计算孔隙区间误差
                epsilon = 1e-8
                valid_pore_mask = real_pore_avg >= epsilon
                if np.sum(valid_pore_mask) == 0:
                    base_passed = False
                else:
                    pore_errors = np.abs(fake_pore_avg[valid_pore_mask] - real_pore_avg[valid_pore_mask])
                    pore_avg_error = np.max(pore_errors)
                    pore_rounded_error = round(pore_avg_error, 2)
                    error_results["孔隙等效直径区间误差"] = pore_rounded_error

                    if pore_avg_error > self.tolerance["孔隙等效直径区间误差"]:
                        base_passed = False
                    else:
                        abs_errors_7_8.append(pore_rounded_error)

                # 计算喉道等效直径区间误差（仅当孔隙检查通过）
                if base_passed:
                    bins_step_throat = 1
                    all_throat_data = np.concatenate([np.concatenate(self.real_throat_data), np.concatenate(fake_throat_data)])
                    min_throat = np.min(all_throat_data)
                    max_throat = np.max(all_throat_data)
                    bins_throat = np.arange(
                        np.floor(min_throat / bins_step_throat) * bins_step_throat,
                        np.ceil(max_throat / bins_step_throat) * bins_step_throat + bins_step_throat,
                        bins_step_throat
                    )
                    # 计算真实样本各区间平均计数
                    real_throat_counts = [np.histogram(data, bins=bins_throat)[0] for data in self.real_throat_data]
                    real_throat_avg = np.mean(real_throat_counts, axis=0) if real_throat_counts else np.zeros_like(bins_throat[:-1])
                    # 计算生成样本各区间平均计数
                    fake_throat_counts = [np.histogram(data, bins=bins_throat)[0] for data in fake_throat_data]
                    fake_throat_avg = np.mean(fake_throat_counts, axis=0) if fake_throat_counts else np.zeros_like(bins_throat[:-1])

                    # 计算喉道区间误差
                    valid_throat_mask = real_throat_avg >= epsilon
                    if np.sum(valid_throat_mask) == 0:
                        base_passed = False
                    else:
                        throat_errors = np.abs(fake_throat_avg[valid_throat_mask] - real_throat_avg[valid_throat_mask])
                        throat_avg_error = np.max(throat_errors)
                        throat_rounded_error = round(throat_avg_error, 2)
                        error_results["喉道等效直径区间误差"] = throat_rounded_error

                        if throat_avg_error > self.tolerance["喉道等效直径区间误差"]:
                            base_passed = False
                        else:
                            abs_errors_7_8.append(throat_rounded_error)

        # 计算合计值
        if len(rel_errors_1_6) == 6:
            total_rel_1_6 = round(sum(rel_errors_1_6), 2)
            error_results["合计1-6（相对误差）"] = total_rel_1_6
        else:
            error_results["合计1-6（相对误差）"] = "未全部合格"

        if len(abs_errors_7_8) == 2:
            total_abs_7_8 = round(sum(abs_errors_7_8), 2)
            error_results["合计7-8（绝对误差）"] = total_abs_7_8
        else:
            error_results["合计7-8（绝对误差）"] = "未全部合格"

        # 保存结果到CSV
        file_exists = os.path.isfile(self.check_csv_path) and os.path.getsize(self.check_csv_path) > 0
        with open(self.check_csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.metrics_order)
            if not file_exists:
                writer.writeheader()
            writer.writerow(error_results)

        # 清理临时文件
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)