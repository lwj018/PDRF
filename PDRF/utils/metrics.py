import numpy as np
import logging
import lpips
import torch
import cv2
import os
import matplotlib.pyplot as plt
import json
from scipy.signal.windows import tukey
from einops import rearrange

plt.switch_backend('agg')
np.seterr(divide='ignore', invalid='ignore')


def print_log(message, is_main_process=True):
    if is_main_process:
        print(message)
        logging.info(message)

def max_pool(arr, pool_size):
    arr = arr.squeeze(1)
    pad_H = pool_size - arr.shape[1] % pool_size
    pad_W = pool_size - arr.shape[2] % pool_size
    pad_size = ((0,0), (0,pad_H), (0,pad_W))
    arr_padded = np.pad(arr, pad_size)
    H = arr.shape[1] + pad_H
    W = arr.shape[2] + pad_W
    arr_reshaped = arr_padded.reshape(arr.shape[0], H//pool_size, pool_size, W//pool_size, pool_size)
    arr_reshaped = arr_reshaped.transpose(0,1,3,2,4)
    arr_max_pooled = np.max(arr_reshaped, axis=(3,4))
    return arr_max_pooled

def cal_ssim(pred, true, data_range = 255):
    C1 = (0.01 * data_range)**2
    C2 = (0.03 * data_range)**2
    img1 = pred.astype(np.float64)
    img2 = true.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def cal_cprs2(pred, true):
    '''cal cprs(continuous ranked probability score) in numpy, the data range is [0, 1]'''
    num_samples = pred.shape[0]
    absolute_error = np.mean(np.abs(pred - true), axis=0)
    pred_ranks = np.argsort(pred, axis=0)
    true_ranks = np.argsort(true, axis=0)
    diff = pred_ranks - true_ranks
    weight = (np.arange(num_samples) + 1) / num_samples - 0.5
    per_obs_crps = absolute_error - np.sum(diff * weight, axis=0) / num_samples**2
    return np.average(per_obs_crps, weights=None)


class PowerSpectrumAnalyzer:
    """雷达回波功率谱分析器 - 用于验证物理一致性"""
    
    def __init__(self, img_size=128, resolution_km=1.0):
        self.img_size = img_size
        self.resolution = resolution_km
        
        # 计算波数坐标
        freq = np.fft.fftfreq(img_size, d=resolution_km)  # 单位: km^-1
        self.fx, self.fy = np.meshgrid(freq, freq)
        self.k_magnitude = np.sqrt(self.fx**2 + self.fy**2)
        
        # 动态设置波数分箱：从可分辨的大尺度到接近奈奎斯特频率
        # 避开 k=0，最大到 0.9*奈奎斯特频率（避免混叠）
        nyquist = 1.0 / (2.0 * resolution_km)
        k_min = 1.0 / (img_size * resolution_km)  # 最大波长对应的波数
        
        # 对于 Shanghai (3.9km)：范围约 0.02 - 0.12 km^-1
        self.k_bins = np.linspace(k_min * 2, nyquist * 0.9, 30)
        self.k_centers = (self.k_bins[:-1] + self.k_bins[1:]) / 2
        
        print(f"[Spectrum Analyzer] Resolution: {resolution_km}km/pix, "
            f"Nyquist: {nyquist:.3f} km^-1, "
            f"Analyzing range: {self.k_bins[0]:.3f} - {self.k_bins[-1]:.3f} km^-1")
        
    def compute_spectrum(self, image):
        """
        计算单张雷达图的功率谱
        Args:
            image: [H, W] 雷达反射率，假设输入是归一化[0,1]的dBZ等效值
        Returns:
            k_centers: 波数中心 (km^-1)
            power_1d: 径向平均功率谱
            slope: 中尺度区间(0.1-1.0 km^-1)拟合斜率
        """
        # 如果是全零帧，返回NaN
        if np.max(image) < 0.01:
            return self.k_centers, np.full_like(self.k_centers, np.nan), np.nan
            
        # 假设输入是归一化的dBZ（0-1对应0-70dBZ），转为实际dBZ
        # 然后根据审稿人建议，转为线性单位做物理正确的谱分析
        # 注意：如果数据集不同，需要调整这个转换
        dbz = image * 70.0  # 假设最大值70dBZ
        linear = 10 ** (dbz / 10.0)  # 转为线性反射率因子 Z
        
        # 去趋势
        linear_centered = linear - np.mean(linear)
        
        # 加Tukey窗（减少边界效应）
        window_1d = tukey(self.img_size, alpha=0.1)
        window_2d = np.outer(window_1d, window_1d)
        image_windowed = linear_centered * window_2d
        
        # 二维FFT并中心化
        fft2d = np.fft.fft2(image_windowed)
        fft_shifted = np.fft.fftshift(fft2d)
        power_2d = np.abs(fft_shifted) ** 2
        
        # 径向平均
        power_1d = []
        for i in range(len(self.k_bins) - 1):
            mask = (self.k_magnitude >= self.k_bins[i]) & (self.k_magnitude < self.k_bins[i+1])
            if np.sum(mask) > 0:
                power_bin = np.mean(power_2d[mask])
                power_1d.append(power_bin)
            else:
                power_1d.append(np.nan)
        
        power_1d = np.array(power_1d)
        
        # 计算中尺度区间（0.1-1.0 km^-1）的斜率（enstrophy级联区，理论-3）
        meso_mask = (self.k_centers > 0.1) & (self.k_centers < 1.0)
        if np.sum(meso_mask) > 3:
            # 过滤掉NaN
            valid_mask = meso_mask & ~np.isnan(power_1d) & (power_1d > 0)
            if np.sum(valid_mask) > 3:
                log_k = np.log(self.k_centers[valid_mask])
                log_p = np.log(power_1d[valid_mask])
                slope, _ = np.polyfit(log_k, log_p, 1)
            else:
                slope = np.nan
        else:
            slope = np.nan
            
        return self.k_centers, power_1d, slope

    def analyze_batch(self, predictions, targets, lead_times=None):
        """
        批量分析功率谱
        Args:
            predictions: [N, T, H, W] 预测结果（归一化[0,1]）
            targets: [N, T, H, W] 真实观测
            lead_times: 要分析的时间步列表，None则分析所有
        Returns:
            dict: 各时间步的统计结果
        """
        N, T = predictions.shape[:2]
        if lead_times is None:
            lead_times = list(range(T))
            
        results = {}
        
        for t in lead_times:
            pred_slopes = []
            tgt_slopes = []
            pred_power_list = []
            tgt_power_list = []
            
            for n in range(N):
                # 检查是否为有效降水帧（避免全零）
                if np.max(predictions[n, t]) < 0.01 and np.max(targets[n, t]) < 0.01:
                    continue
                
                _, p_pred, s_pred = self.compute_spectrum(predictions[n, t])
                _, p_tgt, s_tgt = self.compute_spectrum(targets[n, t])
                
                if not np.isnan(s_pred):
                    pred_slopes.append(s_pred)
                    pred_power_list.append(p_pred)
                if not np.isnan(s_tgt):
                    tgt_slopes.append(s_tgt)
                    tgt_power_list.append(p_tgt)
            
            if len(pred_slopes) > 0:
                results[f't{t}'] = {
                    'pred_slope_mean': np.mean(pred_slopes),
                    'pred_slope_std': np.std(pred_slopes),
                    'tgt_slope_mean': np.mean(tgt_slopes),
                    'tgt_slope_std': np.std(tgt_slopes),
                    'slope_bias': np.mean(pred_slopes) - np.mean(tgt_slopes),
                    'pred_power_mean': np.nanmean(pred_power_list, axis=0),
                    'tgt_power_mean': np.nanmean(tgt_power_list, axis=0),
                    'valid_samples': len(pred_slopes)
                }
            else:
                results[f't{t}'] = None
                
        return results


class Evaluator(object):
    def __init__(self, seq_len, value_scale, thresholds=[20, 30, 35, 40], 
                 img_size=128, resolution_km=1.0, **kwargs):
        self.metrics = {}
        self.thresholds = thresholds
        for threshold in self.thresholds:
            self.metrics[threshold] = {
                "hits": [], "misses": [], "falsealarms": [], "correctnegs": [],
                "hits44": [], "misses44": [], "falsealarms44": [], "correctnegs44": [],
                "hits16": [], "misses16": [], "falsealarms16": [], "correctnegs16": [],
            }
        self.losses = {
            "mse": [], "mae": [], "rmse": [], "psnr": [], "ssim": [], "crps": [], "lpips": [],
        }
        self.seq_len = seq_len
        self.total = 0
        self.value_scale = value_scale
        self.img_size = img_size
        
        # 初始化功率谱分析器
        self.spectrum_analyzer = PowerSpectrumAnalyzer(img_size, resolution_km)
        self.spectrum_predictions = []  # 收集预测用于功率谱分析
        self.spectrum_targets = []      # 收集真值用于功率谱分析
        
        self.lpips_fn = lpips.LPIPS(net='alex', verbose=False)
        if torch.cuda.is_available():
            self.lpips_fn.cuda()
    
    def float2int(self, arr):
        x = arr.clip(0.0, 1.0)
        x = x * self.value_scale
        x = x.astype(np.uint16)
        return x
        
    def evaluate(self, true_batch, pred_batch):
        # [batch_size, seq_len, channel, h, w], data_range [0.,1.]
        if isinstance(pred_batch, torch.Tensor):
            pred_batch = pred_batch.detach().cpu().numpy()
            true_batch = true_batch.detach().cpu().numpy()
        
        if not (true_batch.max() <= 1.0 and true_batch.min() >= 0.0):
            print_log(f"WARNING:: data max: {true_batch.max()}, min: {true_batch.min()}")
        
        pred_batch = pred_batch.clip(0.0, 1.0)
        true_batch = true_batch.clip(0.0, 1.0)
        
        assert pred_batch.shape == true_batch.shape
        
        batch_size, seq_len = true_batch.shape[:2]
        
        # 收集数据用于功率谱分析（去掉channel维度，假设为1）
        # 存储为 [N, T, H, W]
        self.spectrum_predictions.append(pred_batch.squeeze(2))
        self.spectrum_targets.append(true_batch.squeeze(2))
        
        # 原有评估逻辑...
        lpips_batch = self.cal_batch_lpips(pred_batch, true_batch)
        self.losses['lpips'].extend(lpips_batch)
        
        pred = self.float2int(pred_batch)
        gt = self.float2int(true_batch)
        
        for threshold in self.thresholds:
            for b in range(batch_size):
                seq_hit, seq_miss, seq_falsealarm, seq_correctneg = [], [], [], []
                for t in range(seq_len):   
                    hit, miss, falsealarm, correctneg = self.cal_frame(gt[b][t], pred[b][t], threshold)
                    seq_hit.append(hit)
                    seq_miss.append(miss)
                    seq_falsealarm.append(falsealarm)
                    seq_correctneg.append(correctneg)
                    
                self.metrics[threshold]["hits"].append(seq_hit)
                self.metrics[threshold]["misses"].append(seq_miss)
                self.metrics[threshold]["falsealarms"].append(seq_falsealarm)
                self.metrics[threshold]["correctnegs"].append(seq_correctneg)
            
                # 44
                hits44, misses44, falsealarms44, correctnegs44 = self.cal_frame(max_pool(gt[b], 4), max_pool(pred[b], 4), threshold)
                self.metrics[threshold]["hits44"].append(hits44)
                self.metrics[threshold]["misses44"].append(misses44)
                self.metrics[threshold]["falsealarms44"].append(falsealarms44)
                self.metrics[threshold]["correctnegs44"].append(correctnegs44)
                
                # 16
                hits16, misses16, falsealarms16, correctnegs16 = self.cal_frame(max_pool(gt[b], 16), max_pool(pred[b], 16), threshold)
                self.metrics[threshold]["hits16"].append(hits16)
                self.metrics[threshold]["misses16"].append(misses16)
                self.metrics[threshold]["falsealarms16"].append(falsealarms16)
                self.metrics[threshold]["correctnegs16"].append(correctnegs16)

        for b in range(batch_size):
            seq_mse, seq_mae, seq_rmse, seq_psnr, seq_ssim, seq_crps = [], [], [], [], [], []
            for t in range(seq_len):
                mae, mse, rmse, psnr, ssim, crps = self.cal_frame_losses(true_batch[b][t], pred_batch[b][t])
                seq_mse.append(mse)
                seq_mae.append(mae)
                seq_rmse.append(rmse)
                seq_psnr.append(psnr)
                seq_ssim.append(ssim)
                seq_crps.append(crps)
                
            self.losses['mse'].append(seq_mse)
            self.losses['mae'].append(seq_mae)
            self.losses['rmse'].append(seq_rmse)
            self.losses['psnr'].append(seq_psnr)
            self.losses['ssim'].append(seq_ssim)
            self.losses['crps'].append(seq_crps)
        
        self.total += batch_size
            
    def cal_frame(self, obs, sim, threshold):
        obs = np.where(obs >= threshold, 1, 0)
        sim = np.where(sim >= threshold, 1, 0)
        hits = np.sum((obs == 1) & (sim == 1))
        misses = np.sum((obs == 1) & (sim == 0))
        falsealarms = np.sum((obs == 0) & (sim == 1))
        correctnegatives = np.sum((obs == 0) & (sim == 0))
        return hits, misses, falsealarms, correctnegatives
    
    def cal_frame_losses(self, pred, true):
        pred.astype(np.float32)
        true.astype(np.float32)
        pred = pred.squeeze() 
        true = true.squeeze()
        
        try:
            crps = cal_cprs2(pred, true)
        except:
            crps = 0.0

        pred = pred * self.value_scale
        true = true * self.value_scale
        mae = np.mean(np.abs(pred - true))
        mse = np.mean((pred - true) ** 2)
        rmse = np.sqrt(mse)
        psnr = 20 * np.log10(self.value_scale / np.sqrt(mse)) if mse > 0 else 100
        ssim = cal_ssim(pred, true, data_range=self.value_scale) 
        
        return mae, mse, rmse, psnr, ssim, crps
        
    def cal_batch_lpips(self, preds, trues):
        def trans(seq: np.ndarray):
            seq = torch.from_numpy(seq).float()
            seq = seq.repeat(1,1,3,1,1) if len(seq.shape)==5 else seq.unsqueeze(2).repeat(1,1,3,1,1)
            seq = seq * 2.0 - 1.0
            if torch.cuda.is_available():
                seq = seq.cuda()
            return seq

        preds = trans(preds)
        trues = trans(trues)
        
        lpips_seq = []
        for t in range(preds.shape[1]):
            lpips_frame = self.lpips_fn(preds[:, t], trues[:, t]).detach().cpu().numpy()
            lpips_seq.append(lpips_frame)
        lpips_seq = np.array(lpips_seq).squeeze((2,3,4))
        lpips_seq = lpips_seq.transpose(1,0)
        lpips_batch = list(lpips_seq)
        return lpips_batch
    
    def analyze_spectrum(self, lead_times=None):
        """执行功率谱分析并返回结果"""
        if len(self.spectrum_predictions) == 0:
            return None
            
        # 合并所有batch
        all_pred = np.concatenate(self.spectrum_predictions, axis=0)  # [N, T, H, W]
        all_tgt = np.concatenate(self.spectrum_targets, axis=0)
        
        # 分析功率谱
        results = self.spectrum_analyzer.analyze_batch(all_pred, all_tgt, lead_times)
        return results
    
    def print_spectrum_table(self, spectrum_results, is_main_process=True):
        """打印功率谱分析表格"""
        if spectrum_results is None or len(spectrum_results) == 0:
            return
            
        print_log("\n" + "="*80, is_main_process)
        print_log("Power Spectral Density Analysis (Physical Consistency Check)", is_main_process)
        print_log("="*80, is_main_process)
        print_log(f"{'Lead Time':<12} {'Pred Slope':<12} {'Target Slope':<14} {'Bias':<10} {'Theory':<10} {'Status':<12}", is_main_process)
        print_log("-"*80, is_main_process)
        
        # 根据seq_len生成时间标签
        # 假设每帧10分钟（根据数据集调整）
        time_labels = {}
        for key in spectrum_results.keys():
            t_idx = int(key.replace('t', ''))
            time_min = (t_idx + 1) * 10  # 假设每帧10分钟
            time_labels[key] = f"{time_min}min"
        
        for key in sorted(spectrum_results.keys(), key=lambda x: int(x.replace('t', ''))):
            data = spectrum_results[key]
            if data is None:
                continue
                
            pred_slope = data['pred_slope_mean']
            tgt_slope = data['tgt_slope_mean']
            bias = data['slope_bias']
            
            # 判断物理一致性：斜率是否接近-3（在-3.5到-2.5之间认为可接受）
            if -3.5 <= pred_slope <= -2.5:
                status = "Physical"
            elif -2.5 < pred_slope <= -2.0:
                status = "Noisy"
            else:
                status = "Unphysical"
            
            t_label = time_labels.get(key, key)
            print_log(f"{t_label:<12} {pred_slope:<12.3f} {tgt_slope:<14.3f} {bias:<10.3f} {'-3.0':<10} {status:<12}", is_main_process)
        
        print_log("="*80, is_main_process)
        print_log("Note: Slope ~ -3.0 indicates realistic enstrophy cascade (2D turbulence)", is_main_process)
        print_log("      Slope > -2.5 suggests excessive small-scale noise (nonphysical)", is_main_process)
        print_log("      Slope < -3.5 indicates over-smoothing", is_main_process)
        print_log("="*80, is_main_process)
    
    def save_spectrum_results(self, spectrum_results, save_path):
        """保存功率谱结果到JSON"""
        if spectrum_results is None:
            return
            
        # 转换numpy array为list以便JSON序列化
        save_data = {}
        for key, val in spectrum_results.items():
            if val is None:
                continue
            save_data[key] = {
                k: v.tolist() if isinstance(v, np.ndarray) else float(v)
                for k, v in val.items()
            }
        
        json_path = os.path.join(save_path, 'spectrum_analysis.json')
        with open(json_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        print_log(f"Spectrum analysis saved to {json_path}")
    
    def done(self, is_main_process=True, save_path=None):
        """完成评估并输出所有结果"""
        res_dict = {}
        print_log('*'*30+' < Evaluation Results: > '+'*'*30, is_main_process)
        print_log(f"Total {self.total} samples with {self.seq_len} seq_len.", is_main_process)
        print_log('*'*90, is_main_process)
        
        # ... 保留原有的CSI/FAR/POD/HSS等计算逻辑 ...
        avg_csi, avg_far, avg_pod, avg_hss = [], [], [], []
        avg_csi44, avg_csi16 = [], []
        
        for threshold in self.thresholds:
            hits = np.array(self.metrics[threshold]["hits"])
            misses = np.array(self.metrics[threshold]["misses"])
            falsealarms = np.array(self.metrics[threshold]["falsealarms"])
            correctnegs = np.array(self.metrics[threshold]["correctnegs"])
            
            hits = np.nan_to_num(hits)
            misses = np.nan_to_num(misses)
            falsealarms = np.nan_to_num(falsealarms)
            correctnegs = np.nan_to_num(correctnegs)
            
            csi1 = np.mean(hits, axis=0) / (np.mean(hits, axis=0) + np.mean(misses, axis=0) + np.mean(falsealarms, axis=0))
            far1 = np.mean(falsealarms, axis=0) / (np.mean(hits, axis=0) + np.mean(falsealarms, axis=0))
            pod1 = np.mean(hits, axis=0) / (np.mean(hits, axis=0) + np.mean(misses, axis=0))
            hss1 = 2 * (np.mean(hits, axis=0) * np.mean(correctnegs, axis=0) - np.mean(misses, axis=0) * np.mean(falsealarms, axis=0)) / ((np.mean(hits, axis=0) + np.mean(misses, axis=0)) * (np.mean(misses, axis=0) + np.mean(correctnegs, axis=0)) + (np.mean(hits, axis=0) + np.mean(falsealarms, axis=0)) * (np.mean(falsealarms, axis=0) + np.mean(correctnegs, axis=0)))                 
            
            csi1 = np.nan_to_num(csi1)
            far1 = np.nan_to_num(far1)
            pod1 = np.nan_to_num(pod1)
            hss1 = np.nan_to_num(hss1)
            
            avg_csi.append(np.mean(csi1))
            avg_far.append(np.mean(far1))
            avg_pod.append(np.mean(pod1))
            avg_hss.append(np.mean(hss1))
            
            hits44 = np.array(self.metrics[threshold]["hits44"])
            misses44 = np.array(self.metrics[threshold]["misses44"])
            falsealarms44 = np.array(self.metrics[threshold]["falsealarms44"])
            correctnegs44 = np.array(self.metrics[threshold]["correctnegs44"])
            
            hits16 = np.array(self.metrics[threshold]["hits16"])
            misses16 = np.array(self.metrics[threshold]["misses16"])
            falsealarms16 = np.array(self.metrics[threshold]["falsealarms16"])
            correctnegs16 = np.array(self.metrics[threshold]["correctnegs16"])
            
            csi_pool44 = np.mean(hits44) / (np.mean(hits44) + np.mean(misses44) + np.mean(falsealarms44))
            avg_csi44.append(csi_pool44)
            
            csi_pool16 = np.mean(hits16) / (np.mean(hits16) + np.mean(misses16) + np.mean(falsealarms16))
            avg_csi16.append(csi_pool16)
         
            if threshold == self.thresholds[len(self.thresholds)//2]:  # 打印中间阈值
                print_log('='*20 + f"Threshold: {threshold}"+'='*20, is_main_process)
                print_log(f'<CSI> : {np.mean(csi1)}; '+str(csi1), is_main_process)
                print_log(f'<FAR> : {np.mean(far1)}; '+str(far1), is_main_process)
                print_log(f'<POD> : {np.mean(pod1)}; '+str(pod1), is_main_process)
                print_log(f'<HSS> : {np.mean(hss1)}; '+str(hss1), is_main_process)
            
            print_log(f"< CSI_POOL 4x4 > : {csi_pool44}; CSI_POOL 16x16: {csi_pool16}")
            

        print_log('*'*20 + f"Overall Avg Metrics on Thresholds {self.thresholds}"+'*'*20, is_main_process)
        print_log(f"[ avg_csi ] : {np.mean(avg_csi)}; [ avg_far ] : {np.mean(avg_far)}; [ avg_pod ] : {np.mean(avg_pod)}; [ avg_hss] : {np.mean(avg_hss)}", is_main_process)
        print_log(f"[ avg_csi_pool 4x4 ] : {np.mean(avg_csi44)}; [ avg_csi_pool 16x16 ]: {np.mean(avg_csi16)}", is_main_process)

        res_dict['csi'] = np.nan_to_num(np.mean(avg_csi))
        
        # 计算损失
        mses = np.mean(np.array(self.losses['mse']), axis=0)
        mass = np.mean(np.array(self.losses['mae']), axis=0)
        rmses = np.mean(np.array(self.losses['rmse']),axis=0)
        psnrs = np.mean(np.array(self.losses['psnr']), axis=0)
        ssims = np.mean(np.array(self.losses['ssim']), axis=0)
        crpss = np.mean(np.array(self.losses['crps']), axis=0)
        lpipss = np.mean(np.array(self.losses['lpips']), axis=0)
        
        print_log('='*20 + f"Losses with {self.seq_len} seq_len"+'='*20, is_main_process)
        print_log(f'<MSE> : {np.mean(mses)}; '+str(mses), is_main_process)
        print_log(f'<MAE> : {np.mean(mass)}; '+str(mass), is_main_process)
        print_log(f'<RMSE> : {np.mean(rmses)}; '+str(rmses), is_main_process)
        print_log(f'<PSNR> : {np.mean(psnrs)}; '+str(psnrs), is_main_process)
        print_log(f'<SSIM> : {np.mean(ssims)}; '+str(ssims), is_main_process)
        print_log(f'<CRPS> : {np.mean(crpss)}; '+str(crpss), is_main_process)
        print_log(f'<LPIPS> : {np.mean(lpipss)}; '+str(lpipss), is_main_process)
        
        # 新增：功率谱分析
        if save_path:
            # 选择关键时间步：开始、中间、结束
            key_times = [0, self.seq_len//2, self.seq_len-1] if self.seq_len > 5 else list(range(self.seq_len))
            spectrum_results = self.analyze_spectrum(lead_times=key_times)
            self.print_spectrum_table(spectrum_results, is_main_process)
            self.save_spectrum_results(spectrum_results, save_path)
        
        print_log('='*90, is_main_process)
        
        return res_dict


if __name__ == '__main__':
    # 简单测试
    tarray = np.array([i for i in range(36)]).reshape(1,6,6)
    print(tarray)
    pooled = max_pool(tarray, 2)
    print(pooled)