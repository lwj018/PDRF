import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp

class SSIM(nn.Module):
    """
    SSIM (Structural Similarity Index Measure) a Pytorch module.
    
    Args:
        window_size (int): The size of the Gaussian window. Default: 11.
        data_range (float or int): The range of the input images (e.g., 1.0 for images with values in [0, 1] 
                                   or 255 for images with values in [0, 255]). Default: 1.0.
        channel (int): The number of channels of the input images. This is needed to create the Gaussian window.
    """
    def __init__(self, window_size=11, data_range=1.0, channel=20):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.data_range = data_range
        self.channel = channel
        
        # C1 and C2 are small constants to avoid division by zero
        self.C1 = (0.01 * self.data_range) ** 2
        self.C2 = (0.03 * self.data_range) ** 2
        
        # Create a 2D Gaussian window and register it as a buffer
        # Buffers are part of the module's state, but not considered model parameters.
        # They are moved to the correct device (e.g., GPU) along with the module.
        window = self._create_gaussian_window(self.window_size, self.channel)
        self.register_buffer('window', window)

    def _create_gaussian_window(self, window_size, channel):
        """Creates​ a 2D Gaussian window."""
        # 1D Gaussian kernel
        _1D_window = torch.randn(window_size).fill_(0)
        sigma = 1.5
        for i in range(window_size):
            _1D_window[i] = exp(-(i - window_size // 2) ** 2 / (2 * sigma ** 2))
        
        # 2D Gaussian kernel from the outer product of two 1D kernels
        _2D_window = _1D_window.unsqueeze(1) @ _1D_window.unsqueeze(0)
        
        # Normalize
        window = _2D_window.expand(channel, 1, window_size, window_size) / _2D_window.sum()
        
        return window

    def forward(self, img1, img2):
        """
        Forward pass to compute SSIM.
        
        Args:
            img1 (torch.Tensor): The first image tensor of shape (B, C, H, W).
            img2 (torch.Tensor): The second image tensor of shape (B, C, H, W).
        
        Returns:
            torch.Tensor: A scalar tensor with the mean SSIM score over the batch.
        """
        # Get number of channels from the input tensor
        C = img1.size(1)

        # The `groups` argument in conv2d is crucial. It ensures that the convolution
        # is applied to each channel independently.
        # For a (B, C, H, W) input, the window should have shape (C, 1, win_size, win_size)
        # and groups should be C.
        
        # Calculate local means (mu)
        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=C)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=C)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # Calculate local variances (sigma^2) and covariance (sigma_xy)
        # Var(X) = E[X^2] - (E[X])^2
        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=C) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=C) - mu2_sq
        
        # Cov(X, Y) = E[XY] - E[X]E[Y]
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=C) - mu1_mu2

        # Calculate the SSIM map using the simplified formula
        ssim_numerator = (2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)
        ssim_denominator = (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2)
        ssim_map = ssim_numerator / ssim_denominator

        # Return the mean SSIM over the batch
        return ssim_map.sum()


# if __name__ == '__main__':
#     # 1. 设置参数
#     batch_size = 4
#     channels = 9
#     height = 128
#     width = 128
    
#     # 2. 创建模拟的真实图像和预测图像
#     # 假设图像数据范围是 [0, 1]
#     true_images = torch.rand(batch_size, channels, height, width)
#     pred_images = true_images * 0.95 + 0.05 * torch.rand(batch_size,channels,height,width) # 创建一个与真实图像相似的预测图像

#     # 3. 实例化SSIM模块
#     # 注意：这里的 channel 和 data_range 必须与你的数据匹配
#     ssim_module = SSIM(data_range=1.0, channel=channels)
    
#     # 4. 计算SSIM值
#     ssim_score = ssim_module(true_images, pred_images)
#     print(f"SSIM Score: {ssim_score.item()}")

#     # 5. 作为损失函数使用
#     # SSIM值越高越好 (最大为1)，所以作为损失函数时通常用 1 - SSIM
#     ssim_loss = 1 - ssim_score
#     print(f"SSIM Loss: {ssim_loss.item()}")

#     # 检查GPU兼容性
#     if torch.cuda.is_available():
#         print("\n--- Testing on GPU ---")
#         device = torch.device("cuda")
        
#         true_images_gpu = true_images.to(device)
#         pred_images_gpu = pred_images.to(device)
        
#         ssim_module_gpu = SSIM(data_range=1.0, channel=channels).to(device)
        
#         ssim_score_gpu = ssim_module_gpu(true_images_gpu, pred_images_gpu)
#         print(f"SSIM Score on GPU: {ssim_score_gpu.item()}")
        
#         ssim_loss_gpu = 1 - ssim_score_gpu
#         print(f"SSIM Loss on GPU: {ssim_loss_gpu.item()}")
