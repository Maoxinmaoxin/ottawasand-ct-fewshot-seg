# K-fold cross-validation training script for improved U-Net model
# Features: Early stopping, warmup scheduler, data augmentation, multi-metric evaluation

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import json
import random
import torchvision.transforms as transforms
from PIL import Image
import glob
from skimage.metrics import structural_similarity as ssim
import cv2
import torchvision.models as models
import torch.nn.functional as F
from models import ImprovedUNetGenerator
from data_preprocessor import preprocess_data

# 设置随机种子以确保结果可重现
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# 尝试导入 lpips
try:
    import lpips
except ImportError:
    print("Warning: lpips not installed. Please install with: pip install lpips")
    lpips = None

kfold = 10  # Number of folds for cross-validation

# GPU内存监控和管理函数
def get_gpu_memory_info():
    """获取GPU内存使用信息"""
    if not torch.cuda.is_available():
        return None
    
    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    cached_memory = torch.cuda.memory_reserved(device)
    free_memory = total_memory - allocated_memory
    
    return {
        'total': total_memory / 1024**3,  # GB
        'allocated': allocated_memory / 1024**3,  # GB
        'cached': cached_memory / 1024**3,  # GB
        'free': free_memory / 1024**3,  # GB
        'usage_percent': (allocated_memory / total_memory) * 100
    }

def check_gpu_memory_and_adjust(current_batch_size, min_batch_size=1):
    """检查GPU内存并建议调整批次大小"""
    memory_info = get_gpu_memory_info()
    if memory_info is None:
        return current_batch_size, False
    
    # 如果内存使用超过85%，建议减小批次大小
    if memory_info['usage_percent'] > 85 and current_batch_size > min_batch_size:
        new_batch_size = max(min_batch_size, current_batch_size // 2)
        print(f"\n⚠️ High GPU memory usage ({memory_info['usage_percent']:.1f}%). Suggesting batch size reduction: {current_batch_size} -> {new_batch_size}")
        return new_batch_size, True
    
    return current_batch_size, False

def safe_gpu_cleanup():
    """安全清理GPU缓存"""
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception as e:
            print(f"Warning: GPU cleanup failed: {e}")

class TrainingConfig:
    def __init__(self):
        # 基本参数 - 支持更大batch size
        self.batch_size = 8  # 从8增加到16
        self.num_epochs = 1000  # 恢复正常训练轮数
        
        # 学习率策略优化
        base_lr = 0.0002
        # 使用平方根缩放而非线性缩放，更适合大batch size
        self.lr = base_lr * np.sqrt(self.batch_size / 8)  # 平方根缩放更稳定
        
        self.beta1 = 0.5
        self.lambda_l1 = 10
        self.lambda_perceptual = 1.0
        self.image_size = 128
        
        # 正则化参数
        self.weight_decay = 1e-4
        
        # 改进的Warmup策略参数
        self.warmup_epochs = 10  # 测试用最小值
        self.warmup_factor = 0.1  # 更小的起始因子

        # 学习率调度器参数（保留作为备用）
        self.scheduler_patience = 15
        self.scheduler_factor = 0.5
        self.scheduler_min_lr = 5e-5  # 更小的最小学习率

        # 早停机制参数
        self.early_stopping_patience = 150  # 早停耐心，增加到100个epoch
        self.early_stopping_delta = 0.0005  # 损失变化的最小阈值，增加到0.001
        
        # 保存设置
        # 生成带时间戳的checkpoint目录名
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.checkpoint_dir = f'checkpoints_{current_time}'
        self.save_interval = 100
        self.log_interval = 1
        
        # 设备设置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建保存目录
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        print(f'Using device: {self.device}')
        print(f'Batch size: {self.batch_size}, Learning rate: {self.lr:.6f}')
        if torch.cuda.is_available():
            print(f'GPU: {torch.cuda.get_device_name(0)}')
            print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

# WarmupScheduler 类
class WarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, warmup_factor=0.1):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Warmup阶段：线性增加学习率
            lr_scale = self.warmup_factor + (1.0 - self.warmup_factor) * epoch / self.warmup_epochs
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * lr_scale
    
    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self):
        return {
            'warmup_epochs': self.warmup_epochs,
            'warmup_factor': self.warmup_factor,
            'base_lrs': self.base_lrs
        }
    
    def load_state_dict(self, state_dict):
        self.warmup_epochs = state_dict['warmup_epochs']
        self.warmup_factor = state_dict['warmup_factor']
        self.base_lrs = state_dict['base_lrs']

# ImageToImageDataset 类
class ImageToImageDataset(Dataset):
    def __init__(self, file_list_path, transform=None, augment=False):
        with open(file_list_path, 'r') as f:
            self.file_pairs = [line.strip().split(';') for line in f.readlines()]
        self.transform = transform
        self.augment = augment
        
    def __len__(self):
        return len(self.file_pairs)
        
    def __getitem__(self, idx):
        image_path, label_path = self.file_pairs[idx]
        
        # 加载图像和标签
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('RGB')
        
        # 数据增强
        if self.augment:
            # 随机水平翻转
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                label = label.transpose(Image.FLIP_LEFT_RIGHT)
            
            # 随机垂直翻转
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                label = label.transpose(Image.FLIP_TOP_BOTTOM)
            
            # 随机旋转
            if random.random() > 0.5:
                angle = random.randint(0, 360)  # 从0到360度随机选择
                image = image.rotate(angle)
                label = label.rotate(angle)
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
            
        return image, label

# MetricsCalculator 类
class MetricsCalculator:
    def __init__(self, device='cpu'):
        self.device = device
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        # 初始化感知损失网络
        self.vgg = models.vgg19(pretrained=True).features[:16].to(device).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # 初始化LPIPS
        if lpips is not None:
            self.lpips_model = lpips.LPIPS(net='alex').to(device)
        else:
            self.lpips_model = None
    
    def calculate_psnr(self, pred, target):
        """计算峰值信噪比（PSNR）"""
        mse = torch.mean((pred - target) ** 2)
        if mse == 0:
            return torch.tensor(float('inf'))
        return 20 * torch.log10(2.0 / torch.sqrt(mse))
    
    def calculate_perceptual_loss(self, pred, target):
        """计算感知损失"""
        try:
            pred_norm = (pred + 1) / 2
            target_norm = (target + 1) / 2
            
            pred_features = self.vgg(pred_norm)
            target_features = self.vgg(target_norm)
            
            perceptual_loss = self.mse_loss(pred_features, target_features)
            return perceptual_loss
        except RuntimeError as e:
            if "CUDA" in str(e) or "out of memory" in str(e):
                print(f"CUDA错误，跳过感知损失计算: {e}")
                torch.cuda.empty_cache()
                return torch.tensor(0.0, device=self.device, requires_grad=True)
            else:
                raise e
        except Exception as e:
            print(f"感知损失计算错误: {e}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def calculate_gradient_loss(self, pred, target):
        """计算梯度损失"""
        try:
            def gradient(img):
                sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
                sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
                
                grad_x = F.conv2d(img.mean(dim=1, keepdim=True), sobel_x, padding=1)
                grad_y = F.conv2d(img.mean(dim=1, keepdim=True), sobel_y, padding=1)
                
                return torch.sqrt(grad_x**2 + grad_y**2)
            
            pred_grad = gradient(pred)
            target_grad = gradient(target)
            
            return self.l1_loss(pred_grad, target_grad)
        except Exception as e:
            print(f"Error calculating gradient loss: {e}")
            return torch.tensor(0.0, device=self.device)
    
    def calculate_edge_loss(self, pred, target):
        """计算边缘损失"""
        try:
            def detect_edges(img):
                laplacian_kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
                edges = F.conv2d(img.mean(dim=1, keepdim=True), laplacian_kernel, padding=1)
                return edges
            
            pred_edges = detect_edges(pred)
            target_edges = detect_edges(target)
            
            return self.l1_loss(pred_edges, target_edges)
        except Exception as e:
            print(f"Error calculating edge loss: {e}")
            return torch.tensor(0.0, device=self.device)
    
    def calculate_lpips(self, pred, target):
        """计算LPIPS距离"""
        try:
            if self.lpips_model is None:
                return 0.0
            
            pred_norm = (pred + 1) / 2
            target_norm = (target + 1) / 2
            
            lpips_distance = self.lpips_model(pred_norm, target_norm)
            return lpips_distance.mean().item()
        except Exception as e:
            print(f"Error calculating LPIPS: {e}")
            return 0.0
    
    def calculate_pixel_accuracy(self, pred, target, threshold=0.1):
        """计算像素准确率"""
        try:
            pred_norm = (pred + 1) / 2
            target_norm = (target + 1) / 2
            
            pixel_diff = torch.abs(pred_norm - target_norm)
            pixel_diff_mean = torch.mean(pixel_diff, dim=1)
            
            correct_pixels = (pixel_diff_mean < threshold).sum().float()
            total_pixels = pixel_diff_mean.numel()
            
            accuracy = correct_pixels / total_pixels
            return accuracy.item()
        except Exception as e:
            print(f"Error calculating pixel accuracy: {e}")
            return 0.0
    
    def calculate_mpa(self, pred, target, num_classes=3):
        """计算平均像素精度（Mean Pixel Accuracy）"""
        try:
            pred_norm = (pred + 1) / 2
            target_norm = (target + 1) / 2
            
            pred_classes = (pred_norm * (num_classes - 1)).round().long()
            target_classes = (target_norm * (num_classes - 1)).round().long()
            
            accuracies = []
            for c in range(num_classes):
                pred_c = (pred_classes == c)
                target_c = (target_classes == c)
                
                if target_c.sum() > 0:
                    accuracy = (pred_c & target_c).sum().float() / target_c.sum().float()
                    accuracies.append(accuracy.item())
            
            return np.mean(accuracies) if accuracies else 0.0
        except Exception as e:
            print(f"Error calculating MPA: {e}")
            return 0.0
    
    def calculate_miou(self, pred, target, num_classes=3):
        """计算平均交并比（Mean Intersection over Union）"""
        try:
            pred_norm = (pred + 1) / 2
            target_norm = (target + 1) / 2
            
            pred_classes = (pred_norm * (num_classes - 1)).round().long()
            target_classes = (target_norm * (num_classes - 1)).round().long()
            
            ious = []
            for c in range(num_classes):
                pred_c = (pred_classes == c)
                target_c = (target_classes == c)
                
                intersection = (pred_c & target_c).sum().float()
                union = (pred_c | target_c).sum().float()
                
                if union > 0:
                    iou = intersection / union
                    ious.append(iou.item())
            
            return np.mean(ious) if ious else 0.0
        except Exception as e:
            print(f"Error calculating MIoU: {e}")
            return 0.0
    
    def calculate_f1_score(self, pred, target, threshold=0.5):
        """计算F1分数"""
        try:
            pred_norm = (pred + 1) / 2
            target_norm = (target + 1) / 2
            
            pred_binary = (pred_norm > threshold).float()
            target_binary = (target_norm > threshold).float()
            
            tp = (pred_binary * target_binary).sum().float()
            fp = (pred_binary * (1 - target_binary)).sum().float()
            fn = ((1 - pred_binary) * target_binary).sum().float()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            return f1.item()
        except Exception as e:
            print(f"Error calculating F1 Score: {e}")
            return 0.0
    
    def calculate_dice_coefficient(self, pred, target, threshold=0.5):
        """计算Dice相似系数"""
        try:
            pred_norm = (pred + 1) / 2
            target_norm = (target + 1) / 2
            
            pred_binary = (pred_norm > threshold).float()
            target_binary = (target_norm > threshold).float()
            
            intersection = (pred_binary * target_binary).sum().float()
            dice = (2.0 * intersection) / (pred_binary.sum() + target_binary.sum() + 1e-8)
            
            return dice.item()
        except Exception as e:
            print(f"Error calculating Dice Coefficient: {e}")
            return 0.0
    
    def calculate_gpa(self, pred, target):
        """计算全局像素精度（Global Pixel Accuracy）"""
        try:
            pred_norm = (pred + 1) / 2
            target_norm = (target + 1) / 2
            
            # 使用阈值判断像素是否匹配
            threshold = 0.1
            pixel_diff = torch.abs(pred_norm - target_norm)
            pixel_diff_mean = torch.mean(pixel_diff, dim=1)
            
            correct_pixels = (pixel_diff_mean < threshold).sum().float()
            total_pixels = pixel_diff_mean.numel()
            
            gpa = correct_pixels / total_pixels
            return gpa.item()
        except Exception as e:
            print(f"Error calculating GPA: {e}")
            return 0.0
    
    def calculate_iou(self, pred, target, threshold=0.5):
        """计算IoU（Intersection over Union）"""
        try:
            pred_norm = (pred + 1) / 2
            target_norm = (target + 1) / 2
            
            pred_binary = (pred_norm > threshold).float()
            target_binary = (target_norm > threshold).float()
            
            intersection = (pred_binary * target_binary).sum().float()
            union = pred_binary.sum() + target_binary.sum() - intersection
            
            iou = intersection / (union + 1e-8)
            return iou.item()
        except Exception as e:
            print(f"Error calculating IoU: {e}")
            return 0.0
    
    def calculate_metrics(self, pred, target):
        """计算所有指标"""
        pred_np = ((pred + 1) / 2).clamp(0, 1).cpu().detach().numpy()
        target_np = ((target + 1) / 2).clamp(0, 1).cpu().detach().numpy()
        
        metrics = {}
        
        # 基础指标
        metrics['mse'] = self.mse_loss(pred, target).item()
        metrics['mae'] = self.l1_loss(pred, target).item()
        
        # MAPE
        epsilon = 1e-8
        mape = torch.mean(torch.abs((target - pred) / (target + epsilon))) * 100
        metrics['mape'] = mape.item()
        
        # PSNR
        metrics['psnr'] = self.calculate_psnr(pred, target).item()
        
        # 感知损失
        metrics['perceptual_loss'] = self.calculate_perceptual_loss(pred, target).item()
        
        # 梯度损失
        metrics['gradient_loss'] = self.calculate_gradient_loss(pred, target).item()
        
        # 边缘损失
        metrics['edge_loss'] = self.calculate_edge_loss(pred, target).item()
        
        # LPIPS
        metrics['lpips'] = self.calculate_lpips(pred, target)
        
        # 像素准确率
        metrics['pixel_accuracy'] = self.calculate_pixel_accuracy(pred, target)
        
        # 新增指标
        metrics['mpa'] = self.calculate_mpa(pred, target)
        metrics['miou'] = self.calculate_miou(pred, target)
        metrics['iou'] = self.calculate_iou(pred, target)
        metrics['f1_score'] = self.calculate_f1_score(pred, target)
        metrics['dice_coefficient'] = self.calculate_dice_coefficient(pred, target)
        metrics['gpa'] = self.calculate_gpa(pred, target)
        
        # SSIM
        if pred_np.shape[0] > 0:
            pred_img = np.transpose(pred_np[0], (1, 2, 0))
            target_img = np.transpose(target_np[0], (1, 2, 0))
            
            ssim_value = ssim(target_img, pred_img, multichannel=True, data_range=1.0, channel_axis=-1)
            metrics['ssim'] = ssim_value
            
            pred_flat = pred_img.flatten()
            target_flat = target_img.flatten()
            correlation = np.corrcoef(pred_flat, target_flat)[0, 1]
            metrics['correlation'] = correlation if not np.isnan(correlation) else 0.0
        else:
            metrics['ssim'] = 0.0
            metrics['correlation'] = 0.0
        
        return metrics

# 绘制指标图表函数
def plot_individual_metrics(train_history, val_history, save_dir):
    """为每个指标单独绘制训练集和验证集的对比图"""
    import matplotlib
    matplotlib.use('Agg')
    
    metrics_names = list(train_history.keys())
    
    # 指标的英文名称映射
    metric_labels = {
        'losses': 'U-Net Loss',
        'mae_scores': 'Mean Absolute Error (MAE)',
        'mse_scores': 'Mean Squared Error (MSE)',
        'mape_scores': 'Mean Absolute Percentage Error (MAPE)',
        'ssim_scores': 'Structural Similarity Index (SSIM)',
        'psnr_scores': 'Peak Signal-to-Noise Ratio (PSNR)',
        'perceptual_loss_scores': 'Perceptual Loss',
        'gradient_loss_scores': 'Gradient Loss',
        'edge_loss_scores': 'Edge Loss',
        'lpips_scores': 'LPIPS Score',
        'pixel_accuracy_scores': 'Pixel Accuracy',
        'mpa_scores': 'Mean Pixel Accuracy (MPA)',
        'miou_scores': 'Mean Intersection over Union (MIoU)',
        'iou_scores': 'Intersection over Union (IoU)',
        'f1_score_scores': 'F1 Score',
        'dice_coefficient_scores': 'Dice Coefficient',
        'gpa_scores': 'Global Pixel Accuracy (GPA)',
        'correlation_scores': 'Correlation Coefficient'
    }
    
    # 为每个指标创建单独的图表
    for metric in metrics_names:
        if metric in train_history and len(train_history[metric]) > 0:
            plt.figure(figsize=(12, 8))
            
            # 绘制训练集和验证集的曲线
            train_epochs = range(1, len(train_history[metric]) + 1)
            plt.plot(train_epochs, train_history[metric], 'b-', label='Training', linewidth=2, marker='o', markersize=4)

            if metric in val_history and len(val_history[metric]) > 0:
                val_epochs = range(1, len(val_history[metric]) + 1)
                plt.plot(val_epochs, val_history[metric], 'g-', label='Validation', linewidth=2, marker='s', markersize=4)
            
            plt.title(f'{metric_labels.get(metric, metric)} Over Epochs', fontsize=16, fontweight='bold')
            plt.xlabel('Epoch', fontsize=14)
            plt.ylabel(metric_labels.get(metric, metric), fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # 保存图表
            safe_metric_name = metric.replace('/', '_').replace(' ', '_')
            plt.savefig(os.path.join(save_dir, f'{safe_metric_name}_plot.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"📈 Individual metric plots saved to {save_dir}")

# 生成验证集可视化结果
def generate_validation_visualization(model, val_loader, device, save_dir, epoch):
    """生成验证集的可视化结果，将原图、标签和预测结果组合保存"""
    import matplotlib
    matplotlib.use('Agg')
    
    model.eval()
    
    # 创建可视化保存目录
    vis_dir = os.path.join(save_dir, 'validation_visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    with torch.no_grad():
        # 只处理验证集的前几个batch
        for batch_idx, (images, labels) in enumerate(val_loader):
            if batch_idx >= 2:  # 只处理前2个batch
                break
                
            images = images.to(device)
            labels = labels.to(device)
            
            # 生成预测结果
            predictions = model(images)
            
            # 转换为可显示的格式
            images_np = ((images + 1) / 2).clamp(0, 1).cpu().numpy()
            labels_np = ((labels + 1) / 2).clamp(0, 1).cpu().numpy()
            predictions_np = ((predictions + 1) / 2).clamp(0, 1).cpu().numpy()
            
            # 为每个样本创建可视化
            batch_size = images.shape[0]
            for i in range(min(batch_size, 4)):  # 每个batch最多处理4个样本
                # 创建3x1的子图布局
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # 原图
                img = np.transpose(images_np[i], (1, 2, 0))
                axes[0].imshow(img)
                axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
                axes[0].axis('off')
                
                # 标签
                label = np.transpose(labels_np[i], (1, 2, 0))
                axes[1].imshow(label)
                axes[1].set_title('Ground Truth Label', fontsize=14, fontweight='bold')
                axes[1].axis('off')
                
                # 预测结果
                pred = np.transpose(predictions_np[i], (1, 2, 0))
                axes[2].imshow(pred)
                axes[2].set_title('Model Prediction', fontsize=14, fontweight='bold')
                axes[2].axis('off')
                
                # 添加整体标题
                fig.suptitle(f'Validation Results - Epoch {epoch} - Batch {batch_idx+1} - Sample {i+1}', 
                           fontsize=16, fontweight='bold')
                
                plt.tight_layout()
                
                # 保存图像
                save_path = os.path.join(vis_dir, f'validation_epoch_{epoch}_batch_{batch_idx+1}_sample_{i+1}.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
    
    print(f"📸 Validation visualizations saved to {vis_dir}")

# 保存训练日志函数
def save_training_logs(train_history, val_history, config, total_training_time, save_dir='checkpoints'):
    """保存训练集和验证集的训练日志到CSV文件"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存CSV文件
    def save_csv(history, filename):
        df_data = {'epoch': list(range(1, len(history['losses']) + 1))}
        for key, values in history.items():
            df_data[key] = [float(val) for val in values]
        
        df = pd.DataFrame(df_data)
        csv_path = os.path.join(save_dir, filename)
        df.to_csv(csv_path, index=False)
        return csv_path
    
    train_csv_path = save_csv(train_history, f'training_metrics_{timestamp}.csv')
    val_csv_path = save_csv(val_history, f'validation_metrics_{timestamp}.csv')
    
    print(f"📊 Training logs saved:")
    print(f"  CSV: {train_csv_path}")
    print(f"📊 Validation logs saved:")
    print(f"  CSV: {val_csv_path}")
    
    return train_csv_path, val_csv_path

# 在验证集上评估模型
def evaluate_on_validation_set(model, val_loader, metrics_calculator, device):
    """在验证集上评估U-Net模型性能，带有异常处理和内存管理"""
    model.eval()
    val_metrics = {
        'losses': [], 'mae_scores': [], 'mse_scores': [], 'mape_scores': [],
        'ssim_scores': [], 'psnr_scores': [], 'perceptual_loss_scores': [],
        'gradient_loss_scores': [], 'edge_loss_scores': [], 'lpips_scores': [],
        'pixel_accuracy_scores': [], 'correlation_scores': [],
        'mpa_scores': [], 'miou_scores': [], 'iou_scores': [], 'f1_score_scores': [],
        'dice_coefficient_scores': [], 'gpa_scores': []
    }
    
    total_loss = 0
    num_batches = 0
    successful_batches = 0
    
    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    with torch.no_grad():
        for batch_idx, (real_images, real_labels) in enumerate(tqdm(val_loader, desc="Evaluating on validation set")):
            try:
                real_images = real_images.to(device)
                real_labels = real_labels.to(device)
                
                # 生成预测
                fake_labels = model(real_images)
                
                # 计算损失
                l1_loss = nn.L1Loss()(fake_labels, real_labels)
                perceptual_loss = metrics_calculator.calculate_perceptual_loss(fake_labels, real_labels)
                total_batch_loss = l1_loss + 0.1 * perceptual_loss
                
                total_loss += total_batch_loss.item()
                num_batches += 1
                
                # 计算详细指标
                batch_metrics = metrics_calculator.calculate_metrics(fake_labels, real_labels)
                
                # 累积指标
                val_metrics['losses'].append(total_batch_loss.item())
                val_metrics['mae_scores'].append(batch_metrics['mae'])
                val_metrics['mse_scores'].append(batch_metrics['mse'])
                val_metrics['mape_scores'].append(batch_metrics['mape'])
                val_metrics['ssim_scores'].append(batch_metrics['ssim'])
                val_metrics['psnr_scores'].append(batch_metrics['psnr'])
                val_metrics['perceptual_loss_scores'].append(batch_metrics['perceptual_loss'])
                val_metrics['gradient_loss_scores'].append(batch_metrics['gradient_loss'])
                val_metrics['edge_loss_scores'].append(batch_metrics['edge_loss'])
                val_metrics['lpips_scores'].append(batch_metrics['lpips'])
                val_metrics['pixel_accuracy_scores'].append(batch_metrics['pixel_accuracy'])
                val_metrics['correlation_scores'].append(batch_metrics['correlation'])
                val_metrics['mpa_scores'].append(batch_metrics['mpa'])
                val_metrics['miou_scores'].append(batch_metrics['miou'])
                val_metrics['iou_scores'].append(batch_metrics['iou'])
                val_metrics['f1_score_scores'].append(batch_metrics['f1_score'])
                val_metrics['dice_coefficient_scores'].append(batch_metrics['dice_coefficient'])
                val_metrics['gpa_scores'].append(batch_metrics['gpa'])
                
                successful_batches += 1
                
                # 定期清理GPU缓存
                if batch_idx % 4 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n⚠️ CUDA out of memory during evaluation batch {batch_idx}. Clearing cache and skipping...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    print(f"\n❌ Runtime error during evaluation batch {batch_idx}: {e}")
                    # 对于其他运行时错误，尝试继续
                    continue
            except Exception as e:
                print(f"\n❌ Unexpected error during evaluation batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # 如果没有成功处理任何batch，返回默认指标
    if successful_batches == 0:
        print("\n⚠️ No batches were successfully processed during evaluation. Using default metrics.")
        default_metrics = {
            'losses': 1.0, 'mae_scores': 1.0, 'mse_scores': 1.0, 'mape_scores': 100.0,
            'ssim_scores': 0.0, 'psnr_scores': 0.0, 'perceptual_loss_scores': 1.0,
            'gradient_loss_scores': 1.0, 'edge_loss_scores': 1.0, 'lpips_scores': 1.0,
            'pixel_accuracy_scores': 0.0, 'correlation_scores': 0.0,
            'mpa_scores': 0.0, 'miou_scores': 0.0, 'iou_scores': 0.0, 'f1_score_scores': 0.0,
            'dice_coefficient_scores': 0.0, 'gpa_scores': 0.0
        }
        return default_metrics
    
    # 计算平均指标
    avg_metrics = {}
    for key, values in val_metrics.items():
        if values:
            avg_metrics[key] = np.mean(values)
        else:
            avg_metrics[key] = 0.0
    
    # 最终清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return avg_metrics

# 高级学习率调度器
class AdvancedLRScheduler:
    """高级学习率调度器，支持多种策略组合"""
    def __init__(self, optimizer, config):
        self.optimizer = optimizer
        self.config = config
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = 0
        
        # 多阶段学习率衰减里程碑
        self.milestones = {
            config.warmup_epochs: 1.0,  # Warmup结束
            config.num_epochs // 4: 0.5,  # 1/4处衰减到50%
            config.num_epochs // 2: 0.1,  # 1/2处衰减到10%
            3 * config.num_epochs // 4: 0.01,  # 3/4处衰减到1%
            max(config.num_epochs - 50, config.num_epochs - 10): 0.001,  # 最后阶段衰减到0.1%
        }
        
        # 确保最小学习率足够小
        self.min_lr = min(5e-5, config.scheduler_min_lr)  # 确保能到0.00005
        
    def step(self, epoch, val_loss=None):
        self.current_epoch = epoch
        
        if epoch < self.config.warmup_epochs:
            # Warmup阶段：线性增长
            warmup_factor = self.config.warmup_factor + (1.0 - self.config.warmup_factor) * epoch / self.config.warmup_epochs
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.base_lrs[i] * warmup_factor
        else:
            # 多阶段衰减 + Cosine Annealing
            progress = (epoch - self.config.warmup_epochs) / max(1, self.config.num_epochs - self.config.warmup_epochs)
            
            # 计算当前阶段的衰减因子
            decay_factor = 1.0
            for milestone_epoch, factor in self.milestones.items():
                if epoch >= milestone_epoch:
                    decay_factor = factor
            
            # Cosine Annealing (在当前衰减基础上)
            cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
            
            # 组合衰减
            final_factor = decay_factor * (0.1 + 0.9 * cosine_factor)  # 保持10%基础 + 90%cosine
            
            for i, param_group in enumerate(self.optimizer.param_groups):
                new_lr = self.base_lrs[i] * final_factor
                param_group['lr'] = max(new_lr, self.min_lr)
    
    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self):
        return {
            'base_lrs': self.base_lrs,
            'current_epoch': self.current_epoch,
            'milestones': self.milestones,
            'min_lr': self.min_lr
        }
    
    def load_state_dict(self, state_dict):
        self.base_lrs = state_dict['base_lrs']
        self.current_epoch = state_dict['current_epoch']
        self.milestones = state_dict['milestones']
        self.min_lr = state_dict['min_lr']




class CVTrainingConfig(TrainingConfig):
    """Configuration for Cross-Validation Training."""
    def __init__(self, k_folds=2, resume_checkpoint=None, resume_fold=None, resume_epoch=None):  # 测试用最小值
        super().__init__()
        self.k_folds = k_folds
        
        # Resume training parameters
        self.resume_checkpoint = resume_checkpoint  # Path to checkpoint file
        self.resume_fold = resume_fold  # Which fold to resume from
        self.resume_epoch = resume_epoch  # Which epoch to resume from
        
        # Override checkpoint_dir for CV
        if resume_checkpoint:
            # Extract checkpoint directory from resume path
            self.checkpoint_dir = os.path.dirname(os.path.dirname(resume_checkpoint))
            print(f"📂 Resuming training, using existing checkpoint directory: {self.checkpoint_dir}")
        else:
            current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.checkpoint_dir = f'checkpoints_cv_{current_time}'
            os.makedirs(self.checkpoint_dir, exist_ok=True)


def train_unet_cv(config):
    """Trains a U-Net model using K-fold cross-validation, mirroring the structure of train_unet."""
    try:
        # 1. Data Preprocessing
        print("🚀 Starting Data Preprocessing...")
        preprocess_data(input_dir='data', output_dir='processed_data', target_size=(config.image_size, config.image_size))
        print("✅ Data Preprocessing Complete.")

        # 2. Dataset and Transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # 加载所有数据用于K折交叉验证（训练集+验证集）
        # 读取训练集和验证集文件
        all_file_pairs = []
        for file_name in ['train.txt', 'val.txt']:
            file_path = os.path.join('processed_data', file_name)
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    all_file_pairs.extend([line.strip() for line in f.readlines()])
        
        # 创建临时文件列表用于数据集初始化
        temp_file_path = os.path.join('processed_data', 'temp_all_data.txt')
        with open(temp_file_path, 'w') as f:
            for line in all_file_pairs:
                f.write(line + '\n')
        
        full_dataset = ImageToImageDataset(file_list_path=temp_file_path, transform=transform, augment=True)
        
        # 清理临时文件
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        print(f"Total samples for CV: {len(full_dataset)}")

        # 3. K-Fold Cross-Validation Setup
        kf = KFold(n_splits=config.k_folds, shuffle=True, random_state=42)
        all_folds_histories = []
        start_time = time.time()

        for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(full_dataset)))):
            # Check if we need to skip this fold when resuming
            if config.resume_checkpoint and config.resume_fold is not None:
                if fold + 1 < config.resume_fold:
                    print(f"⏭️ Skipping Fold {fold+1} (already completed)")
                    continue
                # Only skip the specific resume fold if we're resuming from a checkpoint
                # All other folds should continue normally
            
            separator = '=' * 20
            print(f'\n{separator} FOLD {fold+1}/{config.k_folds} {separator}')
            fold_start_time = time.time()

            # Create data loaders for the current fold
            train_subset = Subset(full_dataset, train_idx)
            val_subset = Subset(full_dataset, val_idx)
            train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False, num_workers=0)

            # Initialize model and components for each fold
            model = ImprovedUNetGenerator(in_channels=3, out_channels=3).to(config.device)
            optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(config.beta1, 0.999), weight_decay=config.weight_decay)
            
            # 使用新的高级学习率调度器
            advanced_scheduler = AdvancedLRScheduler(optimizer, config)
            metrics_calculator = MetricsCalculator(device=config.device)

            # History tracking for the current fold
            # 初始化一个临时的评估来获取指标键名
            temp_metrics = evaluate_on_validation_set(model, val_loader, metrics_calculator, config.device)
            fold_history = {
                'train_history': {k: [] for k in temp_metrics.keys()},
                'val_history': {k: [] for k in temp_metrics.keys()}
            }

            # Initialize early stopping parameters for the current fold
            best_val_loss = float('inf')
            epochs_no_improve = 0
            start_epoch = 0
            
            # Load checkpoint if resuming training
            if config.resume_checkpoint and fold + 1 == config.resume_fold:
                try:
                    print(f"📂 Loading checkpoint: {config.resume_checkpoint}")
                    checkpoint = torch.load(config.resume_checkpoint, map_location=config.device, weights_only=False)
                    
                    # Load model state
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"✅ Model state loaded from epoch {checkpoint['epoch']}")
                    
                    # Load optimizer state if available
                    if 'optimizer_state_dict' in checkpoint:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        print(f"✅ Optimizer state loaded")
                    
                    # Load scheduler state if available
                    if 'scheduler_state_dict' in checkpoint:
                        advanced_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                        print(f"✅ Scheduler state loaded")
                    
                    # Load fold history if available
                    if 'fold_history' in checkpoint:
                        fold_history = checkpoint['fold_history']
                        print(f"✅ Training history loaded")
                    
                    # Set starting epoch
                    start_epoch = checkpoint['epoch']
                    if config.resume_epoch is not None:
                        start_epoch = config.resume_epoch
                    
                    print(f"🔄 Resuming training from epoch {start_epoch + 1}")
                    
                except Exception as e:
                    print(f"❌ Failed to load checkpoint: {e}")
                    print(f"🔄 Starting fresh training for fold {fold+1}")
                    start_epoch = 0

            # Epoch loop for the current fold
            for epoch in range(start_epoch, config.num_epochs):
                try:
                    # 使用高级学习率调度器
                    advanced_scheduler.step(epoch)
                    current_lr = advanced_scheduler.get_lr()[0]
                    
                    # 定期检查GPU内存状态和系统健康状况
                    if epoch % 10 == 0:
                        memory_info = get_gpu_memory_info()
                        if memory_info:
                            print(f"\n📊 GPU Memory Status (Fold {fold+1}, Epoch {epoch+1}): {memory_info['usage_percent']:.1f}% used ({memory_info['allocated']:.2f}GB/{memory_info['total']:.2f}GB)")
                            if memory_info['usage_percent'] > 90:
                                print(f"⚠️ High memory usage detected. Consider reducing batch size.")

                    # --- Training Phase ---
                    model.train()
                    train_pbar = tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1} [Training]")
                    
                    # 安全清理GPU缓存
                    safe_gpu_cleanup()
                    
                    successful_batches = 0
                    total_batches = len(train_loader)
                    
                    for batch_idx, (real_images, real_labels) in enumerate(train_pbar):
                        try:
                            real_images, real_labels = real_images.to(config.device), real_labels.to(config.device)
                            optimizer.zero_grad()
                            fake_labels = model(real_images)
                            l1_loss = nn.L1Loss()(fake_labels, real_labels)
                            perceptual_loss = metrics_calculator.calculate_perceptual_loss(fake_labels, real_labels)
                            total_loss = l1_loss + config.lambda_perceptual * perceptual_loss
                            total_loss.backward()
                            optimizer.step()
                            train_pbar.set_postfix({'Loss': f'{total_loss.item():.4f}'})
                            successful_batches += 1
                            
                            # 定期清理GPU缓存
                            if batch_idx % 4 == 0:
                                safe_gpu_cleanup()
                                
                        except RuntimeError as e:
                            if "out of memory" in str(e).lower():
                                print(f"\n⚠️ CUDA out of memory at batch {batch_idx}/{total_batches} (Fold {fold+1}, Epoch {epoch+1}). Clearing cache and skipping batch...")
                                safe_gpu_cleanup()
                                # 记录内存状态
                                memory_info = get_gpu_memory_info()
                                if memory_info:
                                    print(f"   Memory after cleanup: {memory_info['usage_percent']:.1f}% used")
                                continue
                            else:
                                print(f"\n❌ Runtime error in training batch {batch_idx}/{total_batches} (Fold {fold+1}, Epoch {epoch+1}): {e}")
                                print(f"   Error type: {type(e).__name__}")
                                import traceback
                                traceback.print_exc()
                                raise e
                        except Exception as e:
                            print(f"\n❌ Unexpected error in training batch {batch_idx}/{total_batches} (Fold {fold+1}, Epoch {epoch+1}): {e}")
                            print(f"   Error type: {type(e).__name__}")
                            import traceback
                            traceback.print_exc()
                            continue
                    
                    # 检查训练批次成功率
                    if successful_batches == 0:
                        print(f"\n⚠️ No successful training batches in Fold {fold+1}, Epoch {epoch+1}. Skipping epoch...")
                        continue
                    elif successful_batches < total_batches * 0.5:
                        print(f"\n⚠️ Only {successful_batches}/{total_batches} batches succeeded in Fold {fold+1}, Epoch {epoch+1}")

                    # --- Evaluation Phase (after each epoch) ---
                    try:
                        # 安全清理GPU缓存
                        safe_gpu_cleanup()
                        
                        print(f"\n🔍 Starting evaluation for Fold {fold+1}, Epoch {epoch+1}...")
                        
                        # Evaluate on training subset of the fold
                        train_metrics = evaluate_on_validation_set(model, train_loader, metrics_calculator, config.device)
                        for key, value in train_metrics.items():
                            fold_history['train_history'][key].append(value)

                        # Evaluate on validation subset of the fold
                        val_metrics = evaluate_on_validation_set(model, val_loader, metrics_calculator, config.device)
                        for key, value in val_metrics.items():
                            fold_history['val_history'][key].append(value)
                            
                        print(f"✅ Evaluation completed successfully for Fold {fold+1}, Epoch {epoch+1}")
                            
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            print(f"\n⚠️ CUDA out of memory during evaluation (Fold {fold+1}, Epoch {epoch+1}). Clearing cache and using fallback metrics...")
                            safe_gpu_cleanup()
                            # 记录内存状态
                            memory_info = get_gpu_memory_info()
                            if memory_info:
                                print(f"   Memory after cleanup: {memory_info['usage_percent']:.1f}% used")
                            
                            # 使用上一个epoch的指标作为备用
                            if epoch > 0:
                                for key in fold_history['train_history'].keys():
                                    fold_history['train_history'][key].append(fold_history['train_history'][key][-1])
                                    fold_history['val_history'][key].append(fold_history['val_history'][key][-1])
                                train_metrics = {k: v[-1] for k, v in fold_history['train_history'].items()}
                                val_metrics = {k: v[-1] for k, v in fold_history['val_history'].items()}
                                print(f"   Using previous epoch metrics as fallback")
                            else:
                                # 第一个epoch，使用默认值
                                default_metrics = {'losses': 1.0, 'miou_scores': 0.0, 'f1_score_scores': 0.0, 'mpa_scores': 0.0}
                                for key in fold_history['train_history'].keys():
                                    value = default_metrics.get(key, 0.0)
                                    fold_history['train_history'][key].append(value)
                                    fold_history['val_history'][key].append(value)
                                train_metrics = default_metrics
                                val_metrics = default_metrics
                                print(f"   Using default metrics for first epoch")
                        else:
                            print(f"\n❌ Runtime error during evaluation (Fold {fold+1}, Epoch {epoch+1}): {e}")
                            print(f"   Error type: {type(e).__name__}")
                            import traceback
                            traceback.print_exc()
                            raise e
                    except Exception as e:
                        print(f"\n❌ Unexpected error during evaluation (Fold {fold+1}, Epoch {epoch+1}): {e}")
                        print(f"   Error type: {type(e).__name__}")
                        import traceback
                        traceback.print_exc()
                        # 使用默认指标继续
                        default_metrics = {'losses': 1.0, 'miou_scores': 0.0, 'f1_score_scores': 0.0, 'mpa_scores': 0.0}
                        for key in fold_history['train_history'].keys():
                            value = default_metrics.get(key, 0.0)
                            fold_history['train_history'][key].append(value)
                            fold_history['val_history'][key].append(value)
                        train_metrics = default_metrics
                        val_metrics = default_metrics
                        print(f"   Using default metrics to continue training")

                    # Early stopping check
                    current_val_loss = val_metrics['losses']
                    if current_val_loss < best_val_loss - config.early_stopping_delta:
                        best_val_loss = current_val_loss
                        epochs_no_improve = 0
                        # Save the best model for this fold
                        try:
                            fold_dir = os.path.join(config.checkpoint_dir, f'fold_{fold+1}')
                            os.makedirs(fold_dir, exist_ok=True)
                            best_model_path = os.path.join(fold_dir, 'best_model.pth')
                            torch.save(model.state_dict(), best_model_path)
                            print(f"  ✅ Validation loss improved: {current_val_loss:.6f} (best: {best_val_loss:.6f}). Saved best model for fold {fold+1}")
                        except Exception as e:
                            print(f"  ⚠️ Failed to save best model: {e}")
                    else:
                        epochs_no_improve += 1
                        print(f"  ⏳ No improvement for {epochs_no_improve}/{config.early_stopping_patience} epochs (current: {current_val_loss:.6f}, best: {best_val_loss:.6f})")

                    if epochs_no_improve >= config.early_stopping_patience:
                        print(f"  🛑 Early stopping triggered for fold {fold+1} after {epoch + 1} epochs (no improvement for {epochs_no_improve} epochs).")
                        break
                    
                    # 学习率已由advanced_scheduler自动调整，无需额外调用

                    # Print epoch results
                    print(f"\nEpoch {epoch+1}/{config.num_epochs} Results (LR: {current_lr:.6f}):")
                    print(f"  Training   - Loss: {train_metrics['losses']:.4f}, MIoU: {train_metrics['miou_scores']:.4f}, F1: {train_metrics['f1_score_scores']:.4f}, MPA: {train_metrics['mpa_scores']:.4f}")
                    print(f"  Validation - Loss: {val_metrics['losses']:.4f}, MIoU: {val_metrics['miou_scores']:.4f}, F1: {val_metrics['f1_score_scores']:.4f}, MPA: {val_metrics['mpa_scores']:.4f}")

                    # Save checkpoint for the fold
                    if (epoch + 1) % config.save_interval == 0 or epoch == config.num_epochs - 1:
                        try:
                            fold_dir = os.path.join(config.checkpoint_dir, f'fold_{fold+1}')
                            os.makedirs(fold_dir, exist_ok=True)
                            checkpoint_path = os.path.join(fold_dir, f'unet_checkpoint_epoch_{epoch+1}.pth')
                            
                            # Save comprehensive checkpoint with all states
                            checkpoint_data = {
                                'epoch': epoch + 1,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': advanced_scheduler.state_dict(),
                                'fold_history': fold_history,
                                'fold': fold + 1,
                                'best_val_loss': best_val_loss,
                                'epochs_no_improve': epochs_no_improve,
                                'config': {
                                    'lr': config.lr,
                                    'batch_size': config.batch_size,
                                    'num_epochs': config.num_epochs,
                                    'k_folds': config.k_folds
                                }
                            }
                            
                            torch.save(checkpoint_data, checkpoint_path)
                            print(f"💾 Comprehensive checkpoint saved for fold {fold+1}: {checkpoint_path}")
                        except Exception as e:
                            print(f"  ⚠️ Failed to save checkpoint: {e}")
                            
                except KeyboardInterrupt:
                    print(f"\n🛑 Training interrupted by user at Fold {fold+1}, Epoch {epoch+1}")
                    print(f"💾 Saving emergency checkpoint...")
                    try:
                        emergency_dir = os.path.join(config.checkpoint_dir, f'fold_{fold+1}', 'emergency')
                        os.makedirs(emergency_dir, exist_ok=True)
                        emergency_path = os.path.join(emergency_dir, f'emergency_epoch_{epoch+1}.pth')
                        torch.save({
                            'epoch': epoch + 1,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'fold_history': fold_history,
                            'fold': fold + 1
                        }, emergency_path)
                        print(f"✅ Emergency checkpoint saved: {emergency_path}")
                    except Exception as save_e:
                        print(f"❌ Failed to save emergency checkpoint: {save_e}")
                    return
                except Exception as e:
                    print(f"\n❌ Critical error in Fold {fold+1}, Epoch {epoch+1}: {e}")
                    print(f"   Error type: {type(e).__name__}")
                    import traceback
                    traceback.print_exc()
                    
                    # 记录系统状态
                    memory_info = get_gpu_memory_info()
                    if memory_info:
                        print(f"   GPU Memory: {memory_info['usage_percent']:.1f}% used ({memory_info['allocated']:.2f}GB/{memory_info['total']:.2f}GB)")
                    
                    # 尝试保存紧急检查点
                    try:
                        emergency_dir = os.path.join(config.checkpoint_dir, f'fold_{fold+1}', 'emergency')
                        os.makedirs(emergency_dir, exist_ok=True)
                        emergency_path = os.path.join(emergency_dir, f'error_recovery_epoch_{epoch+1}.pth')
                        torch.save({
                            'epoch': epoch + 1,
                            'model_state_dict': model.state_dict(),
                            'fold_history': fold_history,
                            'fold': fold + 1,
                            'error_info': str(e)
                        }, emergency_path)
                        print(f"💾 Error recovery checkpoint saved: {emergency_path}")
                    except Exception as save_e:
                        print(f"❌ Failed to save error recovery checkpoint: {save_e}")
                    
                    print(f"\n🔄 Attempting to continue with next epoch...")
                    # 安全清理GPU缓存
                    safe_gpu_cleanup()
                    continue

            # --- End of Fold ---
            fold_duration = time.time() - fold_start_time
            print(f"\n🏁 Fold {fold+1} finished in {fold_duration/60:.2f} minutes.")
            all_folds_histories.append(fold_history)

            # Save final model, plots, and logs for the fold
            fold_dir = os.path.join(config.checkpoint_dir, f'fold_{fold+1}')
            final_model_path = os.path.join(fold_dir, 'unet_final_model.pth')
            torch.save({'model_state_dict': model.state_dict()}, final_model_path)
            plot_individual_metrics(fold_history['train_history'], fold_history['val_history'], fold_dir)
            save_training_logs(fold_history['train_history'], fold_history['val_history'], config, fold_duration, fold_dir)
            generate_validation_visualization(model, val_loader, config.device, fold_dir, config.num_epochs)

        # 4. Aggregate and Summarize CV Results
        total_training_time = time.time() - start_time
        print(f"\n{'='*20} CROSS-VALIDATION SUMMARY {'='*20}")
        print(f"⏱️ Total CV training time: {total_training_time/60:.2f} minutes")

        # Average the metrics from the last epoch of each fold
        summary = {}
        for hist_type in ['train_history', 'val_history']:
            for metric in all_folds_histories[0][hist_type].keys():
                last_epoch_vals = [fold[hist_type][metric][-1] for fold in all_folds_histories]
                summary[f'avg_final_{hist_type}_{metric}'] = float(np.mean(last_epoch_vals))
                summary[f'std_final_{hist_type}_{metric}'] = float(np.std(last_epoch_vals))
        
        print("\n📈 Final Averaged Metrics (from last epoch of each fold):")
        for key, value in summary.items():
            print(f"  - {key}: {value:.4f}")

        summary_path = os.path.join(config.checkpoint_dir, 'cv_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"\n💾 CV summary saved to {summary_path}")

    except KeyboardInterrupt:
        print(f"\n🛑 Cross-validation training interrupted by user")
        print(f"💾 Attempting to save global emergency state...")
        try:
            emergency_global_dir = os.path.join(config.checkpoint_dir, 'global_emergency')
            os.makedirs(emergency_global_dir, exist_ok=True)
            emergency_state = {
                'completed_folds': len(all_folds_histories),
                'total_folds': config.k_folds,
                'all_folds_histories': all_folds_histories,
                'interruption_time': time.time()
            }
            emergency_path = os.path.join(emergency_global_dir, 'global_emergency_state.json')
            with open(emergency_path, 'w') as f:
                json.dump(emergency_state, f, indent=4)
            print(f"✅ Global emergency state saved: {emergency_path}")
        except Exception as save_e:
            print(f"❌ Failed to save global emergency state: {save_e}")
        return
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR during CV training: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        
        # 记录系统状态
        memory_info = get_gpu_memory_info()
        if memory_info:
            print(f"   GPU Memory: {memory_info['usage_percent']:.1f}% used ({memory_info['allocated']:.2f}GB/{memory_info['total']:.2f}GB)")
        
        # 保存错误状态
        try:
            error_dir = os.path.join(config.checkpoint_dir, 'error_logs')
            os.makedirs(error_dir, exist_ok=True)
            error_info = {
                'error_message': str(e),
                'error_type': type(e).__name__,
                'completed_folds': len(all_folds_histories) if 'all_folds_histories' in locals() else 0,
                'total_folds': config.k_folds,
                'error_time': time.time(),
                'traceback': traceback.format_exc()
            }
            error_path = os.path.join(error_dir, f'critical_error_{int(time.time())}.json')
            with open(error_path, 'w') as f:
                json.dump(error_info, f, indent=4)
            print(f"💾 Error information saved: {error_path}")
        except Exception as save_e:
            print(f"❌ Failed to save error information: {save_e}")
        
        # 清理GPU缓存
        safe_gpu_cleanup()
        raise e  # 重新抛出异常，确保程序不会静默退出


if __name__ == '__main__':
    try:
        print("🚀 Initializing K-Fold Cross-Validation Training...")
        
        # Resume training configuration - modify these parameters to resume from checkpoint
        # Example: Resume from fold 5, epoch 400
        # resume_checkpoint_path = ""  # Example: "d:/path/to/checkpoints_cv_20250810_151620/fold_5/unet_checkpoint_epoch_400.pth"
        # resume_fold = 5  # Which fold to resume from
        # resume_epoch = 400  # Which epoch to resume from (optional, will use checkpoint epoch if None)
        
        # Set to None to start fresh training
        resume_checkpoint_path = ""  # Insert path to checkpoint file (e.g., "checkpoints_cv_20250810_151620\\fold_6\\unet_checkpoint_epoch_200.pth")
        resume_fold = 6
        resume_epoch = 200
        
        if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
            print(f"🔄 Resuming training from checkpoint: {resume_checkpoint_path}")
            cv_config = CVTrainingConfig(
                k_folds=kfold,
                resume_checkpoint=resume_checkpoint_path,
                resume_fold=resume_fold,
                resume_epoch=resume_epoch
            )
        else:
            if resume_checkpoint_path:
                print(f"⚠️ Checkpoint file not found: {resume_checkpoint_path}")
                print(f"🔄 Starting fresh training instead...")
            cv_config = CVTrainingConfig(k_folds=kfold)
        
        print(f"🔧 Configuration: {cv_config.num_epochs} epochs, {cv_config.k_folds} folds, batch size {cv_config.batch_size}, image size {cv_config.image_size}")
        print(f"📊 Advanced LR Scheduler: Multi-stage decay + Cosine Annealing, Min LR: {cv_config.scheduler_min_lr}")
        
        # 检查系统状态
        memory_info = get_gpu_memory_info()
        if memory_info:
            print(f"📊 Initial GPU Memory: {memory_info['usage_percent']:.1f}% used ({memory_info['allocated']:.2f}GB/{memory_info['total']:.2f}GB)")
        
        train_unet_cv(cv_config)
        print("\n🎉 Cross-validation training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n🛑 Program interrupted by user")
    except Exception as e:
        print(f"\n💥 FATAL ERROR: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        print("\n❌ Program terminated due to critical error")
        exit(1)