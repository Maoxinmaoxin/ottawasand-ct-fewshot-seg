# Batch prediction script - Ensemble prediction using 10 fold models
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import glob
from models import ImprovedUNetGenerator
from cross_validation_trainer import TrainingConfig

def predict_ensemble_single_image(models, image_path, output_path, patch_size=512):
    """
    Perform ensemble prediction on a single image using loaded models
    
    Args:
        models: List of loaded models
        image_path: Input image path
        output_path: Output image path
        patch_size: Patch size for processing
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Read and resize image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((512, 512), Image.Resampling.LANCZOS)
    img_w, img_h = img.size
    
    # Create ensemble result image
    ensemble_result = np.zeros((img_h, img_w, 3), dtype=np.float32)
    
    # Patch processing, prediction and stitching
    for y in range(0, img_h, patch_size):
        for x in range(0, img_w, patch_size):
            # Crop patch
            patch = img.crop((x, y, x + patch_size, y + patch_size))
            patch_tensor = transform(patch).unsqueeze(0).to(device)
            
            # Store predictions from all models
            patch_predictions = []
            
            # Predict with each model
            for model in models:
                with torch.no_grad():
                    output_patch = model(patch_tensor)
                
                # 后处理
                output_patch = output_patch.squeeze(0).cpu().detach()
                output_patch = (output_patch * 0.5 + 0.5).numpy()
                output_patch = np.transpose(output_patch, (1, 2, 0))
                
                patch_predictions.append(output_patch)
            
            # 计算集成结果（平均）
            ensemble_patch = np.mean(patch_predictions, axis=0)
            h, w, _ = ensemble_patch.shape
            ensemble_result[y:y+h, x:x+w, :] = ensemble_patch
    
    # 保存集成结果
    ensemble_img = (ensemble_result * 255).astype(np.uint8)
    ensemble_img = Image.fromarray(ensemble_img)
    ensemble_img.save(output_path)

def load_all_models(fold_models_dir):
    """
    加载所有fold的模型
    
    Args:
        fold_models_dir: 包含所有fold模型的目录
    
    Returns:
        models: 已加载的模型列表
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载所有fold的模型
    models = []
    fold_paths = [
        os.path.join(fold_models_dir, 'fold_1', 'unet_final_model.pth'),
        os.path.join(fold_models_dir, 'fold_2', 'unet_final_model.pth'),
        os.path.join(fold_models_dir, 'fold_3', 'unet_final_model.pth'),
        os.path.join(fold_models_dir, 'fold_4', 'unet_final_model.pth'),
        os.path.join(fold_models_dir, 'fold_5', 'unet_final_model.pth'),
        os.path.join(fold_models_dir, 'fold_6', 'unet_final_model.pth'),
        os.path.join(fold_models_dir, 'fold_7', 'unet_final_model.pth'),
        os.path.join(fold_models_dir, 'fold_8', 'unet_final_model.pth'),
        os.path.join(fold_models_dir, 'fold_9', 'unet_final_model.pth'),
        os.path.join(fold_models_dir, 'fold_10', 'unet_final_model.pth')
    ]
    
    for i, model_path in enumerate(fold_paths):
        print(f"Loading model from fold {i+1}: {model_path}")
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found: {model_path}")
            continue
        
        # 加载模型
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        except Exception:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        model = ImprovedUNetGenerator(in_channels=3, out_channels=3)
        
        # Extract model state dict from checkpoint
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model = checkpoint
        
        model.to(device)
        model.eval()
        models.append(model)
    
    print(f"Successfully loaded {len(models)} models")
    return models

def predict_batch(input_dir, output_dir, fold_models_dir, patch_size=512):
    """
    批量预测函数
    
    Args:
        input_dir: 输入图像目录
        output_dir: 输出目录
        fold_models_dir: 包含所有fold模型的目录
        patch_size: 切块大小
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载所有模型（只加载一次）
    print("Loading all models...")
    models = load_all_models(fold_models_dir)
    
    if len(models) == 0:
        print("Error: No models loaded successfully!")
        return
    
    # 获取所有图像文件
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if len(image_files) == 0:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # 批量处理图像
    for i, image_path in enumerate(image_files):
        # 获取文件名（不含扩展名）
        filename = os.path.splitext(os.path.basename(image_path))[0]
        output_filename = f"{filename}_pre.png"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)} -> {output_filename}")
        
        try:
            # 对单张图像进行集成预测
            predict_ensemble_single_image(models, image_path, output_path, patch_size)
            print(f"  Saved: {output_path}")
        except Exception as e:
            print(f"  Error processing {image_path}: {e}")
            continue
    
    print(f"\nBatch prediction completed! Results saved in: {output_dir}")

if __name__ == '__main__':
    # Configuration - Replace with actual paths
    input_dir = ''  # Insert path to input image directory (e.g., 'original')
    output_dir = ''  # Insert path to output directory (e.g., 'predict_testtest')
    fold_models_dir = ''  # Insert path to cross-validation models directory (e.g., 'checkpoints_cv_perfect')
    
    print("=== 10-Fold Cross-Validation Ensemble Batch Prediction ===")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Models directory: {fold_models_dir}")
    print()
    
    # 执行批量预测
    try:
        predict_batch(input_dir, output_dir, fold_models_dir)
        print("\n=== Batch prediction completed successfully! ===")
        
    except Exception as e:
        print(f"Error during batch prediction: {e}")
        import traceback
        traceback.print_exc()