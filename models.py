import torch
import torch.nn as nn

# U-Net Generator
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNetGenerator, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder
        self.dec4 = self.upconv_block(1024, 512)
        self.dec3 = self.upconv_block(1024, 256)
        self.dec2 = self.upconv_block(512, 128)
        self.dec1 = self.upconv_block(256, 64)
        
        # Final layer
        self.final = nn.Conv2d(128, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        
        self.pool = nn.MaxPool2d(2)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with skip connections
        d4 = self.dec4(b)
        d4 = torch.cat([d4, e4], dim=1)
        
        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        
        d1 = self.dec1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        
        output = self.final(d1)
        return self.tanh(output)

# CNN Discriminator
class CNNDiscriminator(nn.Module):
    def __init__(self, in_channels=6):
        super(CNNDiscriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, 4, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, input_img, target_img):
        x = torch.cat([input_img, target_img], dim=1)
        return self.model(x)

# Improved Discriminator
class ImprovedCNNDiscriminator(nn.Module):
    def __init__(self, in_channels=6):
        super(ImprovedCNNDiscriminator, self).__init__()
        
        # Use more gentle architecture
        self.model = nn.Sequential(
            # First layer without BatchNorm
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),  # Add dropout
            
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            # Reduce complexity of last layer
            nn.Conv2d(256, 512, 4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, 4, stride=1, padding=1)
            # Remove Sigmoid, use WGAN-GP loss
        )
    
    def forward(self, input_img, target_img):
        x = torch.cat([input_img, target_img], dim=1)
        return self.model(x)

# Loss functions
class WGANLoss(nn.Module):
    def __init__(self):
        super(WGANLoss, self).__init__()
    
    def forward(self, prediction, target_is_real):
        if target_is_real:
            return -torch.mean(prediction)
        else:
            return torch.mean(prediction)

class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()
        self.loss = nn.BCELoss()
        
    def forward(self, prediction, target_is_real):
        if target_is_real:
            target = torch.ones_like(prediction)
        else:
            target = torch.zeros_like(prediction)
        return self.loss(prediction, target)

def gradient_penalty(discriminator, real_samples, fake_samples, device):
    """Calculate gradient penalty"""
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)
    
    d_interpolates = discriminator(interpolates[:, :3], interpolates[:, 3:])
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

class FeatureMatchingLoss(nn.Module):
    def __init__(self, discriminator):
        super(FeatureMatchingLoss, self).__init__()
        self.discriminator = discriminator
        self.l1_loss = nn.L1Loss()
    
    def forward(self, real_input, real_target, fake_target):
        # Get discriminator intermediate features
        real_features = self.get_features(real_input, real_target)
        fake_features = self.get_features(real_input, fake_target)
        
        loss = 0
        for real_feat, fake_feat in zip(real_features, fake_features):
            loss += self.l1_loss(fake_feat, real_feat.detach())
        
        return loss
    
    def get_features(self, input_img, target_img):
        # Need to modify discriminator to return intermediate features
        x = torch.cat([input_img, target_img], dim=1)
        features = []
        for layer in self.discriminator.model:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                features.append(x)
        return features


# Improved U-Net Generator with Residual Blocks
class ImprovedUNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(ImprovedUNetGenerator, self).__init__()
        
        # Encoder with Residual Blocks
        self.enc1 = self.residual_conv_block(in_channels, 64)
        self.enc2 = self.residual_conv_block(64, 128)
        self.enc3 = self.residual_conv_block(128, 256)
        self.enc4 = self.residual_conv_block(256, 512)
        
        # Bottleneck with Attention
        self.bottleneck = self.residual_conv_block(512, 1024)
        self.attention = SelfAttention(1024)
        
        # Decoder with Residual Blocks
        self.dec4 = self.upconv_block(1024, 512)
        self.dec3 = self.upconv_block(1024, 256)
        self.dec2 = self.upconv_block(512, 128)
        self.dec1 = self.upconv_block(256, 64)
        
        # Final layer with residual connection
        self.final_conv = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 1)
        )
        self.tanh = nn.Tanh()
        
        self.pool = nn.MaxPool2d(2)
        
    def residual_conv_block(self, in_channels, out_channels):
        return ResidualBlock(in_channels, out_channels)
    
    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck with attention
        b = self.bottleneck(self.pool(e4))
        b = self.attention(b)
        
        # Decoder with skip connections
        d4 = self.dec4(b)
        d4 = torch.cat([d4, e4], dim=1)
        
        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        
        d1 = self.dec1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        
        output = self.final_conv(d1)
        return self.tanh(output)

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # If input and output channels differ, need 1x1 conv for adjustment
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        
        return out

# Self-Attention Mechanism
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Generate query, key, value
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        value = self.value_conv(x).view(batch_size, -1, width * height)
        
        # Calculate attention
        attention = torch.bmm(query, key)
        attention = self.softmax(attention)
        
        # Apply attention
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        # Residual connection
        out = self.gamma * out + x
        return out