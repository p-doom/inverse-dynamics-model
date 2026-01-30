import torch.nn as nn
import torch.nn.functional as F


# ---------- ResNet-style block ----------

class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        pad = kernel_size // 2
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=pad)
        self.norm2 = nn.GroupNorm(1, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=pad)

    def forward(self, x):
        out = self.conv1(F.relu(self.norm1(x)))
        out = self.conv2(F.relu(self.norm2(out)))
        return F.relu(out + x)


# ---------- ResNet stack (conv + pool + 2 residual blocks) ----------

class ResStack(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=pad)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block1 = ResBlock(out_channels, kernel_size=kernel_size)
        self.block2 = ResBlock(out_channels, kernel_size=kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(F.relu(x))
        x = self.block1(x)
        x = self.block2(x)
        return x


# ---------- Encoder: 3D conv + 3 ResStacks + 2 dense + 4 transformers ----------

class Encoder(nn.Module):
    def __init__(self, d_model, num_transformer_layers, num_heads, ff_dim, frame_mode):
        super().__init__()
        
        self.conv3d = nn.Conv3d(in_channels=6 if frame_mode == "concat" else 3,  out_channels=64, kernel_size=(5, 3, 3),  padding=(2, 1, 1), stride=(1, 2, 2))

        self.stack1 = ResStack(64, 128)   # 256 -> 128
        self.stack2 = ResStack(128, 256)  # 128 -> 64
        self.stack3 = ResStack(256, 512)  # 64 -> 32
        self.stack4 = ResStack(512, 512)  # 32 -> 16
        
        self.bottleneck = nn.Conv2d(512, 32, kernel_size=1)
        
        self.flat_dim = 32 * 16 * 16 
        self.fc1 = nn.Linear(self.flat_dim, d_model)
        self.norm_fc = nn.LayerNorm(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True, dropout=0.3)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

    def forward(self, frames):
        B, T, C, H, W = frames.shape
        x = frames.permute(0, 2, 1, 3, 4)
        x = self.conv3d(x) # [B, 64, T, 256, 256]

        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, 64, 256, 256)
        x = self.stack1(x) 
        x = self.stack2(x) 
        x = self.stack3(x) 
        x = self.stack4(x) # [B*T, 512, 16, 16]

        x = self.bottleneck(x) # [B*T, 32, 16, 16]
        x = x.reshape(B * T, -1)
        
        x = self.norm_fc(F.relu(self.fc1(x)))
        x = x.view(B, T, -1)
        return self.transformer(x)