import torch.nn as nn
import torch.nn.functional as F


# ---------- ResNet-style block ----------

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(1, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(F.relu(self.norm1(x)))
        out = self.conv2(F.relu(self.norm2(out)))
        return F.relu(out + x)


# ---------- ResNet stack (conv + pool + 2 residual blocks) ----------

class ResStack(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block1 = ResBlock(out_channels)
        self.block2 = ResBlock(out_channels)

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

        if frame_mode == "concat":
            in_channels = 6 
        elif frame_mode == "diff":
            in_channels = 3
        else:
            raise NotImplementedError(f"Unknown frame_mode: {frame_mode}")

        # 3D temporal conv: 128 filters, kernel (5,1,1), non-causal via padding (2,0,0)
        self.conv3d = nn.Conv3d(in_channels=in_channels, out_channels=128, kernel_size=(5, 1, 1), padding=(2, 0, 0))

        # ResNet-like stacks with widths {64, 128, 128}
        self.stack1 = ResStack(in_channels=128, out_channels=64)
        self.stack2 = ResStack(in_channels=64, out_channels=128)
        self.stack3 = ResStack(in_channels=128, out_channels=512)

        # Frame-wise dense layers: 32768 -> 256 -> 4096 (per frame)
        flat_dim = 512 * 16 * 16
        self.ln_flat = nn.LayerNorm(flat_dim)
        self.fc1 = nn.Linear(flat_dim, d_model)
        self.ln_fc1 = nn.LayerNorm(d_model)

        # Non-causal Transformer over time
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
            activation="relu",
            dropout=0.3
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

    def forward(self, frames):
        B, T, C, H, W = frames.shape

        x = frames.permute(0, 2, 1, 3, 4)  # [B, 6, T, H, W]
        x = self.conv3d(x)                 # [B, 128, T, H, W]

        x = x.permute(0, 2, 1, 3, 4)       # [B, T, 128, H, W]
        x = x.reshape(B * T, 128, H, W)

        x = self.stack1(x)                 # -> [B*T, 64, 64, 64]
        x = self.stack2(x)                 # -> [B*T, 128, 32, 32]
        x = self.stack3(x)                 # -> [B*T, 128, 16, 16]

        x = x.view(B * T, -1)              # [B*T, 512]
        x = self.ln_flat(x)
        x = F.relu(self.fc1(x))            # [B*T, 256]
        x = self.ln_fc1(x)

        x = x.view(B, T, -1)               # [B, T, d_model]
        x = self.transformer(x)            # [B, T, d_model]

        return x