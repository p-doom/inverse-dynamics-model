import torch.nn as nn
import torch
from blocks import Encoder
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

class KeystrokeIDM(nn.Module):
    def __init__(self, num_keys, d_model=4096, num_transformer_layers=4, num_heads=32, ff_dim=16384, frame_mode="diff", pretrained=False, pretrained_model_path=None):
        super().__init__()

        if pretrained:        
            self.qwen = Qwen3VLForConditionalGeneration.from_pretrained(pretrained_model_path, torch_dtype=torch.bfloat16)
            self.processor = AutoProcessor.from_pretrained(pretrained_model_path)
            
            for param in self.qwen.parameters():
                param.requires_grad = False
            
            d_model = 256
        else: 
            self.encoder = Encoder(
                d_model=d_model,
                num_transformer_layers=num_transformer_layers,
                num_heads=num_heads,
                ff_dim=ff_dim,
                frame_mode=frame_mode
            )

        self.pretrained = pretrained
        self.key_head = nn.Linear(d_model, num_keys)
        self.frame_mode = frame_mode

    def forward(self, frames):
        if self.pretrained:
            with torch.no_grad():
                B, seq_len, C, H, W = frames.shape
                frames_flat = frames.view(B * seq_len, C, H, W)
                patch_size = self.qwen.config.vision_config.patch_size  
                grid_h = H // patch_size
                grid_w = W // patch_size
                
                num_images = B * seq_len
                grid_thw = torch.tensor([[1, grid_h, grid_w]] * num_images, dtype=torch.long, device=frames_flat.device)
                
                vision_outputs = self.qwen.visual(frames_flat, grid_thw=grid_thw)
                
                if isinstance(vision_outputs, tuple):
                    vision_outputs = vision_outputs[0]

                h = vision_outputs.mean(dim=1)
                h = h.view(B, seq_len, -1)
            h = h.float()
        else: 
            h = self.encoder(frames) 
            
        key_logits = self.key_head(h)

        return key_logits