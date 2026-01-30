import torch.nn as nn
from blocks import Encoder
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcesso

class KeystrokeIDM(nn.Module):
    def __init__(self, num_keys, d_model=4096, num_transformer_layers=4, num_heads=32, ff_dim=16384, frame_mode="diff", pretrained=False, pretrained_model_path=None):
        super().__init__()

        if pretrained:        
            self.qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrained_model_path, torch_dtype=torch.bfloat16, device_map="auto")
            self.processor = AutoProcessor.from_pretrained(pretrained_model_path)
            
            for param in self.qwen.parameters():
                param.requires_grad = False
            
            d_model = self.qwen.config.hidden_size
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
            B, seq_len, C, H, W = frames.shape
            frames_flat = frames.view(B * seq_len, C, H, W)
            vision_outputs = self.qwen.visual(frames_flat) 
            h = vision_outputs.mean(dim=1) 
            h = h.view(B, seq_len, -1) 
        else: 
            h = self.encoder(frames) 
            
        key_logits = self.key_head(h)

        return key_logits