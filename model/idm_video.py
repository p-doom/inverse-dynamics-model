import torch.nn as nn
from blocks import Encoder

class KeystrokeIDM(nn.Module):
    def __init__(self, num_actions, d_model=4096, num_transformer_layers=4, num_heads=32, ff_dim=16384, frame_mode="diff"):
        super().__init__()
        self.encoder = Encoder(
            d_model=d_model,
            num_transformer_layers=num_transformer_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            frame_mode=frame_mode
        )

        self.action_head = nn.Linear(d_model, num_actions)

    def forward(self, frames):
        h = self.encoder(frames) 
        action_logits = self.action_head(h)  

        return action_logits