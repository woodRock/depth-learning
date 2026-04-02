import torch
import torch.nn as nn

class AcousticLSTM(nn.Module):
    """
    Bi-directional LSTM for processing temporal sequences of acoustic pings.
    Input: (B, 32, 768) where 768 = 256 depth points * 3 frequencies.
    """
    def __init__(self, input_dim=768, hidden_dim=256, num_layers=2, num_classes=4):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # 2 (bi-directional) * hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x: (B, 32 * 768) or (B, 32, 768)
        if len(x.shape) == 2:
            batch_size = x.size(0)
            x = x.view(batch_size, 32, -1)
            
        # lstm_out: (B, 32, hidden_dim * 2)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Concatenate the final hidden states from both directions
        # h_n shape: (num_layers * 2, B, hidden_dim)
        last_hidden = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        
        logits = self.classifier(last_hidden)
        return logits
