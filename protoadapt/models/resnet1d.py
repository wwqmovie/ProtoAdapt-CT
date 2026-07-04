import torch.nn as nn
import torch
class ResidualBlock1D(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2)  
        self.bn1 = nn.BatchNorm1d(dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)  
        self.bn2 = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        # ：SE
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x

        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.bn2(out)

        # SE
        se_weight = self.se(out.unsqueeze(-1)).unsqueeze(-1)
        out = out * se_weight.squeeze(-1)

        out = self.relu(out + residual)
        out = self.dropout(out)
        return out
# class ResidualBlock1D(nn.Module):
#     def __init__(self, dim, dropout=0.1):
#         super().__init__()
#         self.fc1 = nn.Linear(dim, dim * 2)
#         self.bn1 = nn.BatchNorm1d(dim * 2)
#         self.fc2 = nn.Linear(dim * 2, dim)
#         self.bn2 = nn.BatchNorm1d(dim)
#         self.relu = nn.ReLU(inplace=True)
#
#         # dropout
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout * 1.5)  # dropout
#         self.dropout_shortcut = nn.Dropout(dropout * 0.5)  # shortcutdropout
#
#         # SE（）
#         self.se = nn.Sequential(
#             nn.AdaptiveAvgPool1d(1),
#             nn.Flatten(),
#             nn.Linear(dim, dim // 8),  
#             nn.ReLU(),
#             nn.Linear(dim // 8, dim),
#             nn.Sigmoid()
#         ) if dim >= 32 else None
#
#     def forward(self, x):
#         residual = x
#
#         out = self.fc1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.dropout1(out)
#
#         out = self.fc2(out)
#         out = self.bn2(out)
#
#         # SE
#         if self.se is not None:
#             se_weight = self.se(out.unsqueeze(-1)).unsqueeze(-1)
#             out = out * se_weight.squeeze(-1)
#
#         # Shortcut dropout（）
#         residual = self.dropout_shortcut(residual)
#
#         out = self.relu(out + residual)
#         out = self.dropout2(out)
#         return out
# huaxi

class ResNet1D(nn.Module):
    def __init__(self, input_dim=400, num_classes=2, hidden_dims=[512, 256, 128, 64]):
        super().__init__()
        
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                ResidualBlock1D(hidden_dims[0], dropout=0.1),
                ResidualBlock1D(hidden_dims[0], dropout=0.1)
            ),
            nn.Sequential(
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.BatchNorm1d(hidden_dims[1]),
                nn.ReLU(inplace=True),
                ResidualBlock1D(hidden_dims[1], dropout=0.15),
                ResidualBlock1D(hidden_dims[1], dropout=0.15)
            ),
            nn.Sequential(
                nn.Linear(hidden_dims[1], hidden_dims[2]),
                nn.BatchNorm1d(hidden_dims[2]),
                nn.ReLU(inplace=True),
                ResidualBlock1D(hidden_dims[2], dropout=0.2),
                ResidualBlock1D(hidden_dims[2], dropout=0.2)
            )
        ])
        
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[2] * 2, hidden_dims[3]),  
            nn.BatchNorm1d(hidden_dims[3]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dims[3], num_classes)
        )
        
        # （）
        self.aux_classifier = nn.Linear(hidden_dims[2], num_classes)
        
    def forward(self, x, return_features=False):
        
        x = self.input_proj(x)
        
        
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        
        
        global_feat = self.global_pool(x.unsqueeze(-1)).squeeze(-1)
        
        # （block）
        local_feat = x
        
        
        combined = torch.cat([global_feat, local_feat], dim=1)
        
        
        out = self.classifier(combined)
        
        if return_features:
            return out, combined
        return out

class ResNet1D_Simplified(nn.Module):
    """ProtoAdapt-CT."""

    def __init__(self, input_dim=400, num_classes=2):
        super().__init__()

        
        hidden_dims = [256, 128, 64]  #  [512, 256, 128, 64]

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        
        self.blocks = nn.ModuleList([
            ResidualBlock1D(hidden_dims[0], dropout=0.2),
            ResidualBlock1D(hidden_dims[0], dropout=0.2),
            nn.Sequential(
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.BatchNorm1d(hidden_dims[1]),
                nn.ReLU(inplace=True),
                ResidualBlock1D(hidden_dims[1], dropout=0.25),
            ),
            nn.Sequential(
                nn.Linear(hidden_dims[1], hidden_dims[2]),
                nn.BatchNorm1d(hidden_dims[2]),
                nn.ReLU(inplace=True),
                ResidualBlock1D(hidden_dims[2], dropout=0.3),
            )
        ])

        
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(hidden_dims[2], num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        return self.classifier(x)


if __name__ == '__main__':
    model = ResNet1D()
    x=torch.randn((3, 400))
    out = model(x)