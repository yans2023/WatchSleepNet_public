import torch
import torch.nn as nn
import torch.nn.functional as F


class ResConvBlock(nn.Module):
    """
    Residual Convolutional Block as described in the SleepPPG-Net.
    Contains 3 1D convolutions followed by max pooling and residual addition.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(ResConvBlock, self).__init__()
        
        # First convolution
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, 
            kernel_size=kernel_size, 
            stride=stride,
            padding=(kernel_size // 2) * dilation,
            dilation=dilation,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        
        # Second convolution
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, 
            kernel_size=kernel_size, 
            stride=stride,
            padding=(kernel_size // 2) * dilation,
            dilation=dilation,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        
        # Third convolution
        self.conv3 = nn.Conv1d(
            out_channels, out_channels, 
            kernel_size=kernel_size, 
            stride=stride,
            padding=(kernel_size // 2) * dilation,
            dilation=dilation,
            bias=False
        )
        self.bn3 = nn.BatchNorm1d(out_channels)
        
        # Max pooling layer
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Downsample for residual if needed
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, 
                    kernel_size=1, 
                    stride=stride, 
                    bias=False
                ),
                nn.BatchNorm1d(out_channels),
            )
        
        # Final activation
        self.leaky_relu_out = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        
    def forward(self, x):
        residual = self.downsample(x)
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leaky_relu1(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.leaky_relu2(out)
        
        # Third conv block
        out = self.conv3(out)
        out = self.bn3(out)
        
        # Add residual
        out += residual
        out = self.leaky_relu_out(out)
        
        # Max pooling
        out = self.pool(out)
        
        return out


class TCNBlock(nn.Module):
    """
    Temporal Convolutional Network block with dilated convolutions
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.2):
        super(TCNBlock, self).__init__()
        
        # 5 dilated convolutions with increasing dilation rate
        self.conv_layers = nn.ModuleList()
        
        # Initialize dilations
        dilations = [1, 2, 4, 8, 16]
        
        for i, dilation in enumerate(dilations):
            padding = (kernel_size - 1) * dilation // 2  # Keep temporal dimension
            conv_layer = nn.Sequential(
                nn.Conv1d(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding,
                    bias=False
                ),
                nn.BatchNorm1d(out_channels),
                nn.LeakyReLU(negative_slope=0.01, inplace=True)
            )
            self.conv_layers.append(conv_layer)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        residual = self.residual(x)
        
        for layer in self.conv_layers:
            x = layer(x)
        
        # Add residual and apply dropout
        x = x + residual
        x = self.dropout(x)
        
        return x


class FeatureExtractor(nn.Module):
    """
    Feature Extraction (FE) module with 8 stacked ResConv blocks
    """
    def __init__(self, input_channels, hidden_dim=64):
        super(FeatureExtractor, self).__init__()
        
        # Initial layer to expand to hidden dimension
        self.init_conv = nn.Conv1d(input_channels, hidden_dim, kernel_size=7, stride=1, padding=3, bias=False)
        self.init_bn = nn.BatchNorm1d(hidden_dim)
        self.init_leaky_relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        
        # 8 stacked ResConv blocks with increasing channel dimensions
        self.layers = nn.ModuleList()
        current_channels = hidden_dim
        
        for i in range(8):
            out_channels = current_channels * 2 if i % 2 == 1 and i > 0 else current_channels
            
            self.layers.append(
                ResConvBlock(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1
                )
            )
            current_channels = out_channels
        
        self.output_dim = current_channels
    
    def forward(self, x):
        # Initial layer
        x = self.init_conv(x)
        x = self.init_bn(x)
        x = self.init_leaky_relu(x)
        
        # Pass through ResConv blocks
        for layer in self.layers:
            x = layer(x)
            
        return x


class TimeDistributedDNN(nn.Module):
    """
    Time-distributed Dense Neural Network for temporal window compression
    """
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super(TimeDistributedDNN, self).__init__()
        
        self.fc1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.leaky_relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.leaky_relu2(x)
        x = self.dropout2(x)
        
        return x


class SleepPPGNet(nn.Module):
    """
    SleepPPG-Net architecture as described in the literature
    """
    def __init__(self, 
                 input_channels=1, 
                 num_classes=4,
                 num_res_blocks=8,
                 tcn_layers=2,
                 hidden_dim=128,
                 dropout_rate=0.2):
        super(SleepPPGNet, self).__init__()
        
        # Feature Extractor (FE)
        self.feature_extractor = FeatureExtractor(input_channels, hidden_dim)
        
        # Get the output dimension from the feature extractor
        fe_output_dim = self.feature_extractor.output_dim
        
        # Time-distributed DNN for temporal windows
        self.time_distributed = TimeDistributedDNN(
            input_dim=fe_output_dim, 
            hidden_dim=hidden_dim,
            dropout=dropout_rate
        )
        
        # Feature Sequencer (FS) - stack of TCN blocks
        self.tcn_blocks = nn.ModuleList()
        for _ in range(tcn_layers):
            self.tcn_blocks.append(TCNBlock(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                dropout=dropout_rate
            ))
        
        # Final Classification (FC) layer
        self.classifier = nn.Conv1d(hidden_dim, num_classes, kernel_size=1)
    
    def forward(self, x, lengths):
        # [batch, seq_len, input_channels] -> [batch*seq_len, input_channels, time_steps]
        # Reshape to handle padding correctly
        batch_size, seq_len, samples_per_segment = x.shape
        
        # Reshape the input for the feature extractor
        x = x.view(batch_size * seq_len, 1, samples_per_segment)
        
        # Pass through Feature Extractor
        x = self.feature_extractor(x)
        
        # Get feature dimension and reshape back to [batch, seq_len, features]
        features_dim = x.size(1)
        time_steps = x.size(2)
        
        # Reshape for sequential processing
        x = x.view(batch_size, seq_len, features_dim, time_steps)
        
        # First, get the valid length for each sequence
        max_length = x.size(1)
        # Create a mask for valid positions
        mask = torch.arange(max_length, device=x.device).expand(len(lengths), max_length) < lengths.unsqueeze(1)
        
        # Reshape for time-distributed and TCN processing
        # Swap seq_len and features dim for 1D convolution along sequence dimension
        x = x.permute(0, 2, 1, 3)
        x = x.contiguous().view(batch_size, features_dim, seq_len, time_steps)
        # Average over the time steps dimension
        x = x.mean(dim=-1)  # Now [batch, features, seq_len]
        
        # Time-distributed DNN
        x = self.time_distributed(x)  # [batch, hidden_dim, seq_len]
        
        # TCN blocks
        for tcn_block in self.tcn_blocks:
            x = tcn_block(x)
        
        # Final classification
        x = self.classifier(x)  # [batch, num_classes, seq_len]
        
        # Reshape for output
        x = x.permute(0, 2, 1)  # [batch, seq_len, num_classes]
        
        return x