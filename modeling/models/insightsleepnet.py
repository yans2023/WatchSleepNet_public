import gc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

########################################
# Simple Inception Module
########################################
def pass_through(x):
    return x

class Inception(nn.Module):
    """Single Inception sub-module producing 4*n_filters output channels."""
    def __init__(
        self,
        in_channels: int,
        n_filters: int,
        kernel_sizes=[9, 19, 39],
        bottleneck_channels=32,
        activation=nn.ReLU(),
        use_residual=True,
    ):
        super().__init__()
        self.activation = activation
        self.use_residual = use_residual
        
        if in_channels > 1:
            self.bottleneck = nn.Conv1d(
                in_channels, bottleneck_channels, kernel_size=1, stride=1, bias=False
            )
        else:
            self.bottleneck = pass_through
            bottleneck_channels = 1

        self.conv1 = nn.Conv1d(
            bottleneck_channels,
            n_filters,
            kernel_size=kernel_sizes[0],
            padding=kernel_sizes[0] // 2,
            bias=False,
        )
        self.conv2 = nn.Conv1d(
            bottleneck_channels,
            n_filters,
            kernel_size=kernel_sizes[1],
            padding=kernel_sizes[1] // 2,
            bias=False,
        )
        self.conv3 = nn.Conv1d(
            bottleneck_channels,
            n_filters,
            kernel_size=kernel_sizes[2],
            padding=kernel_sizes[2] // 2,
            bias=False,
        )
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_from_pool = nn.Conv1d(in_channels, n_filters, kernel_size=1, bias=False)

        self.bn = nn.BatchNorm1d(num_features=4 * n_filters)

        if self.use_residual:
            self.residual_conv = nn.Conv1d(in_channels, 4 * n_filters, kernel_size=1)
            self.residual_bn = nn.BatchNorm1d(4 * n_filters)

    def forward(self, x):
        z_bottleneck = self.bottleneck(x)

        z1 = self.conv1(z_bottleneck)
        z2 = self.conv2(z_bottleneck)
        z3 = self.conv3(z_bottleneck)

        pm = self.pool(x)
        z4 = self.conv_from_pool(pm)

        out = torch.cat([z1, z2, z3, z4], dim=1)
        out = self.bn(out)
        out = self.activation(out)

        if self.use_residual:
            res = self.residual_bn(self.residual_conv(x))
            out = self.activation(out + res)
        return out

########################################
# One "Block" = 3 Inception modules in sequence
########################################
class InceptionBlock(nn.Module):
    """
    Similar pattern: 3 sequential Inceptions, each producing 4 * n_filters from its own n_filters.
    But we fix the 'n_filters' for each sub-layer the same here. 
    If you want to be extremely flexible, you can also param each sub-Inception individually.
    """
    def __init__(
        self,
        in_channels: int,
        n_filters: int,
        kernel_sizes=[9, 19, 39],
        bottleneck_channels=32,
        activation=nn.ReLU(),
        use_residual=True,
    ):
        super().__init__()
        self.inception1 = Inception(
            in_channels,
            n_filters,
            kernel_sizes,
            bottleneck_channels,
            activation,
            use_residual=True,
        )
        self.inception2 = Inception(
            4 * n_filters,
            n_filters,
            kernel_sizes,
            bottleneck_channels,
            activation,
            use_residual=True,
        )
        self.inception3 = Inception(
            4 * n_filters,
            n_filters,
            kernel_sizes,
            bottleneck_channels,
            activation,
            use_residual=True,
        )

    def forward(self, x):
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        return x

########################################
# A tiny TCN
########################################
class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        dilation,
        padding,
        dropout=0.1,
    ):
        super().__init__()
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, dilation=dilation)
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, dilation=dilation)
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )
        self.relu = nn.ReLU()
        self.padding = padding

    def forward(self, x):
        out = self.conv1(x)
        # If you want strict 'causal' shape, chomp padding
        if self.padding > 0:
            out = out[:, :, :-self.padding].contiguous()
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        if self.padding > 0:
            out = out[:, :, :-self.padding].contiguous()
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, in_channels, channel_sizes, kernel_size=8, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(channel_sizes):
            dilation_size = 2 ** i
            input_ch = in_channels if i == 0 else channel_sizes[i - 1]
            padding = (kernel_size - 1) * dilation_size
            block = TemporalBlock(
                input_ch, out_ch, kernel_size,
                stride=1,
                dilation=dilation_size,
                padding=padding,
                dropout=dropout
            )
            layers.append(block)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

########################################
# TimeDistributed
########################################
class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=True):
        super().__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        # x: (B, T, F)
        b, t, f = x.shape
        x2 = x.reshape(b * t, f)
        y = self.module(x2)
        return y.reshape(b, t, -1)

########################################
# Parametric "InsightSleepNet" Old-Style
########################################

class InsightSleepNet(nn.Module):
    """
    A more general version of the old architecture that:
      1) Takes in a list of block-configs, each describing a single InceptionBlock or the sub-layers
      2) Builds them in sequence
      3) Optionally does final pooling => shape (B, final_ch, pool_size)
      4) Then has a 1x1 conv, time-distributed linear, TCN, etc.
    """
    def __init__(
        self,
        input_size=750,
        output_size=3,
        initial_conv_out=32,
        block_configs=None,  
        # Example: block_configs = [
        #   dict(in_channels=32,  n_filters=8,  bottleneck_channels=8,  kernel_sizes=[5,11,23], use_residual=True),
        #   dict(in_channels=32,  n_filters=16, bottleneck_channels=16, kernel_sizes=[5,11,23], use_residual=True),
        #   ...
        # ],
        final_pool_size=1100,
        dropout_rate=0.2,
        activation=nn.ReLU(),
    ):
        super().__init__()

        # 1) Initial conv
        self.initial_conv = nn.Conv1d(1, initial_conv_out, kernel_size=40, stride=20)
        self.relu = nn.ReLU()

        # 2) Build the sequence of InceptionBlocks
        #    We'll store them in a nn.ModuleList or nn.Sequential
        self.block_list = nn.ModuleList()
        if block_configs is None:
            block_configs = []
        for cfg in block_configs:
            # Build an InceptionBlock
            block = InceptionBlock(
                in_channels=cfg["in_channels"],
                n_filters=cfg["n_filters"],
                kernel_sizes=cfg.get("kernel_sizes", [9,19,39]),
                bottleneck_channels=cfg.get("bottleneck_channels", 32),
                activation=activation,
                use_residual=cfg.get("use_residual", True),
            )
            self.block_list.append(block)

        # 3) Adaptive pooling to final_pool_size
        self.final_pool = nn.AdaptiveAvgPool1d(output_size=final_pool_size)

        # 4) Suppose after the last block, the channel dimension is 4 * n_filters_of_last_block
        #    Let's guess that from the final block config:
        if len(block_configs) > 0:
            last_n_filters = block_configs[-1]["n_filters"]
            final_ch = 4 * last_n_filters
        else:
            final_ch = initial_conv_out  # fallback if no blocks at all

        # Now a 1x1 conv => e.g. (B, final_ch, final_pool_size) => (B, half_of_that, final_pool_size)
        half_ch = final_ch // 2 if final_ch >= 2 else final_ch
        self.conv1 = nn.Conv1d(final_ch, half_ch, kernel_size=1)

        # time-dist => (B, final_pool_size, half_ch) => linear => e.g. out_feat=128
        self.tdd = TimeDistributed(nn.Linear(half_ch, 128), batch_first=True)

        # TCN => from 128 -> a list of channels
        self.tcn = TemporalConvNet(in_channels=128, channel_sizes=[64,64,64,64,64], kernel_size=8, dropout=dropout_rate)

        # final => (B, out_size, final_pool_size)
        self.downsample = nn.Conv1d(64, output_size, kernel_size=1)

    def forward(self, x, lengths=None):
        # x: (B, T, S=750)
        b, t, s = x.shape
        # Flatten => (B, 1, T*S)
        x = x.view(b, 1, t*s)
        x = self.initial_conv(x)
        x = self.relu(x)

        # Pass through each InceptionBlock in block_list
        for block in self.block_list:
            x = block(x)  # shape changes with each block

        # final pool => shape (B, final_ch, final_pool_size)
        x = self.final_pool(x)

        # conv => shape (B, half_ch, final_pool_size)
        x = self.conv1(x)
        # => transpose => (B, final_pool_size, half_ch)
        x = x.transpose(1, 2)

        # time-dist => shape (B, final_pool_size, 128)
        x = self.tdd(x)

        # => transpose => (B, 128, final_pool_size)
        x = x.transpose(1, 2)

        # TCN => shape (B,64, final_pool_size)
        x = self.tcn(x)

        # final => (B,output_size, final_pool_size)
        x = self.downsample(x)
        # => (B, final_pool_size, output_size)
        x = x.transpose(1, 2)

        # optional masking with lengths
        if lengths is not None:
            lengths = lengths.to(x.device)
            mask = torch.arange(x.size(1), device=x.device).expand(x.size(0), x.size(1)) < lengths.unsqueeze(1)
            x[~mask] = -1e9

        return x

#########################################################
# Single-step training example with param config
#########################################################
def demo_insightsleepnet_train_step():
    """
    Demonstrates how to build a ParamInsightSleepNet with a custom 'block_configs',
    do one forward+backward pass on dummy data.
    """

    # Suppose we want 6 blocks, each with a progressive in_channels, n_filters, etc.
    # This will replicate your old "6 block" approach, but you can tune them as you like.
    block_configs = [
        # block1
        {
            "in_channels": 32,   # matches initial_conv_out=32
            "n_filters": 8,
            "bottleneck_channels": 8,
            "kernel_sizes": [5, 11, 23],
            "use_residual": True
        },
        # block2
        {
            "in_channels": 32,   # 4 * 8 = 32 from block1
            "n_filters": 16,
            "bottleneck_channels": 16,
            "kernel_sizes": [5, 11, 23],
            "use_residual": True
        },
        # block3
        {
            "in_channels": 64,   # 4 * 16 = 64 from block2
            "n_filters": 16,
            "bottleneck_channels": 16,
            "kernel_sizes": [5, 11, 23],
            "use_residual": True
        },
        # block4
        {
            "in_channels": 64,   # 4 * 16 = 64 from block3
            "n_filters": 32,
            "bottleneck_channels": 16,
            "kernel_sizes": [5, 11, 23],
            "use_residual": True
        },
        # block5
        {
            "in_channels": 128,  # 4 * 32 = 128 from block4
            "n_filters": 64,
            "bottleneck_channels": 32,
            "kernel_sizes": [5, 11, 23],
            "use_residual": True
        },
        # block6
        {
            "in_channels": 256,  # 4 * 64 = 256 from block5
            "n_filters": 128,
            "bottleneck_channels": 32,
            "kernel_sizes": [5, 11, 23],
            "use_residual": True
        },
    ]

    # Build the param model
    model = InsightSleepNet(
        input_size=750,
        output_size=3,
        initial_conv_out=32,   # matches 'in_channels' for block1
        block_configs=block_configs,
        final_pool_size=1100,
        dropout_rate=0.2,
        activation=nn.ReLU()
    ).cuda()

    # Create dummy input => (batch_size, num_epochs, 750)
    batch_size = 2
    num_epochs = 1100
    x_dummy = torch.randn(batch_size, num_epochs, 750, device="cuda")

    # Dummy lengths => partial masking
    lengths_dummy = torch.randint(800, 1101, (batch_size,), device="cuda")

    # Create dummy label => shape (batch_size, num_epochs), ignoring beyond lengths
    y_dummy = torch.randint(0, 3, (batch_size, num_epochs), device="cuda")
    for i in range(batch_size):
        y_dummy[i, lengths_dummy[i]:] = -1  # out-of-range => -1

    # 1 step training
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    optimizer.zero_grad()
    output = model(x_dummy, lengths=lengths_dummy)  # shape => (B, 1100, 3)
    print(f"Forward pass done. Output shape={output.shape}")

    loss = criterion(output.reshape(-1, 3), y_dummy.reshape(-1))
    print(f"Loss = {loss.item():.4f}")

    loss.backward()
    optimizer.step()
    print("Backprop + optimizer step completed successfully.")

    # Cleanup
    del model, x_dummy, y_dummy, output
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    demo_insightsleepnet_train_step()
