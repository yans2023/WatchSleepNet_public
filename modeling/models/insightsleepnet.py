# adapted from: https://github.com/TheMrGhostman/InceptionTime-Pytorch/blob/master/inception.py#L99
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import gc
import psutil

def pass_through(X):
    return X

class Inception(nn.Module):
    def __init__(
        self,
        in_channels,
        n_filters,
        kernel_sizes=[9, 19, 39],
        bottleneck_channels=32,
        activation=nn.ReLU(),
        return_indices=False,
    ):
        super(Inception, self).__init__()
        self.return_indices = return_indices
        if in_channels > 1:
            self.bottleneck = nn.Conv1d(
                in_channels=in_channels,
                out_channels=bottleneck_channels,
                kernel_size=1,
                stride=1,
                bias=False,
            )
        else:
            self.bottleneck = pass_through
            bottleneck_channels = 1

        self.conv_from_bottleneck_1 = nn.Conv1d(
            in_channels=bottleneck_channels,
            out_channels=n_filters,
            kernel_size=kernel_sizes[0],
            stride=1,
            padding=kernel_sizes[0] // 2,
            bias=False,
        )
        self.conv_from_bottleneck_2 = nn.Conv1d(
            in_channels=bottleneck_channels,
            out_channels=n_filters,
            kernel_size=kernel_sizes[1],
            stride=1,
            padding=kernel_sizes[1] // 2,
            bias=False,
        )
        self.conv_from_bottleneck_3 = nn.Conv1d(
            in_channels=bottleneck_channels,
            out_channels=n_filters,
            kernel_size=kernel_sizes[2],
            stride=1,
            padding=kernel_sizes[2] // 2,
            bias=False,
        )
        self.max_pool = nn.MaxPool1d(
            kernel_size=3, stride=1, padding=1, return_indices=return_indices
        )
        self.conv_from_maxpool = nn.Conv1d(
            in_channels=in_channels,
            out_channels=n_filters,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm1d(num_features=4 * n_filters)
        self.activation = activation

    def forward(self, X):
        Z_bottleneck = self.bottleneck(X)

        if self.return_indices:
            Z_maxpool, indices = self.max_pool(X)
        else:
            Z_maxpool = self.max_pool(X)

        Z1 = self.conv_from_bottleneck_1(Z_bottleneck)
        Z2 = self.conv_from_bottleneck_2(Z_bottleneck)
        Z3 = self.conv_from_bottleneck_3(Z_bottleneck)
        Z4 = self.conv_from_maxpool(Z_maxpool)

        Z = torch.cat([Z1, Z2, Z3, Z4], axis=1)
        Z = self.activation(self.batch_norm(Z))

        if self.return_indices:
            return Z, indices
        else:
            return Z


class InceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        n_filters=32,
        kernel_sizes=[9, 19, 39],
        bottleneck_channels=32,
        use_residual=True,
        activation=nn.ReLU(),
        return_indices=False,
    ):
        super(InceptionBlock, self).__init__()
        self.use_residual = use_residual
        self.return_indices = return_indices
        self.activation = activation

        self.inception_1 = Inception(
            in_channels=in_channels,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
            return_indices=return_indices,
        )
        self.inception_2 = Inception(
            in_channels=4 * n_filters,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
            return_indices=return_indices,
        )
        self.inception_3 = Inception(
            in_channels=4 * n_filters,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
            return_indices=return_indices,
        )

        if self.use_residual:
            self.residual = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=4 * n_filters,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                nn.BatchNorm1d(num_features=4 * n_filters),
            )

    def forward(self, X):
        """
        If self.return_indices is True, each Inception will also return indices for MaxUnpool,
        but for this example we ignore that and assume no unpooling is used in forward.
        """
        if self.return_indices:
            Z1, i1 = self.inception_1(X)
            Z2, i2 = self.inception_2(Z1)
            Z3, i3 = self.inception_3(Z2)
            Z = Z3
            all_indices = [i1, i2, i3]
        else:
            Z = self.inception_1(X)
            Z = self.inception_2(Z)
            Z = self.inception_3(Z)
            all_indices = None

        if self.use_residual:
            Z = Z + self.residual(X)
            Z = self.activation(Z)

        if self.return_indices:
            return Z, all_indices
        else:
            return Z


# ---- Main InsightSleepNet model ----
class InsightSleepNet(nn.Module):
    def __init__(
        self,
        input_size=750,
        output_size=3,
        n_filters=32,
        bottleneck_channels=32,
        kernel_sizes=[9, 19, 39],
        num_inception_blocks=1,
        use_residual=True,
        dropout_rate=0.2,
        activation=nn.ReLU(),
    ):
        """
        :param input_size: Size of the input signal (e.g., 750).
        :param output_size: Number of output classes (e.g., 3 for classification).
        :param n_filters: Number of filters for each Inception branch.
        :param bottleneck_channels: Bottleneck channels in each Inception block.
        :param kernel_sizes: List of 3 kernel sizes for the Inception block.
        :param num_inception_blocks: How many InceptionBlocks to stack in series.
        :param use_residual: Whether each InceptionBlock includes a residual connection.
        :param dropout_rate: Dropout rate after the final pooling (and before FC).
        :param activation: Activation function to use, e.g., nn.ReLU().
        """
        super(InsightSleepNet, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.n_filters = n_filters
        self.bottleneck_channels = bottleneck_channels
        self.kernel_sizes = kernel_sizes
        self.num_inception_blocks = num_inception_blocks
        self.use_residual = use_residual
        self.dropout_rate = dropout_rate
        self.activation = activation

        # 1) Initial conv to reduce dimension or transform input
        #    Input shape after flattening: (batch_size*num_epochs, 1, input_size)
        self.initial_conv = nn.Conv1d(
            in_channels=1,
            out_channels=n_filters,
            kernel_size=40,
            stride=20
        )

        # 2) Build multiple InceptionBlocks
        blocks = []
        in_channels_block = n_filters
        for _ in range(num_inception_blocks):
            block = InceptionBlock(
                in_channels=in_channels_block,
                n_filters=n_filters,
                kernel_sizes=kernel_sizes,
                bottleneck_channels=bottleneck_channels,
                use_residual=use_residual,
                activation=activation,
                return_indices=False,
            )
            blocks.append(block)
            # If use_residual=True, the block’s output has 4*n_filters channels
            # because each InceptionBlock merges 4 branches => 4*n_filters,
            # then adds a residual (also 4*n_filters).
            in_channels_block = 4 * n_filters if use_residual else 4 * n_filters

        self.inception_blocks = nn.Sequential(*blocks)

        # 3) A final pooling to reduce the time dimension to 1
        self.final_pooling = nn.AdaptiveAvgPool1d(1)

        # 4) Fully connected layer for classification
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(in_channels_block, output_size)

    def forward(self, x, lengths=None):
        """
        Input: x.shape = (batch_size, num_epochs, input_size)
        1) Flatten to (batch_size*num_epochs, 1, input_size)
        2) initial_conv + activation => shape: (B*T, n_filters, new_len)
        3) pass through InceptionBlocks => (B*T, 4*n_filters, new_len_post_inception)
        4) final_pooling => (B*T, 4*n_filters, 1) => flatten => (B*T, 4*n_filters)
        5) dropout => fc => (B*T, output_size) => reshape => (B, T, output_size)
        6) optional masking if lengths is not None
        """
        batch_size, num_epochs, signal_len = x.shape
        # Flatten (B, T, L) -> (B*T, 1, L)
        x = x.view(batch_size * num_epochs, 1, signal_len)

        # Initial conv -> shape: (B*T, n_filters, new_len)
        x = self.initial_conv(x)
        x = self.activation(x)

        # Inception blocks -> shape: (B*T, 4*n_filters, new_len_post_inception)
        x = self.inception_blocks(x)

        # Adaptive pooling to reduce time dim to 1 => (B*T, 4*n_filters, 1)
        x = self.final_pooling(x)
        x = x.squeeze(-1)  # => (B*T, 4*n_filters)

        # Dropout + fully-connected => (B*T, output_size)
        x = self.dropout(x)
        x = self.fc(x)

        # Reshape back => (batch_size, num_epochs, output_size)
        x = x.view(batch_size, num_epochs, self.output_size)

        # Optional masking
        if lengths is not None:
            lengths = lengths.to(x.device)
            mask = torch.arange(x.size(1), device=x.device).expand(
                x.size(0), x.size(1)
            ) < lengths.unsqueeze(1)
            x[~mask] = -1e9

        return x


def get_gpu_memory_usage_mb():
    """
    Returns (allocated_mb, reserved_mb) for the current CUDA device.
    """
    if not torch.cuda.is_available():
        return (0.0, 0.0)
    # Synchronize to ensure all ops are done
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    return (allocated / (1024 ** 2), reserved / (1024 ** 2))


def test_insight_configurations():
    """
    Tests multiple configurations of the new parametric InsightSleepNet by:
      1. Building each config,
      2. Running a forward pass with dummy data,
      3. Measuring GPU memory usage before/after,
      4. Doing a quick training step (one forward+backward).
    """

    # Example configurations for the new InsightSleepNet signature:
    #   (n_filters, bottleneck_channels, kernel_sizes, num_inception_blocks, use_residual, dropout_rate, activation)
    # Adjust or add more as you see fit.
    configs = [
        {
            "desc": "Smaller config",
            "n_filters": 24,
            "bottleneck_channels": 16,
            "kernel_sizes": [5, 11, 23],
            "num_inception_blocks": 1,
            "use_residual": True,
            "dropout_rate": 0.2,
            "activation": nn.ReLU(),
        },
        {
            "desc": "Medium config",
            "n_filters": 32,
            "bottleneck_channels": 24,
            "kernel_sizes": [7, 15, 31],
            "num_inception_blocks": 2,
            "use_residual": True,
            "dropout_rate": 0.2,
            "activation": nn.ReLU(),
        },
        {
            "desc": "Original-like config",
            "n_filters": 32,
            "bottleneck_channels": 32,
            "kernel_sizes": [9, 19, 39],
            "num_inception_blocks": 3,
            "use_residual": True,
            "dropout_rate": 0.2,
            "activation": nn.ReLU(),
        },
    ]

    # Basic test setup
    batch_size = 2       # e.g., 2
    num_epochs = 1100    # 1100 time segments
    input_length = 750   # each segment is length 750
    output_size = 3      # e.g. 3 classes

    for cfg in configs:
        print(f"\n=== Testing configuration: {cfg['desc']} ===")
        print(cfg)

        # Import your updated InsightSleepNet that uses these hyperparams
        # Adjust if your model is in a different module.

        # Instantiate with the chosen hyperparameters
        model = InsightSleepNet(
            input_size=input_length,
            output_size=output_size,
            n_filters=cfg["n_filters"],
            bottleneck_channels=cfg["bottleneck_channels"],
            kernel_sizes=cfg["kernel_sizes"],
            num_inception_blocks=cfg["num_inception_blocks"],
            use_residual=cfg["use_residual"],
            dropout_rate=cfg["dropout_rate"],
            activation=cfg["activation"],
        ).cuda()

        # Create dummy inputs and random lengths
        dummy_input = torch.rand(batch_size, num_epochs, input_length).cuda()
        dummy_lengths = torch.randint(low=800, high=1101, size=(batch_size,), device="cuda")

        # Print parameter count
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Parameter count: {param_count}")

        # Measure memory before forward
        alloc_before, reserved_before = get_gpu_memory_usage_mb()
        print(f"Memory before forward pass: allocated={alloc_before:.2f}MB, reserved={reserved_before:.2f}MB")

        # Forward pass (no gradient)
        model.train()
        with torch.no_grad():
            output = model(dummy_input, lengths=dummy_lengths)

        # Measure memory after forward
        alloc_after, reserved_after = get_gpu_memory_usage_mb()
        print(f"Memory after forward pass: allocated={alloc_after:.2f}MB, reserved={reserved_after:.2f}MB")

        # Print shapes
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")

        # Quick training step to ensure no shape or memory issues with backprop
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss(ignore_index=-1)

        # Create a dummy label (batch_size, num_epochs)
        dummy_label = torch.randint(0, output_size, (batch_size, num_epochs), device="cuda")

        # Mask out invalid positions in the label based on dummy_lengths
        for i in range(batch_size):
            dummy_label[i, dummy_lengths[i] :] = -1

        optimizer.zero_grad()

        # Forward pass with gradient
        output_train = model(dummy_input, lengths=dummy_lengths)
        loss = criterion(output_train.reshape(-1, output_size), dummy_label.reshape(-1))

        # Backward + step
        loss.backward()
        optimizer.step()

        # Memory after training step
        alloc_train_after, reserved_train_after = get_gpu_memory_usage_mb()
        print(f"Memory after training step: allocated={alloc_train_after:.2f}MB, reserved={reserved_train_after:.2f}MB")
        print(f"Dummy training loss: {loss.item():.4f}")

        # Cleanup
        del model, dummy_input, output, output_train, dummy_label
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(1)  # small pause to let GPU catch up


if __name__ == "__main__":
    test_insight_configurations()
