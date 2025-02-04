import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import gc
import psutil


class SleepConvNet(nn.Module):
    def __init__(
        self,
        input_size=750,
        target_size=256,
        num_segments=1100,
        num_classes=3,
        dropout_rate=0.2,
        conv_layers_configs=None,
        dilation_layers_configs=None,
        use_residual=True,
    ):
        super(SleepConvNet, self).__init__()

        # Provide default configs if none are provided
        if conv_layers_configs is None:
            # Format: (in_channels, out_channels, kernel_size, dilation)
            conv_layers_configs = [
                (1, 32, 3, 1),
                (32, 64, 3, 1),
                (64, 128, 3, 1),
            ]

        if dilation_layers_configs is None:
            # Format: (in_channels, out_channels, kernel_size, dilation)
            dilation_layers_configs = [
                (128, 128, 7, 2),
                (128, 128, 7, 4),
                (128, 128, 7, 8),
                (128, 128, 7, 16),
                (128, 128, 7, 32),
            ]

        self.use_residual = use_residual

        # Downsample layer
        self.downsample = nn.Upsample(
            size=target_size, mode="linear", align_corners=True
        )

        # Build convolutional blocks
        self.conv_blocks = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        for in_ch, out_ch, k, d in conv_layers_configs:
            block = self._build_conv_block(in_ch, out_ch, k, d, dropout_rate)
            self.conv_blocks.append(block)

            if self.use_residual and in_ch != out_ch:
                # Residual connection to match channel dimensions
                self.residual_convs.append(nn.Conv1d(in_ch, out_ch, kernel_size=1))
            else:
                self.residual_convs.append(None)

        # Determine final number of channels from the last conv layer config
        final_in_channels = conv_layers_configs[-1][1]

        # Final convolution to reduce spatial dimension (time dimension to 1)
        self.final_conv = nn.Conv1d(
            final_in_channels, final_in_channels, kernel_size=32, stride=1
        )

        # Build dilation block
        dilation_layers = []
        for in_ch, out_ch, k, d in dilation_layers_configs:
            padding = (k - 1) * d // 2
            dilation_layers.append(
                nn.Conv1d(in_ch, out_ch, kernel_size=k, dilation=d, padding=padding)
            )
            dilation_layers.append(nn.LeakyReLU())
            dilation_layers.append(nn.Dropout(dropout_rate))
        self.dilation_block = nn.Sequential(*dilation_layers)

        # Output layer
        self.output_layer = nn.Conv1d(
            in_channels=dilation_layers_configs[-1][1],
            out_channels=num_classes,
            kernel_size=1,
        )

    def _build_conv_block(
        self, in_channels, out_channels, kernel_size, dilation, dropout_rate
    ):
        # Compute padding for 'same' length
        padding = (kernel_size - 1) // 2 * dilation
        return nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
            ),
            nn.LeakyReLU(),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
            ),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x, lengths=None):
        batch_size, num_epochs, _ = x.shape
        x = x.view(batch_size * num_epochs, 1, -1)

        # Downsample
        x = self.downsample(x)

        # Pass through convolutional blocks with residual connections
        for conv_block, residual_conv in zip(self.conv_blocks, self.residual_convs):
            residual = x if residual_conv is None else residual_conv(x)
            residual = F.max_pool1d(residual, kernel_size=2)
            x = conv_block(x)
            x += residual

        # Apply final conv to reduce spatial dimension to 1
        x = self.final_conv(x)
        # shape now: (batch_size * num_epochs, final_in_channels, 1)

        # Reshape:
        # Extract final_in_channels from x
        batch_size_times_num_epochs, final_in_channels, time_dim = x.shape
        assert time_dim == 1, "Expected final time dimension to be 1 after final_conv"
        x = x.squeeze(-1)  # (batch_size * num_epochs, final_in_channels)
        x = x.view(
            batch_size, num_epochs, final_in_channels
        )  # (batch_size, num_epochs, final_in_channels)

        # Apply dilation block
        x = x.permute(0, 2, 1)  # (batch_size, final_in_channels, num_epochs)
        x = self.dilation_block(x)

        # Output layer
        x = self.output_layer(x)  # (batch_size, num_classes, num_epochs)
        x = x.permute(0, 2, 1)  # (batch_size, num_epochs, num_classes)

        # Mask if lengths are given
        if lengths is not None:
            lengths = lengths.to(x.device)
            mask = torch.arange(x.size(1), device=x.device).expand(
                x.size(0), x.size(1)
            ) < lengths.unsqueeze(1)
            x[~mask] = -1e9

        return x


# The rest of your testing functions and code would remain as previously provided.
def get_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / (1024**2)  # Convert bytes to MB


def test_sleepconvnet(num_classes=3, model=None):
    if model is None:
        model = SleepConvNet(num_classes=num_classes)
    dummy_input = torch.rand(2, 1100, 750)  # (batch_size, num_epochs, input_size)
    output = model(dummy_input)
    print(
        "test_sleepconvnet -> Output shape:", output.shape
    )  # Should be (batch_size, num_epochs, num_classes)


def check_model_shapes_and_memory(
    input_size, output_size, batch_size, num_epochs, input_length, model=None
):
    if model is None:
        model = SleepConvNet(input_size=input_size, num_classes=output_size)

    # Create dummy data with the correct input shape: (batch_size, num_epochs, input_length)
    dummy_input = torch.rand(batch_size, num_epochs, input_length)

    # Create varying dummy lengths (random between 800 and 1100 for each sample in the batch)
    dummy_lengths = torch.randint(low=800, high=1101, size=(batch_size,))

    # Track memory before forward pass
    memory_before = get_memory_usage()
    print(f"Memory before forward pass: {memory_before:.2f} MB")

    # Perform a forward pass to check the input and output shape
    print(
        "check_model_shapes_and_memory -> Running a forward pass with varying lengths..."
    )

    model.train()
    with torch.no_grad():
        output = model(dummy_input, lengths=dummy_lengths)

    # Track memory after forward pass
    memory_after = get_memory_usage()
    print(f"Memory after forward pass: {memory_after:.2f} MB")

    # Output expected shapes
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Lengths: {dummy_lengths}")

    return model, dummy_input, dummy_lengths


def simulate_training_step(
    input_size, output_size, batch_size, num_epochs, input_length, model=None
):
    print("simulate_training_step -> Checking shapes and performing a forward pass:")

    model, dummy_input, dummy_lengths = check_model_shapes_and_memory(
        input_size, output_size, batch_size, num_epochs, input_length, model=model
    )

    # Create a dummy label (random target) with the same varying lengths
    dummy_label = torch.randint(0, output_size, (batch_size, num_epochs))

    # Adjust the dummy label according to the varying dummy_lengths by setting invalid positions to -1
    for i in range(batch_size):
        dummy_label[i, dummy_lengths[i] :] = (
            -1
        )  # Mark padded regions as invalid (ignored in loss)

    # Print the shapes of dummy inputs and labels before forward pass
    print(f"Dummy input shape: {dummy_input.shape}")
    print(f"Dummy label shape: {dummy_label.shape}")
    print(f"Dummy lengths: {dummy_lengths}")

    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    # Track memory before training step
    memory_before = get_memory_usage()
    print(f"Memory before training step: {memory_before:.2f} MB")

    # Run one gradient descent step
    print(
        "simulate_training_step -> Performing one gradient descent step with varying lengths..."
    )
    start_time = time.time()

    optimizer.zero_grad()

    # Forward pass
    output = model(dummy_input, lengths=dummy_lengths)

    # Print the output shape after forward pass
    print(f"Output shape after forward pass: {output.shape}")

    # Compute loss using the ignore_index for invalid positions
    loss = criterion(output.reshape(-1, output_size), dummy_label.reshape(-1))

    # Print loss for inspection
    print(f"Loss: {loss.item()}")

    # Backward pass
    loss.backward()

    # Optimize
    optimizer.step()

    end_time = time.time()

    # Track memory after training step
    memory_after = get_memory_usage()
    print(f"Memory after training step: {memory_after:.2f} MB")

    print(f"Gradient descent step completed in {end_time - start_time:.2f} seconds")

    # Cleanup to release memory
    gc.collect()


def run_tests_with_configurations():
    # Define a set of three slightly different hyperparameter configurations
    # All are small variations to avoid huge differences in model size.
    configurations = [
        {
            "conv_layers_configs": [
                (1, 32, 3, 1), 
                (32, 64, 3, 1), 
                (64, 128, 3, 1)
            ],
            "dilation_layers_configs": [
                (128, 128, 7, 2),
                (128, 128, 7, 4),
                (128, 128, 7, 8),
                (128, 128, 7, 16),
                (128, 128, 7, 32),
            ],
        },
        {
            # Slightly fewer output channels in the last conv layer
            "conv_layers_configs": [
                (1, 32, 3, 1), 
                (32, 48, 3, 1), 
                (48, 64, 3, 1)
            ],
            "dilation_layers_configs": [
                (64, 64, 7, 2),
                (64, 64, 7, 4),
                (64, 64, 7, 8),
                (64, 64, 7, 16),
            ],
        },
        {
            # Slightly different intermediate channels and fewer dilation layers
            "conv_layers_configs": [
                (1, 24, 3, 1), 
                (24, 48, 3, 1), 
                (48, 64, 3, 1)
            ],
            "dilation_layers_configs": [
                (64, 64, 7, 2), 
                (64, 64, 7, 4), 
                (64, 64, 7, 8)
            ],
        },
    ]

    # Set parameters for testing
    input_size = 750  # Input sequence length
    output_size = 3  # Number of output classes
    batch_size = 8  # Smaller batch for demonstration
    num_epochs = 1100  # Number of epochs/time steps
    input_length = 750

    for i, cfg in enumerate(configurations, start=1):
        print(f"\n===== Testing configuration {i} =====")
        model = SleepConvNet(
            input_size=input_size,
            target_size=256,
            num_segments=num_epochs,
            num_classes=output_size,
            dropout_rate=0.2,
            conv_layers_configs=cfg["conv_layers_configs"],
            dilation_layers_configs=cfg["dilation_layers_configs"],
        )

        # Run the tests using the model with given configuration
        test_sleepconvnet(num_classes=output_size, model=model)
        simulate_training_step(
            input_size, output_size, batch_size, num_epochs, input_length, model=model
        )


if __name__ == "__main__":
    run_tests_with_configurations()
