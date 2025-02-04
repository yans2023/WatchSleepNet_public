from torch import nn
import torch
import numpy as np

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(0.2)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        residual = self.downsample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x += residual
        return x


class DeepFeatureExtractor(nn.Module):
    def __init__(self, initial_channels, num_layers=4):
        super(DeepFeatureExtractor, self).__init__()

        layers = []
        current_channels = initial_channels
        out_channels = 16
        stride = 1

        layers.append(
            nn.Conv1d(
                initial_channels, out_channels, kernel_size=7, stride=stride, padding=3
            )
        )

        layers.append(nn.ReLU(inplace=True))
        current_channels = out_channels

        for i in range(num_layers):
            out_channels = 32 * (2**i)
            stride = 4
            layers.append(ResBlock(current_channels, out_channels, stride=stride))
            current_channels = out_channels

        self.layers = nn.Sequential(*layers)
        self.final_conv = nn.Conv1d(256, 256, kernel_size=3, stride=stride)

    def forward(self, x):
        for _, layer in enumerate(self.layers):
            x = layer(x)

        x = self.final_conv(x)

        x = x.view(x.size(0), -1)
        return x


class DeepTCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_layers=3):
        super(DeepTCNBlock, self).__init__()
        self.layers = nn.ModuleList()
        current_channels = in_channels
        dilation = 1

        for _ in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        current_channels,
                        out_channels,
                        kernel_size,
                        padding=(kernel_size - 1) * dilation // 2,
                        dilation=dilation,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                )
            )
            current_channels = out_channels
            dilation *= 2

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class LSTMWithMultiheadAttention(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_heads,
        num_layers,
        batch_first=True,
        bidirectional=True,
    ):
        super(LSTMWithMultiheadAttention, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
        )

        feature_size = hidden_size * 2 if bidirectional else hidden_size
        self.attention = nn.MultiheadAttention(
            feature_size, num_heads, batch_first=batch_first
        )

    def forward(self, x, lengths):
        # lengths must be on CPU for pack_padded_sequence
        if lengths.is_cuda:
            lengths_cpu = lengths.cpu()
        else:
            lengths_cpu = lengths

        # Ensure lengths are of type torch.int64
        lengths_cpu = lengths_cpu.to(dtype=torch.int64)

        # Pack the padded batch of sequences for LSTM
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths_cpu, batch_first=True, enforce_sorted=False
        )

        # Pass through LSTM
        packed_output, _ = self.lstm(packed_input)

        # Unpack output
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Move lengths to the same device as output for mask creation
        lengths_device = lengths.to(output.device)

        # Create key padding mask for attention
        max_seq_len = output.size(1)
        key_padding_mask = torch.arange(max_seq_len, device=output.device).unsqueeze(
            0
        ) >= lengths_device.unsqueeze(1)
        # key_padding_mask: (batch_size, seq_length)

        # Apply attention with key_padding_mask
        attn_output, attn_weights = self.attention(
            output, output, output, key_padding_mask=key_padding_mask
        )
        return attn_output, attn_weights


class LSTMLayer(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        batch_first=True,
        bidirectional=True,
    ):
        super(LSTMLayer, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
        )

    def forward(self, x, lengths):
        # lengths must be on CPU for pack_padded_sequence
        if lengths.is_cuda:
            lengths_cpu = lengths.cpu()
        else:
            lengths_cpu = lengths
        lengths_cpu = lengths_cpu.to(dtype=torch.int64)

        # Pack the padded batch of sequences for LSTM
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths_cpu, batch_first=True, enforce_sorted=False
        )

        # Pass through LSTM
        packed_output, _ = self.lstm(packed_input)

        # Unpack output
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        return output, None


class WatchSleepNet(nn.Module):
    def __init__(
        self,
        num_features,
        num_channels,
        kernel_size,
        hidden_dim,
        num_heads,
        num_layers,
        num_classes,
        tcn_layers,
        use_tcn=True,
        use_attention=True,
    ):
        super(WatchSleepNet, self).__init__()
        self.use_tcn = use_tcn

        self.feature_extractor = DeepFeatureExtractor(initial_channels=num_features)
        feature_dim = 256  # Assuming the feature extractor outputs 256 features

        if self.use_tcn:
            self.tcn = DeepTCNBlock(feature_dim, num_channels, kernel_size, tcn_layers)
            lstm_input_dim = num_channels
        else:
            lstm_input_dim = feature_dim

        self.use_attention = use_attention
        if self.use_attention:
            self.lstm_layer = LSTMWithMultiheadAttention(
                input_size=lstm_input_dim,
                hidden_size=hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
            )
        else:
            self.lstm_layer = LSTMLayer(
                input_size=lstm_input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
            )

        classifier_input_dim = hidden_dim * 2  # Due to bidirectional LSTM
        self.classifier = nn.Linear(classifier_input_dim, num_classes)

    def forward(self, x, lengths):
        batch_size, max_num_segments, num_samples_per_segment = x.shape

        # Reshape x for the feature extractor
        x = x.view(-1, 1, num_samples_per_segment)
        x = self.feature_extractor(x)
        x = x.view(batch_size, max_num_segments, -1)

        # Optional TCN layer
        if self.use_tcn:
            x = x.permute(0, 2, 1)
            x = self.tcn(x)
            x = x.permute(0, 2, 1)

        # Pass through LSTM layer (with or without attention)
        x, attn_weights = self.lstm_layer(x, lengths)

        # Classification
        x = self.classifier(x)

        return x


def display_model_summary(model, x, lengths):
    """
    Registers forward hooks to print input/output shapes of each leaf module
    in the model during a single forward pass.
    """
    # List to store all the hooks so we can remove them later
    hooks = []

    # Define a hook function that prints layer names, input shapes, and output shapes
    def hook_fn(module, module_input, module_output):
        class_name = module.__class__.__name__
        
        # module_input and module_output can be tuples (especially when multiple inputs/outputs)
        if isinstance(module_input, tuple):
            input_shapes = [inp.shape for inp in module_input if hasattr(inp, 'shape')]
        else:
            input_shapes = [module_input.shape] if hasattr(module_input, 'shape') else []
            
        if isinstance(module_output, tuple):
            output_shapes = [out.shape for out in module_output if hasattr(out, 'shape')]
        else:
            output_shapes = [module_output.shape] if hasattr(module_output, 'shape') else []
        
        print(f"{class_name}:\n"
              f"  Input shape(s):  {input_shapes}\n"
              f"  Output shape(s): {output_shapes}\n")

    # Recursively register hook_fn for all leaf modules
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(hook_fn))

    # Perform a forward pass to trigger the hooks
    model.eval()  # Ensures we're in eval mode (no dropout scaling, etc.)
    with torch.no_grad():
        _ = model(x, lengths)

    # Remove hooks after one forward pass to avoid printing repeatedly
    for h in hooks:
        h.remove()


def test_model():

    # Parameters
    batch_size = 4
    num_samples_per_segment = 750  # Example value (adjust as per your data)
    max_num_segments = 1100  # As per your dataset code
    num_features = 1  # Since IBI data has a single channel
    num_classes = 3  # As per your remapped labels
    num_channels = 64
    kernel_size = 3
    hidden_dim = 128
    num_heads = 4
    num_layers = 2
    tcn_layers = 3

    # Generate random data to simulate dataloader outputs
    ibis_list = []
    labels_list = []
    lengths_list = []
    for _ in range(batch_size):
        num_segments = np.random.randint(
            100, max_num_segments + 1
        )  # Random sequence length
        lengths_list.append(num_segments)
        ibis = torch.randn(num_segments, num_samples_per_segment)
        labels = torch.randint(0, num_classes, (num_segments,))
        ibis_list.append(ibis)
        labels_list.append(labels)

    # Pad sequences
    ibis_padded = nn.utils.rnn.pad_sequence(
        ibis_list, batch_first=True, padding_value=0
    )
    labels_padded = nn.utils.rnn.pad_sequence(
        labels_list, batch_first=True, padding_value=-1
    )
    lengths = torch.tensor(lengths_list, dtype=torch.int64)

    # Move tensors to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ibis_padded = ibis_padded.to(device)
    labels_padded = labels_padded.to(device)
    # lengths stays on CPU for the packing operations

    # Define configurations to test
    configs = [
        {"use_tcn": True, "use_attention": True},
        {"use_tcn": False, "use_attention": True},
        {"use_tcn": True, "use_attention": False},
        {"use_tcn": False, "use_attention": False},
    ]

    for config in configs:
        print(
            f"\nTesting model with use_tcn={config['use_tcn']}, use_attention={config['use_attention']}"
        )
        model = WatchSleepNet(
            num_features=num_features,
            num_channels=num_channels,
            kernel_size=kernel_size,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_classes=num_classes,
            tcn_layers=tcn_layers,
            use_tcn=config["use_tcn"],
            use_attention=config["use_attention"],
        )

        model = model.to(device)

        # Display the model summary (shapes) once per configuration
        print("\n--- MODEL LAYER SHAPES ---")
        display_model_summary(model, ibis_padded, lengths)

        # Actual forward pass (with gradient) to check for runtime issues
        model.train()
        outputs = model(ibis_padded, lengths)
        max_length = labels_padded.size(1)
        lengths_device = lengths.to(device)
        mask = torch.arange(max_length, device=device).expand(len(lengths), max_length) < lengths_device.unsqueeze(1)

        # Flatten outputs and labels
        outputs_flat = outputs.reshape(-1, num_classes)
        labels_flat = labels_padded.view(-1)
        mask_flat = mask.view(-1)

        # Filter out padded positions
        outputs_masked = outputs_flat[mask_flat]
        labels_masked = labels_flat[mask_flat]

        # Remove positions where labels are -1 (padding)
        valid_indices = labels_masked != -1
        outputs_valid = outputs_masked[valid_indices]
        labels_valid = labels_masked[valid_indices]

        # Compute loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs_valid, labels_valid)
        print(f"Loss: {loss.item()}")

        # Backward pass
        loss.backward()
        print("Backward pass successful.")


if __name__ == "__main__":
    # Call the test function
    test_model()
