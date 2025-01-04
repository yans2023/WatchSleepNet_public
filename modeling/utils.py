import os
import torch
import sys
from datetime import datetime
import torch
import torch
from torch.nn.utils.rnn import PackedSequence


def print_model_info(model_name, model, input_tensor, lengths_tensor=None):
    """
    Print the input and output shape of each layer in the model by registering forward hooks,
    and also print the number of total and trainable model parameters. The model runs on CPU.

    Parameters:
    - model_name (str): The name of the model.
    - model (torch.nn.Module): The PyTorch model.
    - input_tensor (torch.Tensor): Input data to pass through the model.
    - lengths_tensor (torch.Tensor, optional): Lengths of the sequences (for models with variable-length inputs). Default is None.
    """
    print(f"===== {model_name} =====")

    # Move the model to the CPU
    model = model.to("cpu")

    # Print number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Dictionary to store input and output shapes of each layer
    layer_shapes = {}

    def hook_fn(module, input, output):
        """
        Hook function to capture input and output shapes of a layer.
        """
        class_name = module.__class__.__name__
        module_idx = len(layer_shapes) + 1
        layer_key = f"{module_idx}. {class_name}"

        # Handle PackedSequence inputs and outputs
        if isinstance(input[0], PackedSequence):
            input_shape = list(
                input[0].data.size()
            )  # Unpack the PackedSequence input tensor
        else:
            input_shape = (
                list(input[0].size())
                if isinstance(input, tuple)
                else list(input.size())
            )

        if isinstance(output, PackedSequence):
            output_shape = list(
                output.data.size()
            )  # Unpack the PackedSequence output tensor
        else:
            output_shape = (
                list(output.size()) if isinstance(output, torch.Tensor) else None
            )

        layer_shapes[layer_key] = {
            "input_shape": input_shape,
            "output_shape": output_shape,
        }

    # Register hooks for each layer
    hooks = []
    full_layer_names = []
    for name, layer in model.named_modules():
        hook = layer.register_forward_hook(hook_fn)
        hooks.append(hook)
        full_layer_names.append(name)

    # Move input tensors to the CPU
    input_tensor = input_tensor.to("cpu")
    if lengths_tensor is not None:
        lengths_tensor = lengths_tensor.to("cpu")
        model(input_tensor, lengths_tensor)  # Forward pass with lengths
    else:
        model(input_tensor)  # Forward pass without lengths

    # Remove hooks after forward pass
    for hook in hooks:
        hook.remove()

    # Print layer-wise information with full layer names (first loop: print layer names)
    print(f"{'Layer Name':<40}")
    print("=" * 100)
    for idx, layer_name in enumerate(full_layer_names):
        layer_key = f"{idx + 1}. {layer_name}"
        print(f"{layer_key:<40}")
    print("=" * 100)

    # Print input and output shapes with layer number and type (second loop)
    print(f"{'Layer Number and Type':<40} {'Input Shape':<30} {'Output Shape':<30}")
    print("=" * 100)
    for idx, (layer_key, shapes) in enumerate(layer_shapes.items()):
        input_shape = str(shapes["input_shape"])
        output_shape = str(shapes["output_shape"])
        print(f"{layer_key:<40} {input_shape:<30} {output_shape:<30}")
    print("=" * 100)


def model_memory_usage(model, input_size, lengths, dtype=torch.float32):
    # Define total_params using a dictionary to avoid scoping issues in hooks
    param_info = {"total_params": 0}

    def hook(module, input, output):
        param_info["total_params"] += input[0].numel()
        param_info["total_params"] += output.numel()

    hooks = []
    try:
        for layer in model.modules():
            if isinstance(layer, (torch.nn.TransformerEncoderLayer, torch.nn.Linear, torch.nn.TransformerEncoder)):
                hooks.append(layer.register_forward_hook(hook))
        # Dummy input based on the input size provided
        input = torch.zeros(input_size, dtype=dtype)
        if next(model.parameters()).is_cuda:
            input = input.cuda()
        with torch.no_grad():
            model(input)
    finally:
        for hook in hooks:
            hook.remove()

    # Each parameter requires memory to store the value and the gradient
    memory_bytes = param_info["total_params"] * 2 * dtype.itemsize
    return memory_bytes / (1024**2)  # Convert to Megabytes


def generate_checkpoint_filename(config_file, fold, base_dir="checkpoints"):
    # Extract the config filename without extension
    config_name = os.path.basename(config_file).replace(".json", "")

    # Get the current date and time
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate the checkpoint filename
    filename = f"{config_name}_fold-{fold}_{current_time}.pt"

    # Create the full path
    filepath = os.path.join(base_dir, filename)

    return filepath
