import torch
from models.watchsleepnet import WatchSleepNet
from models.insightsleepnet import InsightSleepNet
from models.sleepconvnet import SleepConvNet
from utils import print_model_info  # Import the utility function

# Configuration for the models
BATCH_SIZE = 2
SEQ_LENGTH = 1100  # Sequence length (number of epochs)
FEATURE_SIZE = 750  # Input size for each epoch
NUM_CLASSES = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Generate dummy input data
dummy_input = torch.randn(BATCH_SIZE, SEQ_LENGTH, FEATURE_SIZE).to(DEVICE)

# Dummy sequence lengths (required for WatchSleepNet)
dummy_lengths = torch.tensor([SEQ_LENGTH for _ in range(BATCH_SIZE)]).to(DEVICE)

# Load and test WatchSleepNet
watchsleepnet = WatchSleepNet(
    num_features=1,
    feature_channels=256,
    num_channels=256,
    kernel_size=5,
    hidden_dim=256,
    num_heads=16,
    num_layers=4,
    num_classes=NUM_CLASSES,
    tcn_layers=3,
)
# Pass dummy_lengths to WatchSleepNet
print_model_info("WatchSleepNet", watchsleepnet, dummy_input, dummy_lengths)

# Load and test InsightSleepNet
insightsleepnet = InsightSleepNet(input_size=FEATURE_SIZE, output_size=NUM_CLASSES)
# InsightSleepNet does not require lengths, so pass only dummy_input
print_model_info("InsightSleepNet", insightsleepnet, dummy_input)

# Load and test SleepConvNet
sleepconvnet = SleepConvNet(
    input_size=FEATURE_SIZE, num_segments=SEQ_LENGTH, num_classes=NUM_CLASSES
)
# SleepConvNet does not require lengths, so pass only dummy_input
print_model_info("SleepConvNet", sleepconvnet, dummy_input)
