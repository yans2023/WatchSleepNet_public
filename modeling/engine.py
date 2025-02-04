import torch
import numpy as np
import traceback
from tqdm.auto import tqdm
from sklearn.metrics import (
    f1_score, 
    accuracy_score, 
    cohen_kappa_score, 
    roc_auc_score, 
    confusion_matrix
    )
from data_setup import create_dataloaders_kfolds
from models.sleepconvnet import SleepConvNet
from models.watchsleepnet import WatchSleepNet
from models.insightsleepnet import InsightSleepNet
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

class EarlyStoppingAndCheckpoint:
    def __init__(
        self,
        patience=10,
        verbose=True,
        delta=0,
        checkpoint_path="model_checkpoint.pt",
        trace_func=print,
        monitor=("val_loss",),  # Monitor one or more metrics
        monitor_modes=(
            "min",
        ),  # 'min' or 'max' for each monitored metric
    ):
        """
        ...
        (Docstring omitted for brevity; no changes made)
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.checkpoint_path = checkpoint_path
        self.trace_func = trace_func
        self.monitor = monitor if isinstance(monitor, tuple) else (monitor,)
        self.monitor_modes = (
            monitor_modes if isinstance(monitor_modes, tuple) else (monitor_modes,)
        )
        self.best_scores = {metric: None for metric in self.monitor}
        self.best_metrics = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, metrics, model):
        scores = {metric: metrics[metric] for metric in self.monitor}

        if None in self.best_scores.values():
            self.best_scores = scores
            self.best_metrics = metrics
            self.save_checkpoint(metrics, model)
        else:
            improvement_conditions = []
            for metric, mode in zip(self.monitor, self.monitor_modes):
                if mode == "min":
                    improvement_conditions.append(
                        scores[metric] <= self.best_scores[metric] - self.delta
                    )
                elif mode == "max":
                    improvement_conditions.append(
                        scores[metric] >= self.best_scores[metric] + self.delta
                    )

            if all(improvement_conditions):
                self.save_checkpoint(metrics, model)
                self.best_scores = {
                    metric: (
                        min(scores[metric], self.best_scores[metric])
                        if self.monitor_modes[i] == "min"
                        else max(scores[metric], self.best_scores[metric])
                    )
                    for i, metric in enumerate(self.monitor)
                }
                self.best_metrics = metrics
                self.counter = 0
            else:
                self.counter += 1
                self.trace_func(
                    f"Early stopping counter: {self.counter}/{self.patience}"
                )
                if self.counter >= self.patience:
                    self.early_stop = True

    def save_checkpoint(self, metrics, model):
        """Saves model when validation metric(s) improve."""
        if self.verbose:
            improvement_msg = " and ".join(
                [
                    f"{metric} improved from {self.best_scores[metric]:.4f} to {metrics[metric]:.4f}"
                    for metric in self.monitor
                ]
            )
            self.trace_func(f"Saving checkpoint: {improvement_msg}")
        torch.save(model.state_dict(), self.checkpoint_path)


def pretty_print_confusion_matrix(cm, labels, title=None):
    if title:
        print(title)
        print()

    row_sums = cm.sum(axis=1)
    row_sums[row_sums == 0] = 1

    max_len = max(len(str(label)) for label in labels)
    cell_width = max_len + 2

    header = " " * (cell_width + 1) + " | ".join(
        f"{label:^{cell_width}}" for label in labels
    )
    separator = "-" * len(header)

    rows = []
    for i, row in enumerate(cm):
        row_str = f"{labels[i]:<{cell_width}} |"
        for j, num in enumerate(row):
            percentage = 100 * num / row_sums[i]
            row_str += f" {num:>{cell_width}} ({percentage:.1f}%) |"
        rows.append(row_str)

    print(header)
    print(separator)
    print("\n".join(rows))


def compute_metrics(
    predictions,
    labels,
    pred_probs=None,
    testing=False,
    task="sleep_staging",
    print_conf_matrix=False,
    category_name=None
):
    nan_predictions_count = np.isnan(predictions).sum()

    if nan_predictions_count > 0:
        print(f"NaN detected in predictions! Count: {nan_predictions_count}")
        predictions = torch.where(torch.isnan(predictions), torch.zeros_like(predictions), predictions)
    if pred_probs is not None:
        nan_pred_probs_count = np.isnan(pred_probs).sum()
        if nan_pred_probs_count > 0:
            print(f"NaN detected in labels! Count: {nan_pred_probs_count}")
            pred_probs = torch.where(torch.isnan(pred_probs), torch.zeros_like(pred_probs), pred_probs)

    # Filter out invalid indices where label is -1
    valid_indices = [i for i, label in enumerate(labels) if label != -1]
    valid_labels = [labels[i] for i in valid_indices]
    valid_predictions = [predictions[i] for i in valid_indices]

    accuracy = accuracy_score(valid_labels, valid_predictions)
    f1 = f1_score(valid_labels, valid_predictions, average="weighted")
    kappa = cohen_kappa_score(valid_labels, valid_predictions)

    if print_conf_matrix:
        cm = confusion_matrix(valid_labels, valid_predictions)
        if task == "sleep_staging":
            class_names = ["Wake", "NREM", "REM"]
        else:
            class_names = ["Wake", "Sleep"]

        title = "Confusion Matrix"
        if category_name:
            title += f" ({category_name})"

        pretty_print_confusion_matrix(cm, class_names, title=title)

    if testing:
        valid_probabilities = [pred_probs[i] for i in valid_indices]
        rem_f1 = f1_score(valid_labels, valid_predictions, labels=[2], average="macro")
        if task == "sleep_wake":
            auroc = roc_auc_score(valid_labels, valid_probabilities, average="macro")
        elif task == "sleep_staging":
            try:
                auroc = roc_auc_score(
                    valid_labels,
                    valid_probabilities,
                    multi_class="ovr",
                    average="macro",
                )
            except ValueError as e:
                print(f"Error: {e}")
                num_classes_y_true = len(set(valid_labels))
                num_columns_y_score = np.array(valid_probabilities).shape[1]
                print(f"Number of classes in y_true: {num_classes_y_true}")
                print(f"Number of columns in y_score: {num_columns_y_score}")
                raise
        return accuracy, f1, kappa, rem_f1, auroc

    return accuracy, f1, kappa


def check_for_nan_grads(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"NaN found in gradients for layer: {name}")
                return True
    return False


def replace_nan_grads(model):
    for param in model.parameters():
        if param.grad is not None:
            param.grad = torch.where(
                torch.isnan(param.grad), torch.zeros_like(param.grad), param.grad
            )


#
# NEW HELPER FUNCTION BELOW
#
def forward_pass_with_mask(model, X, y, lengths, device, task, model_name=None):
    """
    Performs the forward pass through the model, checks for NaNs in outputs,
    creates a mask for valid time steps, and returns the masked outputs/labels,
    predictions, and probabilities.

    Args:
        model (nn.Module): The model to run forward pass on.
        X (torch.Tensor): Input tensor of shape (batch_size, seq_len, features, ...).
        y (torch.Tensor): Label tensor of shape (batch_size, seq_len).
        lengths (torch.Tensor): Actual sequence lengths for each sample in the batch.
        device (torch.device): Device to move tensors to.
        task (str): Task name ("sleep_staging" or "sleep_wake").
        model_name (str): Optional model name for special handling (e.g., "insightsleepnet").

    Returns:
        masked_outputs (torch.Tensor): Flattened valid outputs of shape (sum_of_seq_lengths, num_classes).
        masked_labels (torch.Tensor): Flattened valid labels of shape (sum_of_seq_lengths,).
        predicted (torch.Tensor): Flattened predictions of shape (sum_of_seq_lengths,).
        pred_probs (torch.Tensor): Flattened predicted probabilities of shape (sum_of_seq_lengths, num_classes) 
                                   or (sum_of_seq_lengths,) for "sleep_wake".
        mask (torch.Tensor): A boolean mask indicating valid time steps.
        outputs (torch.Tensor): The original unmasked outputs for the entire batch (batch_size, seq_len, num_classes).
    """
    X, y, lengths = X.to(device), y.to(device), lengths.to(device)

    outputs = model(X, lengths)

    # Check for NaNs in outputs
    if torch.isnan(outputs).any():
        print("NaN detected in model outputs, replacing with zeros.")
        outputs = torch.where(
            torch.isnan(outputs), torch.zeros_like(outputs), outputs
        )

    batch_size, max_len, num_classes = outputs.shape
    # Create a mask based on the lengths of the sequences, but limited by y's shape
    mask = torch.arange(y.shape[1], device=outputs.device).expand(
        batch_size, y.shape[1]
    ) < lengths.unsqueeze(1)

    # Flatten outputs and labels, but only keep valid time steps based on the mask
    masked_outputs = outputs[:, : y.shape[1], :][mask]
    masked_labels = y[mask]

    # Get predictions and probabilities
    _, predicted_2d = torch.max(outputs, dim=2)
    pred_probs_2d = torch.softmax(outputs, dim=2)
    pred_probs_2d = torch.where(
        torch.isnan(pred_probs_2d), torch.zeros_like(pred_probs_2d), pred_probs_2d
    )

    # Flatten for valid time steps
    predicted = predicted_2d[:, : y.shape[1]][mask].view(-1)
    if task == "sleep_staging":
        pred_probs = pred_probs_2d[:, : y.shape[1], :][mask]
    elif task == "sleep_wake":
        # Only the probability for the "Sleep" class (index=1)
        pred_probs = pred_probs_2d[:, : y.shape[1], 1][mask]

    return masked_outputs, masked_labels, predicted, pred_probs, mask, outputs


def train_step(model, dataloader, loss_fn, optimizer, device, model_name="watchsleepnet", task="sleep_staging"):
    model.train()
    train_loss = 0.0
    all_labels = []
    all_predictions = []
    all_probabilities = []
    total_batches = 0

    for batch_idx, (X, y, lengths, AHIs) in enumerate(
        tqdm(dataloader, desc="Training Batch", leave=False)
    ):
        try:
            optimizer.zero_grad()

            masked_outputs, masked_labels, predicted, pred_probs, mask, outputs = forward_pass_with_mask(
                model, X, y, lengths, device, task, model_name
            )
            # Compute the loss using the masked outputs and labels
            loss = loss_fn(masked_outputs, masked_labels)

            # Check for NaNs in loss
            if torch.isnan(loss):
                print(f"NaN detected in loss on batch {batch_idx}. Ending epoch.")
                train_loss = float("nan")
                acc = f1 = kappa = rem_f1 = auroc = float("nan")
                break  # End the epoch immediately

            loss.backward()

            # Apply gradient clipping only for 'insightsleepnet'
            if model_name == "insightsleepnet":
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)

                # Check for NaNs in gradients
                if check_for_nan_grads(model):
                    print(f"NaN detected in gradients on batch {batch_idx}. Ending epoch.")
                    train_loss = float("nan")
                    acc = f1 = kappa = rem_f1 = auroc = float("nan")
                    break  # End the epoch immediately

            optimizer.step()

            train_loss += loss.item()
            total_batches += 1

            # Collect the labels and predictions for metric computation
            all_labels.extend(masked_labels.view(-1).cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            if task == "sleep_staging":
                all_probabilities.extend(pred_probs.detach().cpu().numpy())
            else:  # "sleep_wake"
                all_probabilities.extend(pred_probs.detach().cpu().numpy())

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Out of memory error encountered on batch {batch_idx}.")
                print(f"Batch size: {X.size()}")
                print(f"Input shape: {X.shape}")
                print(
                    f"Current memory allocated: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB"
                )
                print(
                    f"Max memory allocated: {torch.cuda.max_memory_allocated(device) / 1024 ** 3:.2f} GB"
                )
                print(
                    f"Current memory cached: {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB"
                )
                print(
                    f"Max memory cached: {torch.cuda.max_memory_reserved(device) / 1024 ** 3:.2f} GB"
                )
                traceback.print_exc()
                torch.cuda.empty_cache()
            else:
                raise e

    if total_batches > 0 and not np.isnan(train_loss):
        avg_loss = train_loss / total_batches
        acc, f1, kappa, rem_f1, auroc = compute_metrics(
            all_predictions,
            all_labels,
            pred_probs=all_probabilities,
            testing=True,
            task=task,
        )
    else:
        avg_loss = train_loss  # This might be NaN
        acc = f1 = kappa = rem_f1 = auroc = float("nan")

    return avg_loss, acc, f1, kappa, rem_f1, auroc


def validate_step(model, dataloader, loss_fn, device, task="sleep_staging"):
    model.eval()
    val_loss = 0.0
    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for X, y, lengths, AHIs in tqdm(dataloader, desc="Validation Batch", leave=False):
            masked_outputs, masked_labels, predicted, pred_probs, mask, outputs = forward_pass_with_mask(
                model, X, y, lengths, device, task
            )
            loss = loss_fn(masked_outputs, masked_labels)
            val_loss += loss.item()

            # Collect for metric computation
            all_labels.extend(masked_labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            if task == "sleep_staging":
                all_probabilities.extend(pred_probs.detach().cpu().numpy())
            else:  # "sleep_wake"
                all_probabilities.extend(pred_probs.detach().cpu().numpy())

    avg_loss = val_loss / len(dataloader)
    acc, f1, kappa, rem_f1, auroc = compute_metrics(
        all_predictions,
        all_labels,
        pred_probs=all_probabilities,
        testing=True,
        task=task,
    )
    return avg_loss, acc, f1, kappa, rem_f1, auroc


def test_step(model, dataloader, device, task="sleep_staging"):
    model.eval()
    all_true_labels = []
    all_predicted_labels = []
    all_predicted_probabilities = []
    all_ahi_values = []

    with torch.no_grad():
        for data, labels, lengths, ahis in dataloader:
            # --- Refactored forward pass (no grad) ---
            masked_outputs, masked_labels, predicted, pred_probs, mask, outputs = forward_pass_with_mask(
                model, data, labels, lengths, device, task
            )
            # Store valid (flattened) results
            all_true_labels.extend(masked_labels.cpu().numpy())
            all_predicted_labels.extend(predicted.cpu().numpy())
            all_predicted_probabilities.extend(pred_probs.detach().cpu().numpy())

            # Since AHI values are per sample, we repeat each 'ahi' for the valid sequence length
            for ahi, length in zip(ahis, lengths):
                all_ahi_values.extend([ahi] * length)

    return (
        all_true_labels,
        all_predicted_labels,
        all_predicted_probabilities,
        all_ahi_values,
    )


def compute_metrics_per_ahi_category(
    true_labels, predicted_labels, predicted_probabilities, ahi_values
):
    ahi_categories = {
        "Normal": (0, 5),
        "Mild": (5, 15),
        "Moderate": (15, 30),
        "Severe": (30, np.inf),
    }

    metrics_per_category = {}

    for category, (lower_bound, upper_bound) in ahi_categories.items():
        indices = [
            i for i, ahi in enumerate(ahi_values) if lower_bound <= ahi < upper_bound
        ]

        if not indices:
            print(f"No data for AHI category: {category}")
            continue

        category_true_labels = [true_labels[i] for i in indices]
        category_predicted_labels = [predicted_labels[i] for i in indices]
        category_predicted_probabilities = [predicted_probabilities[i] for i in indices]

        acc, f1, kappa, rem_f1, auroc = compute_metrics(
            category_predicted_labels,
            category_true_labels,
            pred_probs=category_predicted_probabilities,
            testing=True,
            task="sleep_staging",
            print_conf_matrix=True,
            category_name=category,
        )

        metrics_per_category[category] = {
            "Accuracy": acc,
            "F1 Score": f1,
            "Kappa": kappa,
            "REM F1 Score": rem_f1,
            "AUROC": auroc,
        }

    return metrics_per_category

def train(
    model,
    model_name,
    train_dataloader,
    val_dataloader,
    optimizer,
    loss_fn,
    num_epochs,
    patience,
    device,
    model_save_path,
):
    results = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "train_kappa": [],
        "train_rem_f1": [],
        "train_auroc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_kappa": [],
        "val_rem_f1": [],
        "val_auroc": [],
    }

    early_stopping = EarlyStoppingAndCheckpoint(
        patience=patience,
        checkpoint_path=model_save_path,
        monitor="val_loss",
        monitor_modes="min",
    )
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        train_loss, train_acc, train_f1, train_kappa, train_rem_f1, train_auroc = train_step(
            model, train_dataloader, loss_fn, optimizer, device, model_name=model_name
        )
        val_loss, val_acc, val_f1, val_kappa, val_rem_f1, val_auroc = validate_step(
            model, val_dataloader, loss_fn, device
        )

        # Store metrics for each epoch
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["train_f1"].append(train_f1)
        results["train_kappa"].append(train_kappa)
        results["train_rem_f1"].append(train_rem_f1)
        results["train_auroc"].append(train_auroc)

        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
        results["val_f1"].append(val_f1)
        results["val_kappa"].append(val_kappa)
        results["val_rem_f1"].append(val_rem_f1)
        results["val_auroc"].append(val_auroc)

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(
            f"Training: Loss={train_loss:.3f}, Accuracy={train_acc:.3f}, "
            f"F1 Score={train_f1:.3f}, Kappa={train_kappa:.3f}, "
            f"REM F1={train_rem_f1:.3f}, AUROC={train_auroc:.3f}"
        )
        print(
            f"Validation: Loss={val_loss:.4f}, Accuracy={val_acc:.3f}, "
            f"F1 Score={val_f1:.3f}, Kappa={val_kappa:.3f}, "
            f"REM F1={val_rem_f1:.3f}, AUROC={val_auroc:.3f}"
        )
        print("-" * 80)
        # Log the metrics
        metrics = {"val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1}

        early_stopping(metrics, model)
        if early_stopping.early_stop:
            print("Early stopping triggered. Ending training.")
            break

    return results


def setup_model_and_optimizer(
    model_name,
    model_params,
    device,
    saved_model_path=None,
    learning_rate=1e-3,
    weight_decay=1e-5,
    freeze_layers=False,
):
    """
    A revamped helper function for instantiating a model, optionally loading a checkpoint,
    optionally freezing layers, and creating an optimizer, supporting custom hyperparameters 
    for SleepConvNet, InsightSleepNet, and WatchSleepNet (including ablation flags).
    
    Args:
        model_name (str): One of ["sleepconvnet", "insightsleepnet", "watchsleepnet", "watchsleepnet2"].
        model_params (dict): A dictionary (or config object) that holds 
            all relevant hyperparameters for the chosen model. For example:
            
            # For SleepConvNet:
            {
                "input_size": 750,
                "target_size": 256,
                "num_segments": 1100,
                "num_classes": 3,
                "dropout_rate": 0.2,
                "conv_layers_configs": [(1,32,3,1), (32,64,3,1), ...],
                "dilation_layers_configs": [...],
                "use_residual": True
            }
            
            # For InsightSleepNet:
            {
                "input_size": 750,
                "output_size": 3,
                "n_filters": 32,
                "bottleneck_channels": 32,
                "kernel_sizes": [9, 19, 39],
                "num_inception_blocks": 1,
                "use_residual": True,
                "dropout_rate": 0.2,
                "activation": nn.ReLU()
            }
            
            # For WatchSleepNet/WatchSleepNet2:
            {
                "num_features": 1,
                "num_channels": 32,
                "kernel_size": 3,
                "hidden_dim": 64,
                "num_heads": 4,
                "num_layers": 2,
                "tcn_layers": 3,
                "num_classes": 3,
                "use_tcn": True,
                "use_attention": True
            }
        
        device (torch.device): 'cpu' or 'cuda'.
        saved_model_path (str, optional): Path to a pretrained model checkpoint to load.
        learning_rate (float, optional): Optimizer learning rate. Default=1e-3.
        weight_decay (float, optional): L2 regularization. Default=1e-5.
        freeze_layers (bool, optional): If True, freeze all but certain layers. Default=False.

    Returns:
        (model, optimizer): 
            model (nn.Module): The instantiated model with loaded weights (if provided),
                               properly configured with your hyperparams.
            optimizer (torch.optim.Optimizer): An Adam optimizer for this model's trainable parameters.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from models.sleepconvnet import SleepConvNet
    from models.insightsleepnet import InsightSleepNet
    from models.watchsleepnet import WatchSleepNet

    # 1) Instantiate the model based on model_name
    if model_name == "sleepconvnet":
        model = SleepConvNet(
            input_size=model_params.get("input_size", 750),
            target_size=model_params.get("target_size", 256),
            num_segments=model_params.get("num_segments", 1100),
            num_classes=model_params.get("num_classes", 3),
            dropout_rate=model_params.get("dropout_rate", 0.2),
            conv_layers_configs=model_params.get("conv_layers_configs", None),
            dilation_layers_configs=model_params.get("dilation_layers_configs", None),
            use_residual=model_params.get("use_residual", True),
        ).to(device)

    elif model_name == "insightsleepnet":
        from models.insightsleepnet import InsightSleepNet

        # 1) Check if the user is passing a "block_configs" list
        block_configs = model_params.get("block_configs", None)
        
        if block_configs is not None:
            model = InsightSleepNet(
                input_size = model_params.get("input_size", 750),
                output_size = model_params.get("output_size", 3),
                block_configs = block_configs,
                dropout_rate = model_params.get("dropout_rate", 0.2),
                final_pool_size = model_params.get("final_pool_size", 1100),  
                activation = model_params.get("activation", nn.ReLU())
            ).to(device)
        
        else:
            print("[setup_model_and_optimizer] 'block_configs' not found in model_params. InsightSleepNet cannot be initialized.")
            return None, None

    elif model_name == "watchsleepnet":
        model = WatchSleepNet(
            num_features=model_params.get("num_features", 1),
            num_channels=model_params.get("num_channels", 32),
            kernel_size=model_params.get("kernel_size", 3),
            hidden_dim=model_params.get("hidden_dim", 64),
            num_heads=model_params.get("num_heads", 4),
            num_layers=model_params.get("num_layers", 2),
            tcn_layers=model_params.get("tcn_layers", 3),
            use_tcn=model_params.get("use_tcn", True),
            use_attention=model_params.get("use_attention", True),
            num_classes=model_params.get("num_classes", 3),
        ).to(device)

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    # 2) (Optional) Load saved model weights if provided
    if saved_model_path is not None:
        print(f"[setup_model_and_optimizer] Loading model weights from {saved_model_path}")
        model.load_state_dict(torch.load(saved_model_path, map_location=device))

    # 3) Optionally freeze layers (partial or full)
    if freeze_layers:
        for name, param in model.named_parameters():
            # Example partial freeze logic:
            # if "classifier" in name or "attention" in name or "tcn" in name:
            #     param.requires_grad = True
            # else:
            #     param.requires_grad = False
            param.requires_grad = False  # or your custom logic
        print("[setup_model_and_optimizer] Freezing layers as per 'freeze_layers=True'")

    else:
        # Ensure all parameters are trainable
        for param in model.parameters():
            param.requires_grad = True

    # 4) Create the optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    return model, optimizer


def run_training_epochs(
    model,
    train_dataloader,
    val_dataloader,
    loss_fn,
    optimizer,
    device,
    num_epochs,
    patience,
    checkpoint_path,
    monitor_metrics=("val_loss",),
    monitor_modes=("min",),
    model_name="watchsleepnet",
    # Add any additional logging parameters you need
):
    """
    Helper function that runs the actual training loop with early stopping.
    Returns the best model path (if checkpointing) and training logs.
    """

    # Prepare early stopping
    early_stopping = EarlyStoppingAndCheckpoint(
        patience=patience,
        checkpoint_path=checkpoint_path,
        monitor=monitor_metrics,
        monitor_modes=monitor_modes,
    )

    # For storing intermediate results if needed
    training_logs = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "train_kappa": [],
        "train_rem_f1": [],
        "train_auroc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_kappa": [],
        "val_rem_f1": [],
        "val_auroc": [],
    }

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        # 1. Train step
        (
            train_loss,
            train_acc,
            train_f1,
            train_kappa,
            train_rem_f1,
            train_auroc,
        ) = train_step(model, train_dataloader, loss_fn, optimizer, device, model_name=model_name)

        # 2. Validation step
        (
            val_loss,
            val_acc,
            val_f1,
            val_kappa,
            val_rem_f1,
            val_auroc,
        ) = validate_step(model, val_dataloader, loss_fn, device)

        # Store logs
        training_logs["train_loss"].append(train_loss)
        training_logs["train_acc"].append(train_acc)
        training_logs["train_f1"].append(train_f1)
        training_logs["train_kappa"].append(train_kappa)
        training_logs["train_rem_f1"].append(train_rem_f1)
        training_logs["train_auroc"].append(train_auroc)

        training_logs["val_loss"].append(val_loss)
        training_logs["val_acc"].append(val_acc)
        training_logs["val_f1"].append(val_f1)
        training_logs["val_kappa"].append(val_kappa)
        training_logs["val_rem_f1"].append(val_rem_f1)
        training_logs["val_auroc"].append(val_auroc)

        print(
            f"\nEpoch {epoch+1}/{num_epochs}:\n"
            f"Training: "
            f"Loss={train_loss:.3f}, Acc={train_acc:.3f}, F1={train_f1:.3f}, "
            f"Kappa={train_kappa:.3f}, REM_F1={train_rem_f1:.3f}, AUROC={train_auroc:.3f}\n"
            f"Validation: "
            f"Loss={val_loss:.3f}, Acc={val_acc:.3f}, F1={val_f1:.3f}, "
            f"Kappa={val_kappa:.3f}, REM_F1={val_rem_f1:.3f}, AUROC={val_auroc:.3f}"
        )
        print("-" * 80)

        # Early stopping
        # Prepare dict of monitored metrics
        # e.g. if monitor_metrics=("val_loss", "val_kappa"), we create:
        #   metrics = {"val_loss": val_loss, "val_kappa": val_kappa}
        metrics_dict = {}
        for metric in monitor_metrics:
            if metric == "val_loss":
                metrics_dict[metric] = val_loss
            elif metric == "val_kappa":
                metrics_dict[metric] = val_kappa
            elif metric == "val_acc":
                metrics_dict[metric] = val_acc
            # etc. for other possible metrics

        early_stopping(metrics_dict, model)
        if early_stopping.early_stop:
            print("Early stopping triggered. Ending training.")
            break

    # The best checkpoint is early_stopping.checkpoint_path
    return early_stopping.checkpoint_path, training_logs


def finalize_fold_or_repeat(
    model,
    checkpoint_path,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    loss_fn,
    device,
    best_results_dict,
    do_test=True,
    task="sleep_staging",
    model_name="watchsleepnet",
):
    """
    Helper function that:
        loads the best checkpoint,
        re-evaluates on train/val set to get best metrics,
        runs on the test set (optionally),
        updates best_results_dict with the new metrics,
        returns predictions for further aggregation.
    """

    # 1. Load best checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        print("No checkpoint found. Using current model weights (possibly not best).")

    # 2. Evaluate on train/val to store best metrics
    (train_loss, train_acc, train_f1, train_kappa, train_rem_f1, train_auroc,) = validate_step(
        model, train_dataloader, loss_fn, device
    )
    (val_loss, val_acc, val_f1, val_kappa, val_rem_f1, val_auroc,) = validate_step(
        model, val_dataloader, loss_fn, device
    )

    best_results_dict["train_loss"].append(train_loss)
    best_results_dict["train_acc"].append(train_acc)
    best_results_dict["train_f1"].append(train_f1)
    best_results_dict["train_kappa"].append(train_kappa)
    best_results_dict["train_rem_f1"].append(train_rem_f1)
    best_results_dict["train_auroc"].append(train_auroc)

    best_results_dict["val_loss"].append(val_loss)
    best_results_dict["val_acc"].append(val_acc)
    best_results_dict["val_f1"].append(val_f1)
    best_results_dict["val_kappa"].append(val_kappa)
    best_results_dict["val_rem_f1"].append(val_rem_f1)
    best_results_dict["val_auroc"].append(val_auroc)

    test_true_labels, test_predicted_labels, test_predicted_probs, test_ahi = [], [], [], []

    # 3. Optional test
    if do_test and test_dataloader is not None:
        test_loss, test_acc, test_f1, test_kappa, test_rem_f1, test_auroc = validate_step(
            model, test_dataloader, loss_fn, device
        )
        best_results_dict["test_loss"].append(test_loss)
        best_results_dict["test_acc"].append(test_acc)
        best_results_dict["test_f1"].append(test_f1)
        best_results_dict["test_kappa"].append(test_kappa)
        best_results_dict["test_rem_f1"].append(test_rem_f1)
        best_results_dict["test_auroc"].append(test_auroc)

        (
            test_true_labels,
            test_predicted_labels,
            test_predicted_probs,
            test_ahi,
        ) = test_step(model, test_dataloader, device, task=task)

    return test_true_labels, test_predicted_labels, test_predicted_probs, test_ahi

def aggregate_and_print_results(
    results_dict,
    true_labels,
    predicted_labels,
    predicted_probs,
    ahi_values,
    final_task="sleep_staging",
):
    """
    Helper function that:
        Prints mean ± std for each metric in results_dict (e.g., train_loss, val_loss, etc.)
        Computes and prints overall performance metrics (Accuracy, F1, Kappa, REM F1, AUROC)
        Optionally prints confusion matrix + AHI category metrics
        Returns the tuple: (overall_acc, overall_f1, overall_kappa, rem_f1, auroc)
    """
    import numpy as np
    
    # 1) Print aggregated fold/repeat metrics
    for key, values in results_dict.items():
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{key}: {mean_val:.4f} ± {std_val:.4f}")
        else:
            print(f"{key}: No data collected.")

    # 2) If we have predictions for test sets (or final sets) across all folds
    if predicted_labels and true_labels:
        print("\nOverall Metrics:")
        overall_acc, overall_f1, overall_kappa, rem_f1, auroc = compute_metrics(
            predicted_labels,
            true_labels,
            pred_probs=predicted_probs,
            testing=True,
            task=final_task,
            print_conf_matrix=True,
        )
        print(f"Overall Accuracy: {overall_acc:.4f}")
        print(f"Overall F1 Score: {overall_f1:.4f}")
        print(f"Overall Kappa: {overall_kappa:.4f}")
        print(f"REM F1 Score: {rem_f1:.4f}")
        print(f"AUROC: {auroc:.4f}")

        # 3) Compute metrics per AHI category
        print("\nMetrics per AHI Category:")
        metrics_per_category = compute_metrics_per_ahi_category(
            true_labels,
            predicted_labels,
            predicted_probs,
            ahi_values,
        )
        for category, sub_metrics in metrics_per_category.items():
            print(f"\nMetrics for AHI Category '{category}':")
            for metric_name, val in sub_metrics.items():
                print(f"  {metric_name}: {val:.4f}")

        return overall_acc, overall_f1, overall_kappa, rem_f1, auroc
    else:
        print("No final test predictions available to compute overall metrics.")
        return None, None, None, None, None

def train(
    model_name,
    model_params,          # Dictionary containing model hyperparams (including num_classes, use_tcn, use_attention if needed)
    train_dataloader,
    val_dataloader,
    loss_fn,
    num_epochs,
    patience,
    device,
    model_save_path,
    saved_model_path=None, # If you want to load an existing checkpoint initially
    learning_rate=1e-3,
    weight_decay=1e-5,
    freeze_layers=False,
):
    """
    Train a model (on a single dataset) for a specified number of epochs with early stopping,
    using the run_training_epochs helper.

    Args:
        model_name (str): 
            E.g., "watchsleepnet", "insightsleepnet", or "sleepconvnet".
        model_params (dict): 
            A dictionary containing hyperparameters for the model’s constructor.
            For example:
                {
                  "num_features": 1,
                  "num_channels": 256,
                  "kernel_size": 5,
                  "hidden_dim": 256,
                  "num_heads": 16,
                  "tcn_layers": 3,
                  "num_layers": 4,
                  "use_tcn": True,
                  "use_attention": True,
                  "num_classes": 3,
                  ... (etc. for other models)
                }
            This function also can read from these fields if you prefer
            to unify them instead of passing them as separate arguments below.
        train_dataloader (DataLoader): 
            The DataLoader for the training dataset.
        val_dataloader (DataLoader): 
            The DataLoader for the validation dataset.
        loss_fn (callable): 
            The loss function (e.g., CrossEntropyLoss).
        num_epochs (int): 
            Maximum number of training epochs.
        patience (int): 
            Patience for early stopping.
        device (torch.device): 
            'cpu' or 'cuda'.
        model_save_path (str): 
            File path to save the best model checkpoint.
        saved_model_path (str, optional): 
            Path to an existing checkpoint to load before training. Defaults to None.
        learning_rate (float, optional): 
            Learning rate for the optimizer. Defaults to 1e-3.
        weight_decay (float, optional): 
            Weight decay (L2 regularization). Defaults to 1e-5.
        freeze_layers (bool, optional): 
            Whether to freeze some or all layers in the model. Defaults to False.

    Returns:
        dict:
            A dictionary containing epoch-wise logs of metrics. Keys might include:
              "train_loss", "train_acc", "train_f1", "train_kappa", "train_rem_f1", "train_auroc",
              "val_loss",   "val_acc",   "val_f1",   "val_kappa",   "val_rem_f1",   "val_auroc"
            Each key maps to a list of floats, one per epoch.
    """

    # Set up the model and optimizer in one shot.
    model, optimizer = setup_model_and_optimizer(
        model_name=model_name,
        model_params=model_params,  
        device=device,
        saved_model_path=saved_model_path,
        learning_rate=learning_rate, 
        weight_decay=weight_decay,  
        freeze_layers=freeze_layers,
    )

    # Call run_training_epochs to train & validate with early stopping.
    best_ckpt_path, training_logs = run_training_epochs(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        patience=patience,
        checkpoint_path=model_save_path,   
        monitor_metrics=("val_kappa", "val_loss"), 
        monitor_modes=("max", "min"),               
        model_name=model_name,       
    )

    # Load the best checkpoint, if it exists.
    if best_ckpt_path and os.path.exists(best_ckpt_path):
        model.load_state_dict(torch.load(best_ckpt_path))

    # Return the training logs (epoch-wise metrics).
    return training_logs

def train_and_evaluate(
    model_name,
    model_params,     
    dataloader_folds,
    saved_model_path,
    loss_fn,
    num_epochs,
    patience,
    device,
    checkpoint_path=None,
    learning_rate=1e-3,
    weight_decay=1e-5,
    freeze_layers=False,
):
    """
    Pretrain and finetune on a smaller dataset

    Args:
        model_name (str):
            E.g. "sleepconvnet", "watchsleepnet", or "insightsleepnet".
        model_params (dict):
            Dictionary with the hyperparameters required by the chosen model's constructor.
            Example fields for WatchSleepNet might include:
                {
                    "num_features": 1,
                    "num_channels": 256,
                    "kernel_size": 5,
                    "hidden_dim": 256,
                    "num_heads": 16,
                    "tcn_layers": 3,
                    "num_layers": 4,
                    "use_tcn": True,
                    "use_attention": True,
                    "num_classes": 3,
                    ...
                }
        dataloader_folds (list of tuples):
            Each tuple is (train_dataloader, val_dataloader, test_dataloader) for one fold.
        saved_model_path (str or None):
            Optional path to load a checkpoint before training each fold (e.g. for transfer learning).
        loss_fn (callable):
            Loss function (e.g., CrossEntropyLoss).
        num_epochs (int):
            Maximum number of epochs to train for each fold.
        patience (int):
            Early stopping patience for each fold.
        device (torch.device):
            'cpu' or 'cuda' for model training/evaluation.
        checkpoint_path (str or None):
            If provided, base path to save the best checkpoint per fold.
            The actual file will be checkpoint_path.replace(".pt", f"_fold{fold_idx}.pt").
        learning_rate (float, optional):
            Learning rate for the optimizer (defaults to 1e-3).
        weight_decay (float, optional):
            Weight decay (L2 regularization) for the optimizer (defaults to 1e-5).
        freeze_layers (bool, optional):
            Whether to freeze some or all layers in the model.

    Returns:
        (best_results, overall_acc, overall_f1, overall_kappa, rem_f1, auroc):
            best_results (dict): Aggregated metrics (train/val/test) across folds.
            overall_acc, overall_f1, overall_kappa, rem_f1, auroc (float):
                Final metrics computed across all folds' test sets.
    """
    import numpy as np
    

    # Prepare a dictionary to store aggregated metrics for all folds
    best_results = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "train_kappa": [],
        "train_rem_f1": [],
        "train_auroc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_kappa": [],
        "val_rem_f1": [],
        "val_auroc": [],
        "test_loss": [],
        "test_acc": [],
        "test_f1": [],
        "test_kappa": [],
        "test_rem_f1": [],
        "test_auroc": [],
    }

    # Lists to store combined test predictions and labels across folds
    all_true_labels = []
    all_predicted_labels = []
    all_predicted_probabilities = []
    all_ahi_values = []

    num_folds = len(dataloader_folds)
    print(f"Number of folds: {num_folds}")

    # ---- 1) Loop over each fold ----
    for fold_idx, (train_dl, val_dl, test_dl) in enumerate(dataloader_folds, start=1):
        print(f"\n===== Fold {fold_idx}/{num_folds} =====")

        # Create a fold-specific checkpoint path if we have a base path
        fold_checkpoint = None
        if checkpoint_path is not None:
            fold_checkpoint = checkpoint_path.replace(".pt", f"_fold{fold_idx}.pt")

        # ---- 2) Setup model & optimizer for this fold ----
        # Possibly loading a saved checkpoint if saved_model_path is provided
        model, optimizer = setup_model_and_optimizer(
            model_name=model_name,
            model_params=model_params,     # dictionary with fields for the chosen model
            device=device,
            saved_model_path=saved_model_path,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            freeze_layers=freeze_layers,
        )

        # ---- 3) Run training epochs (train + val) with early stopping
        best_ckpt_path, training_logs = run_training_epochs(
            model=model,
            train_dataloader=train_dl,
            val_dataloader=val_dl,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            num_epochs=num_epochs,
            patience=patience,
            checkpoint_path=fold_checkpoint,
            monitor_metrics=("val_kappa", "val_loss"),  # multi-metric example
            monitor_modes=("max", "min"),
            model_name=model_name,  # for special logic like gradient clipping
        )

        # ---- 4) Finalize fold: reload best checkpoint, test on test_dl, update best_results
        test_true, test_pred, test_probs, test_ahi = finalize_fold_or_repeat(
            model=model,
            checkpoint_path=best_ckpt_path,
            train_dataloader=train_dl,
            val_dataloader=val_dl,
            test_dataloader=test_dl,
            loss_fn=loss_fn,
            device=device,
            best_results_dict=best_results,
            do_test=True,
            task="sleep_staging",  # or "sleep_wake"
            model_name=model_name,
        )

        # Accumulate test predictions for overall metrics
        all_true_labels.extend(test_true)
        all_predicted_labels.extend(test_pred)
        all_predicted_probabilities.extend(test_probs)
        all_ahi_values.extend(test_ahi)

    # ---- 5) Print aggregated fold metrics (mean ± std)
    for key, values in best_results.items():
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{key}: {mean_val:.4f} ± {std_val:.4f}")
        else:
            print(f"{key}: No data available")

    # ---- 6) Compute overall metrics across folds (aggregated test sets)
    print("\nOverall Metrics (Aggregated Across All Folds):")
    overall_acc, overall_f1, overall_kappa, rem_f1, auroc = compute_metrics(
        all_predicted_labels,
        all_true_labels,
        pred_probs=all_predicted_probabilities,
        testing=True,
        task="sleep_staging",
        print_conf_matrix=True,  # prints confusion matrix
    )
    print(f"Overall Accuracy: {overall_acc:.4f}")
    print(f"Overall F1 Score: {overall_f1:.4f}")
    print(f"Overall Kappa: {overall_kappa:.4f}")
    print(f"REM F1 Score: {rem_f1:.4f}")
    print(f"AUROC: {auroc:.4f}")

    # ---- 7) Compute per-AHI category metrics
    print("\nMetrics per AHI Category:")
    metrics_per_category = compute_metrics_per_ahi_category(
        all_true_labels,
        all_predicted_labels,
        all_predicted_probabilities,
        all_ahi_values,
    )
    for category, cat_metrics in metrics_per_category.items():
        print(f"\nMetrics for AHI Category '{category}':")
        for metric_name, value in cat_metrics.items():
            print(f"  {metric_name}: {value:.4f}")

    # Return aggregated results + final overall metrics
    return best_results, overall_acc, overall_f1, overall_kappa, rem_f1, auroc

def train_ablate_evaluate(
    model_name,
    model_params,         # Includes ablation flags, e.g. "use_tcn": True, "use_attention": False, etc.
    dataloader_folds,
    saved_model_path=None,
    learning_rate=1e-3,
    weight_decay=1e-5,
    loss_fn=None,
    num_epochs=50,
    patience=10,
    device=None,
    checkpoint_path=None,
    freeze_layers=False,
):
    """
    Refactored train_ablate_evaluate function that leverages helper functions
    for setting up models, running epochs with early stopping, and finalizing each fold.

    Args:
        model_name (str): 
            One of ["sleepconvnet", "watchsleepnet", "insightsleepnet"].
            (If your watchsleepnet is ablation-enabled, ablation flags are in model_params).
        model_params (dict):
            Dictionary of hyperparameters for the model, e.g.
            {
                "num_features": 1,
                "num_channels": 256,
                "kernel_size": 5,
                "hidden_dim": 256,
                "num_heads": 16,
                "tcn_layers": 3,
                "num_layers": 4,
                "use_tcn": True,
                "use_attention": False,
                "num_classes": 3,
                ...
            }
        dataloader_folds (list of tuples):
            Each element: (train_dataloader, val_dataloader, test_dataloader).
        saved_model_path (str or None):
            If provided, load from checkpoint before training each fold.
        learning_rate (float):
            Learning rate.
        weight_decay (float):
            Weight decay.
        loss_fn (callable):
            e.g., CrossEntropyLoss.
        num_epochs (int):
            Max epochs.
        patience (int):
            Early stopping patience.
        device (torch.device or None):
            'cuda' or 'cpu'. If None, auto-detect.
        checkpoint_path (str or None):
            Base path for saving best checkpoint each fold 
            (fold path = checkpoint_path.replace('.pt', f'_fold{idx}.pt')).
        freeze_layers (bool):
            If True, freeze part or all of the model.

    Returns:
        tuple: (best_results, overall_acc, overall_f1, overall_kappa, rem_f1, auroc)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dictionary to store aggregated results across folds
    best_results = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "train_kappa": [],
        "train_rem_f1": [],
        "train_auroc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_kappa": [],
        "val_rem_f1": [],
        "val_auroc": [],
        "test_loss": [],
        "test_acc": [],
        "test_f1": [],
        "test_kappa": [],
        "test_rem_f1": [],
        "test_auroc": [],
    }

    all_true_labels = []
    all_predicted_labels = []
    all_predicted_probabilities = []
    all_ahi_values = []

    num_folds = len(dataloader_folds)
    print(f"Number of folds: {num_folds}")

    # Loop over folds
    for fold_idx, (train_dl, val_dl, test_dl) in enumerate(dataloader_folds, 1):
        print(f"\n===== Fold {fold_idx}/{num_folds} =====")

        # Checkpoint for this fold
        fold_ckpt = None
        if checkpoint_path:
            fold_ckpt = checkpoint_path.replace(".pt", f"_fold{fold_idx}.pt")

        # 1) Setup model & optimizer for ablation
        model, optimizer = setup_model_and_optimizer(
            model_name=model_name,
            model_params=model_params,
            device=device,
            saved_model_path=saved_model_path,  # e.g. for partial transfer
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            freeze_layers=freeze_layers,
        )

        # 2) Run training + validation epochs
        best_ckpt_path, training_logs = run_training_epochs(
            model=model,
            train_dataloader=train_dl,
            val_dataloader=val_dl,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            num_epochs=num_epochs,
            patience=patience,
            checkpoint_path=fold_ckpt,
            monitor_metrics=("val_kappa", "val_loss"),  
            monitor_modes=("max", "min"),
            model_name=model_name,
        )

        # 3) Finalize fold: load best checkpoint, evaluate, test, update best_results
        (test_true,
         test_pred,
         test_probs,
         test_ahi) = finalize_fold_or_repeat(
            model=model,
            checkpoint_path=best_ckpt_path,
            train_dataloader=train_dl,
            val_dataloader=val_dl,
            test_dataloader=test_dl,
            loss_fn=loss_fn,
            device=device,
            best_results_dict=best_results,
            do_test=True,
            task="sleep_staging",
            model_name=model_name,
        )

        # Accumulate test predictions
        all_true_labels.extend(test_true)
        all_predicted_labels.extend(test_pred)
        all_predicted_probabilities.extend(test_probs)
        all_ahi_values.extend(test_ahi)

    # ---- 4) Print mean ± std dev for each metric in best_results ----
    for key, values in best_results.items():
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{key}: {mean_val:.4f} ± {std_val:.4f}")
        else:
            print(f"{key}: No data available")

    # ---- 5) Compute overall metrics across folds ----
    print("\nOverall Metrics (Ablation):")
    (overall_acc,
     overall_f1,
     overall_kappa,
     rem_f1,
     auroc) = compute_metrics(
        all_predicted_labels,
        all_true_labels,
        pred_probs=all_predicted_probabilities,
        testing=True,
        task="sleep_staging",
        print_conf_matrix=True,
    )
    print(f"Overall Accuracy: {overall_acc:.4f}")
    print(f"Overall F1 Score: {overall_f1:.4f}")
    print(f"Overall Kappa: {overall_kappa:.4f}")
    print(f"REM F1 Score: {rem_f1:.4f}")
    print(f"AUROC: {auroc:.4f}")

    # AHI category metrics
    print("\nMetrics per AHI Category:")
    metrics_per_category = compute_metrics_per_ahi_category(
        all_true_labels,
        all_predicted_labels,
        all_predicted_probabilities,
        all_ahi_values,
    )
    for category, cat_metrics in metrics_per_category.items():
        print(f"\nMetrics for AHI Category '{category}':")
        for metric_name, val in cat_metrics.items():
            print(f"  {metric_name}: {val:.4f}")

    return best_results, overall_acc, overall_f1, overall_kappa, rem_f1, auroc

def train_cross_validate(
    model_name,
    model_params,         # Dictionary containing hyperparameters (including ablation flags, e.g. "use_tcn": True)
    dataloader_folds,
    saved_model_path=None,
    learning_rate=1e-3,
    weight_decay=1e-5,
    loss_fn=None,
    num_epochs=50,
    patience=10,
    device=None,
    checkpoint_path=None,
    freeze_layers=False,
):
    """
    Refactored cross-validation training using helper functions to reduce repetition.
    Trains from scratch (unless saved_model_path is specified) for each fold.

    Args:
        model_name (str): 
            One of ["sleepconvnet", "watchsleepnet", "insightsleepnet"].
            (If your 'watchsleepnet' is the ablation-enabled version, you'll place 'use_tcn', 'use_attention' in model_params.)
        model_params (dict):
            A dictionary containing the constructor arguments for your model. 
            Example for ablation:
                {
                  "num_features": 1, 
                  "num_channels": 256, 
                  "kernel_size": 5,
                  ...
                  "use_tcn": True,
                  "use_attention": False,
                  "num_classes": 3
                }
        dataloader_folds (list of tuples):
            A list where each item is (train_dataloader, val_dataloader, test_dataloader).
        saved_model_path (str or None):
            Optional path to load weights before each fold. If None, trains from scratch.
        learning_rate (float):
            The learning rate for the optimizer.
        weight_decay (float):
            L2 regularization coefficient.
        loss_fn (callable):
            The loss function (e.g. CrossEntropyLoss).
        num_epochs (int):
            Max number of epochs per fold.
        patience (int):
            Early stopping patience per fold.
        device (torch.device or None):
            'cuda' or 'cpu'. If None, tries to detect automatically.
        checkpoint_path (str or None):
            Base path for saving fold-specific checkpoints. 
            For fold i, file = checkpoint_path.replace('.pt', f'_fold{i}.pt').
        freeze_layers (bool):
            Whether to freeze some or all layers in the model.

    Returns:
        tuple: (results, overall_acc, overall_f1, overall_kappa, rem_f1, auroc)
          - results (dict): Aggregated metrics across folds.
          - overall_acc, overall_f1, overall_kappa, rem_f1, auroc (float):
            Final metrics on the combined test sets across folds.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # A dictionary to store aggregated results from all folds
    results = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "train_kappa": [],
        "train_rem_f1": [],
        "train_auroc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_kappa": [],
        "val_rem_f1": [],
        "val_auroc": [],
        "test_loss": [],
        "test_acc": [],
        "test_f1": [],
        "test_kappa": [],
        "test_rem_f1": [],
        "test_auroc": [],
    }

    # For final overall metrics (aggregating all folds' test predictions)
    all_true_labels = []
    all_predicted_labels = []
    all_predicted_probabilities = []
    all_ahi_values = []

    num_folds = len(dataloader_folds)
    print(f"Number of folds: {num_folds}")

    # Loop over folds
    for fold_idx, (train_dl, val_dl, test_dl) in enumerate(dataloader_folds, start=1):
        print(f"\n=== Fold {fold_idx}/{num_folds} ===")

        # Construct fold-specific checkpoint path if provided
        fold_checkpoint = None
        if checkpoint_path is not None:
            fold_checkpoint = checkpoint_path.replace(".pt", f"_fold{fold_idx}.pt")

        # ---- 1) Setup model & optimizer
        model, optimizer = setup_model_and_optimizer(
            model_name=model_name,
            model_params=model_params,
            device=device,
            saved_model_path=saved_model_path,  # If you want preloaded weights or not
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            freeze_layers=freeze_layers,
        )

        # ---- 2) Train for num_epochs with early stopping
        best_ckpt_path, training_logs = run_training_epochs(
            model=model,
            train_dataloader=train_dl,
            val_dataloader=val_dl,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            num_epochs=num_epochs,
            patience=patience,
            checkpoint_path=fold_checkpoint,
            monitor_metrics=("val_kappa", "val_loss"),  # multi-metric
            monitor_modes=("max", "min"),
            model_name=model_name,  # if special logic needed (gradient clipping for insightsleepnet)
        )

        # ---- 3) Reload best checkpoint
        if best_ckpt_path and os.path.exists(best_ckpt_path):
            model.load_state_dict(torch.load(best_ckpt_path))
        else:
            print("No checkpoint found for this fold. Using last trained state.")

        # Evaluate on train/val with best checkpoint
        (train_loss, train_acc, train_f1, train_kappa, train_rem_f1, train_auroc) = validate_step(
            model, train_dl, loss_fn, device
        )
        (val_loss, val_acc, val_f1, val_kappa, val_rem_f1, val_auroc) = validate_step(
            model, val_dl, loss_fn, device
        )

        # Store best training/validation metrics for this fold
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["train_f1"].append(train_f1)
        results["train_kappa"].append(train_kappa)
        results["train_rem_f1"].append(train_rem_f1)
        results["train_auroc"].append(train_auroc)

        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
        results["val_f1"].append(val_f1)
        results["val_kappa"].append(val_kappa)
        results["val_rem_f1"].append(val_rem_f1)
        results["val_auroc"].append(val_auroc)

        # ---- 4) Test the best model
        (test_loss, test_acc, test_f1, test_kappa, test_rem_f1, test_auroc) = validate_step(
            model, test_dl, loss_fn, device
        )
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        results["test_f1"].append(test_f1)
        results["test_kappa"].append(test_kappa)
        results["test_rem_f1"].append(test_rem_f1)
        results["test_auroc"].append(test_auroc)

        print(
            f"\nFold {fold_idx} Test: "
            f"Loss={test_loss:.4f}, Acc={test_acc:.3f}, F1={test_f1:.3f}, "
            f"Kappa={test_kappa:.3f}, REM_F1={test_rem_f1:.3f}, AUROC={test_auroc:.3f}"
        )

        # Retrieve test predictions for final metrics
        (test_true_labels,
         test_pred_labels,
         test_pred_probs,
         test_ahi_vals) = test_step(model, test_dl, device, task="sleep_staging")

        # If we find NaNs in test predictions/labels, skip final metrics
        if np.isnan(test_pred_labels).any() or np.isnan(test_true_labels).any():
            print("NaNs in test predictions/labels. Skipping final metrics for this fold.")
            continue

        # Accumulate them
        all_true_labels.extend(test_true_labels)
        all_predicted_labels.extend(test_pred_labels)
        all_predicted_probabilities.extend(test_pred_probs)
        all_ahi_values.extend(test_ahi_vals)

    # ---- 5) Print aggregated fold metrics (mean ± std)
    overall_acc = overall_f1 = overall_kappa = rem_f1 = auroc = 0.0
    for key, values in results.items():
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{key}: {mean_val:.4f} ± {std_val:.4f}")
        else:
            print(f"{key}: No data available")

    # ---- 6) Compute final overall metrics across folds
    if all_predicted_labels and all_true_labels:
        print("\nOverall Metrics Across Folds:")
        overall_acc, overall_f1, overall_kappa, rem_f1, auroc = compute_metrics(
            all_predicted_labels,
            all_true_labels,
            pred_probs=all_predicted_probabilities,
            testing=True,
            task="sleep_staging",
            print_conf_matrix=True,
        )
        print(f"Overall Accuracy: {overall_acc:.4f}")
        print(f"Overall F1 Score: {overall_f1:.4f}")
        print(f"Overall Kappa: {overall_kappa:.4f}")
        print(f"Overall REM F1 Score: {rem_f1:.4f}")
        print(f"AUROC: {auroc:.4f}")

        # AHI category metrics
        print("\nMetrics per AHI Category:")
        metrics_per_category = compute_metrics_per_ahi_category(
            all_true_labels,
            all_predicted_labels,
            all_predicted_probabilities,
            all_ahi_values,
        )
        for category, cat_metrics in metrics_per_category.items():
            print(f"\nMetrics for AHI Category '{category}':")
            for metric_name, val in cat_metrics.items():
                print(f"  {metric_name}: {val:.4f}")
    else:
        print("Insufficient data for final overall metrics.")

    return results, overall_acc, overall_f1, overall_kappa, rem_f1, auroc


def train_cross_validate_hpo(
    model_init,             # A callable that returns a new model instance with the desired hyperparams
    dataloader_folds,
    learning_rate=1e-3,
    weight_decay=1e-5,
    loss_fn=None,
    num_epochs=50,
    patience=10,
    device=None,
    checkpoint_path=None,
):
    """
    A refactored version of train_cross_validate_hpo that uses run_training_epochs
    and finalize_fold_or_repeat to handle multiple folds for hyperparameter optimization.

    Args:
        model_init (callable):
            A function/callable that returns a fresh model instance (nn.Module),
            pre-configured with whatever hyperparameters your HPO suggests.
        dataloader_folds (list of tuples):
            Each tuple is (train_dataloader, val_dataloader, test_dataloader) for each fold.
        learning_rate (float, optional):
            Learning rate for the optimizer. Defaults to 1e-3.
        weight_decay (float, optional):
            Weight decay (L2 regularization). Defaults to 1e-5.
        loss_fn (callable, optional):
            The loss function (e.g., CrossEntropyLoss). If None, must be provided or set globally.
        num_epochs (int):
            Max number of epochs to train per fold.
        patience (int):
            Early stopping patience per fold.
        device (torch.device or None):
            'cuda' or 'cpu'. If None, auto-detects.
        checkpoint_path (str or None):
            If provided, a base path for saving the best model checkpoint each fold.
            E.g., "model_checkpoint.pt" => "model_checkpoint_fold1.pt", etc.

    Returns:
        (results, overall_acc, overall_f1, overall_kappa, rem_f1, auroc)
          - results (dict): Aggregated metrics across folds (train, val, test).
          - overall_* (float): Final metrics computed across all folds' test sets.
    """

    # If device is not specified, detect automatically
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dictionary to store aggregated metrics from all folds
    results = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "train_kappa": [],
        "train_rem_f1": [],
        "train_auroc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_kappa": [],
        "val_rem_f1": [],
        "val_auroc": [],
        "test_loss": [],
        "test_acc": [],
        "test_f1": [],
        "test_kappa": [],
        "test_rem_f1": [],
        "test_auroc": [],
    }

    # To accumulate final test predictions from all folds
    all_true_labels = []
    all_predicted_labels = []
    all_predicted_probabilities = []
    all_ahi_values = []

    num_folds = len(dataloader_folds)
    print(f"Number of folds: {num_folds}")

    # Loop over folds
    for fold_idx, (train_dataloader, val_dataloader, test_dataloader) in enumerate(dataloader_folds, start=1):
        print(f"\n=== Fold {fold_idx}/{num_folds} ===")

        # Build a path for this fold's checkpoint if base path is given
        fold_checkpoint_path = None
        if checkpoint_path is not None:
            fold_checkpoint_path = checkpoint_path.replace(".pt", f"_fold{fold_idx}.pt")

        # 1) Initialize the model using the provided model_init callable
        model = model_init()
        model.to(device)

        # Create the optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # 2) Train + validate with run_training_epochs
        best_ckpt_path, training_logs = run_training_epochs(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            num_epochs=num_epochs,
            patience=patience,
            checkpoint_path=fold_checkpoint_path,
            monitor_metrics=("val_kappa", "val_loss"),  # multi-metric early stopping
            monitor_modes=("max", "min"),
            model_name=None,  # set if you have special gradient clipping logic in train_step
        )

        # 3) Reload best checkpoint if it exists
        if best_ckpt_path and os.path.exists(best_ckpt_path):
            model.load_state_dict(torch.load(best_ckpt_path))
        else:
            print("No checkpoint found for this fold. Using current model state.")

        # Evaluate final train/val metrics with the best checkpoint
        train_loss, train_acc, train_f1, train_kappa, train_rem_f1, train_auroc = validate_step(
            model, train_dataloader, loss_fn, device
        )
        val_loss, val_acc, val_f1, val_kappa, val_rem_f1, val_auroc = validate_step(
            model, val_dataloader, loss_fn, device
        )

        # Store final train/val metrics
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["train_f1"].append(train_f1)
        results["train_kappa"].append(train_kappa)
        results["train_rem_f1"].append(train_rem_f1)
        results["train_auroc"].append(train_auroc)

        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
        results["val_f1"].append(val_f1)
        results["val_kappa"].append(val_kappa)
        results["val_rem_f1"].append(val_rem_f1)
        results["val_auroc"].append(val_auroc)

        # 4) Test the best model
        test_loss, test_acc, test_f1, test_kappa, test_rem_f1, test_auroc = validate_step(
            model, test_dataloader, loss_fn, device
        )
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        results["test_f1"].append(test_f1)
        results["test_kappa"].append(test_kappa)
        results["test_rem_f1"].append(test_rem_f1)
        results["test_auroc"].append(test_auroc)

        print(
            f"\nFold {fold_idx} Test: "
            f"Loss={test_loss:.4f}, Acc={test_acc:.3f}, F1={test_f1:.3f}, "
            f"Kappa={test_kappa:.3f}, REM_F1={test_rem_f1:.3f}, AUROC={test_auroc:.3f}"
        )

        # Gather test predictions for final metrics
        (test_true_labels,
         test_predicted_labels,
         test_predicted_probs,
         test_ahi_vals) = test_step(model, test_dataloader, device, task="sleep_staging")

        # If we detect NaNs, skip final metrics for this fold
        if np.isnan(test_predicted_labels).any() or np.isnan(test_true_labels).any():
            print("NaN in test predictions/labels. Skipping final metrics for this fold.")
            continue

        # Accumulate them for overall metrics
        all_true_labels.extend(test_true_labels)
        all_predicted_labels.extend(test_predicted_labels)
        all_predicted_probabilities.extend(test_predicted_probs)
        all_ahi_values.extend(test_ahi_vals)

    # 5) Print aggregated fold metrics (mean ± std)
    overall_acc = overall_f1 = overall_kappa = rem_f1 = auroc = 0.0
    for key, values in results.items():
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{key}: {mean_val:.4f} ± {std_val:.4f}")
        else:
            print(f"{key}: No data available")

    # 6) Compute final overall metrics across folds (aggregated test data)
    if all_predicted_labels and all_true_labels:
        print("\nOverall Metrics Across All Folds (HPO):")
        (overall_acc,
         overall_f1,
         overall_kappa,
         rem_f1,
         auroc) = compute_metrics(
            all_predicted_labels,
            all_true_labels,
            pred_probs=all_predicted_probabilities,
            testing=True,
            task="sleep_staging",
            print_conf_matrix=True,
        )
        print(f"Overall Accuracy: {overall_acc:.4f}")
        print(f"Overall F1 Score: {overall_f1:.4f}")
        print(f"Overall Kappa: {overall_kappa:.4f}")
        print(f"REM F1 Score: {rem_f1:.4f}")
        print(f"AUROC: {auroc:.4f}")

        # Compute AHI category metrics if desired
        print("\nMetrics per AHI Category:")
        metrics_per_category = compute_metrics_per_ahi_category(
            all_true_labels,
            all_predicted_labels,
            all_predicted_probabilities,
            all_ahi_values,
        )
        for category, cat_metrics in metrics_per_category.items():
            print(f"\nMetrics for AHI Category '{category}':")
            for metric_name, val in cat_metrics.items():
                print(f"  {metric_name}: {val:.4f}")
    else:
        print("No data available for final overall metrics.")

    return results, overall_acc, overall_f1, overall_kappa, rem_f1, auroc
