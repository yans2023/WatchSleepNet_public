import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import KFold
from pathlib import Path
import json
seed = 0

class SSDataset(Dataset):
    def __init__(self, dir, dataset, file_list = None, multiplier=1.0, downsample_rate=1, task="sleep_staging", return_file_name=False):
        self.dir = dir
        self.dataset = dataset
        self.multiplier = multiplier
        self.downsample_rate = downsample_rate
        self.task = task
        self.return_file_name = return_file_name
        if file_list:
            # If file list is provided, use only those files
            self.files = [self.dir / file_name for file_name in file_list]
        else:
            # If no file list is provided, follow the previous file selection logic
            if "shhs" in dataset and "mesa" in dataset:
                self.files = [file for file in self.dir.glob("*.npz")]
            elif "shhs" in dataset:
                self.files = [
                    file
                    for file in self.dir.glob("shhs1*")
                    if file.name != "shhs1-204822.npz"
                ]
            elif "dreamt" in dataset:
                self.files = [file for file in self.dir.glob("*.npz")]
            elif "mesa" in dataset:
                self.files = [file for file in self.dir.glob("*.npz")]
                print("MESA contains files: ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        data = np.load(file_path, allow_pickle=True)
        ibi = data["data"].flatten() * self.multiplier # Remove the second dimension
        labels = data["stages"]
        fs = data["fs"].item()
        ahi = data["ahi"].item()

        # Downsample 125 Hz to 25 Hz
        ibi = ibi[::self.downsample_rate]
        labels = labels[::self.downsample_rate]

        # Calculate the number of samples for each sequence element and number of ibi segments
        num_samples = int((fs / self.downsample_rate) * 30) # for 30 second segments
        num_segments = len(ibi) // num_samples

        # Reshape the ibi and labels array
        ibi = ibi[:num_samples*num_segments]
        ibi = ibi.reshape(num_segments, num_samples)
        labels = labels[::num_samples]
        labels = labels[:num_segments]
        labels = self.remap_labels(labels, self.task)

        if num_segments > 1100:
            ibi = ibi[:1100]
            labels = labels[:1100]
            num_segments = 1100

        # Convert data to PyTorch tensors
        ibi = torch.from_numpy(ibi).float()
        labels = torch.from_numpy(labels).long()

        if self.return_file_name:
            return ibi, labels, num_segments, ahi, file_path.stem
        else:
            return ibi, labels, num_segments, ahi

    def remap_labels(self, labels, task):
        # 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM, 5=Movement (mapped to -1 to ignore in loss computation)
        # simplify to do 3-stage classification
        if task == "sleep_staging":
            label_map = {0:0, 1:1, 2:1, 3:1, 4:2, 5:-1, -1:-1}
        elif task == "sleep_wake":
            label_map = {0:0, 1:1, 2:1, 3:1, 4:1, 5:-1, -1:-1}
        remapped_labels = np.vectorize(label_map.get)(labels)
        return remapped_labels

    @staticmethod
    def collate_fn(batch):
        ibis, labels, lengths, ahis = zip(*batch)
        lengths = torch.tensor(lengths)

        ibis_padded = pad_sequence(ibis, batch_first=True, padding_value=0)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1)

        return ibis_padded, labels_padded, lengths, ahis


def create_dataloaders(dir, train_ratio, val_ratio, batch_size, num_workers, dataset, multiplier=1.0, downsampling_rate=1, task="sleep_staging"):
    """ Create train, validation, and test data loaders
    test_ratio = 1 - train_ratio - val_ratio

    Args:
        dir (_type_): _description_
        train_ratio (_type_): _description_
        val_ratio (_type_): _description_
        batch_size (_type_): _description_
        num_workers (_type_): _description_
        dataset (_type_): _description_
        multiplier (float, optional): _description_. Defaults to 1.0.
        downsampling_rate (int, optional): _description_. Defaults to 1.
        task (str, optional): _description_. Defaults to "sleep_staging".

    Returns:
        _type_: _description_
    """
    # Initialize the ibi dataset
    dataset = SSDataset(dir=dir, dataset=dataset, multiplier=multiplier, downsample_rate=downsampling_rate, task=task)
    # Create train/val/test split
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    # Construct the data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=SSDataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=SSDataset.collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=SSDataset.collate_fn
    )

    return train_loader, val_loader, test_loader


def create_dataloaders_kfolds(
    dir,
    dataset,
    num_folds=5,
    val_ratio=0.1,
    batch_size=16,
    num_workers=8,
    multiplier=1.0,
    downsampling_rate=1,
):
    # Initialize the ibi dataset
    dataset = SSDataset(
        dir=dir, multiplier=multiplier, dataset=dataset, downsample_rate=downsampling_rate
    )

    kfold = KFold(n_splits=num_folds, shuffle=True)
    folds = []

    for train_idx, test_idx in kfold.split(dataset):
        train_dataset = torch.utils.data.Subset(dataset, train_idx)

        # Create train/val split
        train_size = int(len(train_dataset))
        val_size = int(val_ratio * len(dataset))
        train_size = train_size - val_size
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size]
        )

        test_dataset = torch.utils.data.Subset(dataset, test_idx)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=SSDataset.collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=SSDataset.collate_fn,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=SSDataset.collate_fn,
        )
        folds.append((train_loader, val_loader, test_loader))

    return folds

# Test function
def test_dataloader_folds(dataloader_folds):
    for fold_index, (train_loader, val_loader, test_loader) in enumerate(dataloader_folds):
        print(f"Testing Fold {fold_index + 1}")

        # Test each loader
        for loader_name, loader in zip(['Train', 'Validation', 'Test'], [train_loader, val_loader, test_loader]):
            print(f"Loader: {loader_name}")
            for i, (X, y, lengths, AHIs) in enumerate(loader):
                print(f"Batch {i + 1}")
                print("X shape:", X.shape)  # Shape of the input batch
                print("y shape:", y.shape)  # Shape of the label batch
                print("Lengths:", lengths)  # Length of each sequence in the batch
                print("AHI scores: ", AHIs)
                if i == 0:  # Only print for the first batch to avoid excessive output
                    break


def create_experiment_dataloaders(
    shhs_mesa_dir,
    dreamt_dir,
    severity_json,
    experiment_type="control",
    train_ratio=0.8,
    val_ratio=0.2,
    batch_size=16,
    num_workers=8,
    multiplier=1.0,
    downsampling_rate=1,
    seed=None,
):
    """
    Create custom dataloaders based on experiment type and severity of apnea.
    """
    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)
    
    # Load severity info
    with open(severity_json, "r") as f:
        severity_info = json.load(f)

    # Select files based on experiment type
    selected_files, finetune_files, test_files = [], [], []
    dreamt_mild_normal = (severity_info["dreamt"]["normal"] + severity_info["dreamt"]["mild"])
    dreamt_mod_severe = (severity_info["dreamt"]["moderate"] + severity_info["dreamt"]["severe"])
    # test_files = random.sample(dreamt_mod_severe, len(dreamt_mod_severe) // 2)
    test_files = severity_info["dreamt"]["severe"]
    if experiment_type == "control":
        # Add both normal/mild and moderate/severe for SHHS+MESA
        selected_files += (
            severity_info["shhs"]["normal"]
            + severity_info["shhs"]["mild"]
            + severity_info["mesa"]["normal"]
            + severity_info["mesa"]["mild"]
            + severity_info["shhs"]["moderate"]
            + severity_info["shhs"]["severe"]
            + severity_info["mesa"]["moderate"]
            + severity_info["mesa"]["severe"]
        )

        # normal_mild_len = len(severity_info["shhs"]["normal"]) + len(severity_info["shhs"]["mild"]) + len(severity_info["mesa"]["normal"]) + len(severity_info["mesa"]["mild"])
        selected_files = random.sample(selected_files, len(severity_info["shhs"]["normal"]+severity_info["mesa"]["normal"]))

        # Construct finetune test files
        finetune_files = random.sample(
            dreamt_mild_normal+severity_info["dreamt"]["moderate"],
            len(severity_info["dreamt"]["normal"]),
        )

    elif experiment_type == "data_drift":
        # Only normal for SHHS+MESA
        selected_files += (
            severity_info["shhs"]["normal"]
            # + severity_info["shhs"]["mild"]
            + severity_info["mesa"]["normal"]
            # + severity_info["mesa"]["mild"]
        )

        # Split DREAMT for finetuning and testing
        finetune_files = severity_info["dreamt"]["normal"]
    else:
        raise ValueError("Invalid experiment type. Choose 'control' or 'data_drift'.")
    
    # print all file list lengths
    print("Selected Files: ", len(selected_files))
    print("Finetune Files: ", len(finetune_files))
    print("Test Files: ", len(test_files))
    
    
    # Initialize datasets
    pretrain_dataset = SSDataset(
        dir=shhs_mesa_dir,
        dataset="shhs_mesa_ibi",
        file_list=selected_files,
        multiplier=multiplier,
        downsample_rate=downsampling_rate,
    )

    finetune_dataset = SSDataset(
        dir=dreamt_dir,
        dataset="dreamt_pibi",
        file_list=finetune_files,
        multiplier=multiplier,
        downsample_rate=downsampling_rate,
    )

    test_dataset = SSDataset(
        dir=dreamt_dir,
        dataset="dreamt_pibi",
        file_list=test_files,
        multiplier=multiplier,
        downsample_rate=downsampling_rate,
    )

    # Split pretrain dataset into train/val
    train_size = int(train_ratio * len(pretrain_dataset))
    val_size = int(val_ratio * len(pretrain_dataset))
    test_size = len(pretrain_dataset) - train_size - val_size

    train_dataset, val_dataset, _ = random_split(
        pretrain_dataset, [train_size, val_size, test_size]
    )

    # Split finetune dataset into train/val
    finetune_train_size = int(train_ratio * len(finetune_dataset))
    finetune_val_size = len(finetune_dataset) - finetune_train_size

    finetune_train_dataset, finetune_val_dataset = random_split(
        finetune_dataset, [finetune_train_size, finetune_val_size]
    )

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=SSDataset.collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=SSDataset.collate_fn,
    )

    finetune_train_loader = DataLoader(
        finetune_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=SSDataset.collate_fn,
    )

    finetune_val_loader = DataLoader(
        finetune_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=SSDataset.collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=SSDataset.collate_fn,
    )

    return (
        train_loader,
        val_loader,
        finetune_train_loader,
        finetune_val_loader,
        test_loader,
    )


def test_experiment_dataloaders(
    shhs_mesa_dir, dreamt_dir, severity_json_path, batch_size=16, num_workers=8, seed=0
):
    """Test the custom dataloaders for both control and data drift experiments with different directories.

    Args:
        shhs_mesa_dir (str): Path to the SHHS_MESA_IBI dataset directory.
        dreamt_dir (str): Path to the DREAMT dataset directory.
        severity_json_path (str): Path to the JSON file containing apnea severity information.
        batch_size (int, optional): Batch size for the dataloader. Defaults to 16.
        num_workers (int, optional): Number of workers for the dataloader. Defaults to 8.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
    """
    print("Running Control Experiment...\n")

    # Run Control Experiment
    train_loader, val_loader, finetune_train_loader, finetune_val_loader, test_loader = (
        create_experiment_dataloaders(
            shhs_mesa_dir=shhs_mesa_dir,
            dreamt_dir=dreamt_dir,
            severity_json=severity_json_path,
            experiment_type="control",
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
        )
    )

    print(f"Control Experiment Train Loader has {len(train_loader)} batches.")
    for ibi, labels, num_segments, ahi in train_loader:
        print(
            f"Train - ibi shape: {ibi.shape}, Labels shape: {labels.shape}, AHI: {ahi}"
        )
        break

    print(f"Control Experiment Validation Loader has {len(val_loader)} batches.")
    for ibi, labels, num_segments, ahi in val_loader:
        print(
            f"Validation - ibi shape: {ibi.shape}, Labels shape: {labels.shape}, AHI: {ahi}"
        )
        break

    print(f"Control Experiment Finetune Train Loader has {len(finetune_train_loader)} batches.")
    for ibi, labels, num_segments, ahi in finetune_train_loader:
        print(
            f"Finetune - ibi shape: {ibi.shape}, Labels shape: {labels.shape}, AHI: {ahi}"
        )
        print(np.unique(labels.numpy(), return_counts=True))

    print(f"Control Experiment Finetune Validation Loader has {len(finetune_val_loader)} batches.")
    for ibi, labels, num_segments, ahi in finetune_val_loader:
        print(
            f"Finetune - ibi shape: {ibi.shape}, Labels shape: {labels.shape}, AHI: {ahi}"
        )
        print(np.unique(labels.numpy(), return_counts=True))

    print(f"Control Experiment Test Loader has {len(test_loader)} batches.")
    for ibi, labels, num_segments, ahi in test_loader:
        print(
            f"Test - ibi shape: {ibi.shape}, Labels shape: {labels.shape}, AHI: {ahi}"
        )
        print(np.unique(labels.numpy(), return_counts=True))


# Example usage
if __name__ == "__main__":
    test_experiment_dataloaders(
        shhs_mesa_dir = Path("/mnt/nvme2/") / "SHHS_MESA_IBI", 
        dreamt_dir = Path("/mnt/nvme2/") / "DREAMT_PIBI_SE", 
        severity_json_path = "cleaned_files_apnea_severity.json", 
        batch_size=16, 
        num_workers=8, 
        seed=0
    )
