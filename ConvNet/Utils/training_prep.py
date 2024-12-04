from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import torch
from torchvision.transforms import v2
from Utils import ColourAugment


def get_logger(config: dict, model_name: str) -> TensorBoardLogger:
    """
    Initialize the TensorBoard logger.

    Args:
        config (dict): Configuration dictionary.
        model_name (str): Name of the model.

    Returns:
        TensorBoardLogger: Configured logger instance.
    """
    logger_folder = config['CHECKPOINT'].get('logger_folder', './logs')
    return TensorBoardLogger(logger_folder, name=model_name)


def get_callbacks(config: dict, model_name: str):
    """
    Get a list of callbacks for the model training.

    Args:
        config (dict): Configuration dictionary.
        model_name (str): Name of the model.

    Returns:
        list: List of Lightning callbacks.
    """
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        monitor=config['CHECKPOINT'].get('monitor', 'val_loss'),
        filename=f"model-epoch{{epoch:02d}}-valacc{{val_accuracy_macro_epoch:.2f}}",
        save_top_k=1,
        mode=config['CHECKPOINT'].get('mode', 'min')
    )

    return [lr_monitor, checkpoint_callback]


def get_class_weights(tile_dataset, label_encoder) -> torch.Tensor:
    """
    Calculate class weights based on dataset class distribution.

    Args:
        tile_dataset (pd.DataFrame): Dataset with a 'label' column.
        label_encoder (LabelEncoder): Encoder for class labels.

    Returns:
        torch.Tensor: Normalized class weights.
    """
    class_counts = tile_dataset['label'].value_counts(normalize=True)
    num_classes = len(label_encoder.classes_)
    weights = np.zeros(num_classes)

    for idx, count in zip(class_counts.index, class_counts):
        weights[idx] = 1 / count

    weights = torch.tensor(weights / weights.sum(), dtype=torch.float32)
    return weights


def get_transforms(config: dict):
    """
    Generate training and validation data transformations.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        tuple: (train_transform, val_transform)
    """
    # Base transformations for training
    train_transform_list = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=False),
        v2.RandomHorizontalFlip(p=0.4),
        v2.RandomVerticalFlip(p=0.4),
        ColourAugment.ColourAugment(
            sigma=config['AUGMENTATION']['colour_sigma'],
            mode=config['AUGMENTATION']['colour_mode']
        )
    ]

    # Base transformations for validation
    val_transform_list = [v2.ToImage()]

    # Additional transformations based on pretrained configuration
    if config['ADVANCEDMODEL'].get('pretrained', False):
        print('Using pretrained network - scaling [0-1] and normalizing for torchvision models.')
        additional_transforms = [
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    else:
        print('Training from scratch - no scaling or normalization applied.')
        additional_transforms = [v2.ToDtype(torch.float32, scale=False)]

    train_transform = v2.Compose(train_transform_list + additional_transforms)
    val_transform = v2.Compose(val_transform_list + additional_transforms)

    return train_transform, val_transform
