from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import torch
from torchvision.transforms import v2


from Utils import ColourAugment


def get_logger(config, model_name):
    return TensorBoardLogger(config['CHECKPOINT']['logger_folder'], name=model_name)


def get_callbacks(config, model_name):
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        monitor=config['CHECKPOINT']['monitor'],
        filename  = f"model-epoch{{epoch:02d}}-valacc{{val_accuracy_macro_epoch:02f}}",
        save_top_k=1,
        mode=config['CHECKPOINT']['mode'])

    return [lr_monitor, checkpoint_callback]


def get_class_weights(tile_dataset, label_encoder):
    class_counts = tile_dataset['label'].value_counts(normalize=True)
    indices = class_counts.index
    num_classes = len(label_encoder.classes_)
    w = np.zeros(num_classes)
    for idx, count in zip(indices, class_counts):
        w[idx] = 1 / count    
    w = torch.tensor(w / np.sum(w))

    return w


def get_transforms(config): # transforms made on CPU - should just be image formatting.

    train_transform_list = [v2.ToImage(),
                            v2.ToDtype(torch.float32, scale=False), # because color augment takes float32
                            v2.RandomHorizontalFlip(p=0.4),
                            v2.RandomVerticalFlip(p=0.4),
                            ColourAugment.ColourAugment(sigma=config['AUGMENTATION']['colour_sigma'],
                                                        mode=config['AUGMENTATION']['colour_mode'])]

    val_transform_list = [v2.ToImage()]

    if config['ADVANCEDMODEL']['pretrained']:
        # if using a pretrained torchvision network, scale 0-1 and normalise
        print('Assuming pre-trained network - scaling [0-1] then normalising according to torchvision models.')
        tr = [v2.ToDtype(torch.float32, scale=True),
              v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    else:
        print('Assuming network trained from scratch - no data scaling done.')
        # Otherwise, just set array to float32 by default.
        tr = [v2.ToDtype(torch.float32, scale=False)]

    train_transform = v2.Compose(train_transform_list + tr)
    val_transform = v2.Compose(val_transform_list + tr)

    return train_transform, val_transform