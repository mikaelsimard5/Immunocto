import lightning as L
from sklearn import preprocessing
import toml
import torch

from Data_loading.dataloader_digpat import get_tile_dataset, DataGenerator, digpat_datamodule
from models.HE_mask_classifiers import SAM_ConvNet
from Utils.parser import parse_arguments
from Utils.esthetics import generate_model_name
from Utils.training_prep import get_logger, get_callbacks, get_class_weights, get_transforms


def load_config(config_file):
    return toml.load(config_file)


def main(config_file=None, n_gpus=1, gpu_devices=0):
    config = load_config(config_file)
    config['ADVANCEDMODEL']['n_gpus'] = n_gpus

    print("********************************************************************************************************")
    print(f"Available GPUs: {torch.cuda.device_count()}, {config['ADVANCEDMODEL']['n_gpus']} GPUs are used for training.")
    print("********************************************************************************************************")
    # Load dataset and print stats

    tile_dataset = get_tile_dataset(config['DATA']['data_file'], valid_labels=config['DATA'].get("valid_labels", None))
    config['DATA']['n_classes'] = len(tile_dataset['label'].unique())
    print(f"There are {config['DATA']['n_classes']} classes in the training dataset.")
    value_counts = tile_dataset['label'].value_counts()
    print("\n".join([f"{label}     : {count} instances ({count/len(tile_dataset) * 100:.1f}%)" for label, count in value_counts.items()]))
    print("********************************************************************************************************")
    # Create callbacks, transforms, label encoder, trainer and dataset

    model_name = generate_model_name(config)
    logger = get_logger(config, model_name)
    callbacks = get_callbacks(config, model_name=model_name)
    L.seed_everything(config['ADVANCEDMODEL']['random_seed'], workers=True)
    torch.set_float32_matmul_precision('medium')

    train_transform, val_transform = get_transforms(config)
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(tile_dataset['label']) 

    # Encode tile_dataset classes
    tile_dataset['label'] = label_encoder.transform(tile_dataset['label'])

    trainer = L.Trainer(devices=gpu_devices,
                        strategy="ddp",
                        accelerator="gpu",
                        max_epochs=config['ADVANCEDMODEL']['max_epochs'],
                        precision=config['BASEMODEL']['precision'],
                        callbacks=callbacks,
                        logger=logger,
                        check_val_every_n_epoch=1,
                        log_every_n_steps=5,
                        sync_batchnorm=True)

    config['DATA']['weights'] = get_class_weights(tile_dataset, label_encoder)   
    data = digpat_datamodule(tile_dataset, config=config, train_transform=train_transform, val_transform=val_transform)

    print("********************************************************************************************************")
    # train model

    model = SAM_ConvNet(config, LabelEncoder = label_encoder)
    trainer.fit(model, datamodule=data)

    if config['DATA']['test_size']>0:    
        trainer.test(model, data.test_dataloader())


if __name__ == "__main__":

    config_file, n_gpus, gpu_devices = parse_arguments()
    main(config_file, n_gpus=n_gpus, gpu_devices=gpu_devices)

