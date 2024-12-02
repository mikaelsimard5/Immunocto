import boto3
import cv2
import matplotlib.pyplot as plt
import sys
import os
import datetime
import torch
from torch import cuda
from torchvision import transforms
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
import lightning as L
import toml
from sklearn import preprocessing
from lightning.pytorch.strategies import DDPStrategy
from dataloader.Dataloader import DataGenerator, DataModule
from Utils import ColourAugment
import datetime
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import ast
from tqdm import tqdm
from torchvision.transforms import v2
from PIL import Image

from models.CellClassifier import base_convnet
from models.Cell_Classifier import HE_mask_cell_classifier


def get_model_name(config, class_counts=None):
    backbone = config['BASEMODEL'].get('Backbone','NA')
    batch = config['BASEMODEL'].get('Batch_Size', 'NA')
    loss = config['BASEMODEL'].get('Loss_Function', 'NA')
    epochs = config['ADVANCEDMODEL'].get('Max_Epochs','NA')
    seed = config['ADVANCEDMODEL'].get('Random_Seed','NA')
    modeltype = config['PROCESSING_TYPE'].get('Model','NA')

    if isinstance(config['DATA']['Train_Size'], list) and isinstance(config['DATA']['Val_Size'], list) and isinstance(config['DATA']['Test_Size'], list):
        n = len(config['DATA']['Train_Size']) + len(config['DATA']['Val_Size']) + len(config['DATA']['Test_Size'])
        trsize = round(len(config['DATA']['Train_Size'])/n, 3)
        valsize = round(len(config['DATA']['Val_Size'])/n, 3)
    else:
        trsize = config['DATA'].get('Train_Size','NA')
        valsize = config['DATA'].get('Val_Size','NA')
    lr = config['OPTIMIZER']['lr']
    ls = config['REGULARIZATION'].get('Label_Smoothing','NA')
    WD = config['REGULARIZATION'].get('Weight_Decay','NA')
    Nc = config['DATA']['N_Classes']
    target_pixel_size = config['BASEMODEL']['Target_Pixel_Size']
    target_patch_size = config['BASEMODEL']['Target_Patch_Size']

    if class_counts is not None:
        classes = '_'.join([f"{count}{class_name}" for class_name, count in class_counts.items()])
    else:
        classes = 'NA'
    sched = config['SCHEDULER'].get('type','NA')
    nGPUS = config['ADVANCEDMODEL']['n_gpus']

    if config['ADVANCEDMODEL']['Pretrained'] == True:
        pretrained="pretrained"
    else:
        pretrained="scratch"

    b1 = f"{backbone}_{pretrained}_b{batch}_{loss}_{epochs}epochs_s{seed}_{modeltype}"
    b1p5 = f"_pixsize_{target_pixel_size}_patchsize_{target_patch_size[0]}"
    b2 = f"_{Nc}classes_{classes}_tr{trsize}_val{valsize}_lr{lr}_ls{ls}_WD{WD}"
    b3 = f"_sched{sched}_nGPUS{nGPUS}"

    return b1+b1p5+b2+b3


def load_config(config_file):
    return toml.load(config_file)


def get_logger(config, model_name):
    return TensorBoardLogger(config['CHECKPOINT']['logger_folder'], name=model_name)


def get_callbacks(config, model_name):
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        monitor=config['CHECKPOINT']['Monitor'],
        filename  = f"model-epoch{{epoch:02d}}-valacc{{val_accuracy_macro_epoch:02f}}",
        save_top_k=1,
        mode=config['CHECKPOINT']['Mode'])

    return [lr_monitor, checkpoint_callback]


def get_class_weights(tile_dataset, label_encoder):
    class_counts = tile_dataset['classes'].value_counts(normalize=True)
    indices = class_counts.index
    num_classes = len(label_encoder.classes_)
    w = np.zeros(num_classes)
    for idx, count in zip(indices, class_counts):
        print(count)
        w[idx] = 1 / count    
    w = torch.tensor(w / np.sum(w))

    return w


def load_dataset(config):
    tile_dataset = pd.read_csv(config['CRITERIA']['dataset'])
    # Rename "class" to "classes" if misnamed
    if "class" in tile_dataset.columns:
        tile_dataset.rename(columns={'class': 'classes'}, inplace=True)
    return tile_dataset


def get_transforms(config): # transforms made on CPU - should just be image formatting.

    train_transform_list = [v2.ToImage(),
                            v2.ToDtype(torch.float32, scale=False), # because color augment takes float32
                            v2.RandomHorizontalFlip(p=0.4),
                            v2.RandomVerticalFlip(p=0.4),
                            ColourAugment.ColourAugment(sigma=config['AUGMENTATION']['Colour_Sigma'],
                                                        mode=config['AUGMENTATION']['Colour_Mode'])]

    val_transform_list = [v2.ToImage()]

    if config['ADVANCEDMODEL']['Pretrained']:
        # if using a pretrained torchvision network, scale 0-1 and normalise
        print('Assuming pre-trained network ; scaling [0-1] then normalising according to torchvision models.')
        tr = [v2.ToDtype(torch.float32, scale=True),
              v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    else:
        print('Assuming network trained from scratch ; no data scaling done.')
        # Otherwise, just set array to float32 by default.
        tr = [v2.ToDtype(torch.float32, scale=False)]

    train_transform = v2.Compose(train_transform_list + tr)
    val_transform = v2.Compose(val_transform_list + tr)

    return train_transform, val_transform

def getinfo(data, label_encoder):
    # Print stats on the classes in training and validation, as well as their distribution.

    print('------------------------------------------------------------')
    print(f"Total number of examples in training dataset: {len(data.train_data.tile_dataset)}")
    data.train_data.tile_dataset['dc'] = label_encoder.inverse_transform(data.train_data.tile_dataset['classes'])
    print(data.train_data.tile_dataset['dc'].value_counts())

    print(f"Total number of examples in validation dataset: {len(data.val_data.tile_dataset)}")
    data.val_data.tile_dataset['dc'] = label_encoder.inverse_transform(data.val_data.tile_dataset['classes'])
    print(data.val_data.tile_dataset['dc'].value_counts())
    print('------------------------------------------------------------')

def main(config, n_gpus_literal):

    if isinstance(n_gpus_literal, list):
        config['ADVANCEDMODEL']['n_gpus'] = len(n_gpus_literal)
    else:
        config['ADVANCEDMODEL']['n_gpus'] = n_gpus_literal

    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"{config['ADVANCEDMODEL']['n_gpus']} GPUs are used for training")    

    tile_dataset = load_dataset(config)

    #tile_dataset = tile_dataset.sample(n=5000)  
    # Possible manual fiddling
    #tile_dataset.loc[tile_dataset['classes'] == 'layer_CD4T', 'classes'] = 'lymphocite'
    #tile_dataset.loc[tile_dataset['classes'] == 'layer_CD8T', 'classes'] = 'lymphocite'
    #print('Replaced CD4 and CD8 to lymphocite!')

    print('Unique classes:')
    print(tile_dataset['classes'].value_counts())

    config['DATA']['N_Classes'] = tile_dataset['classes'].nunique()
    model_name = get_model_name(config, class_counts=tile_dataset['classes'].value_counts())
    logger = get_logger(config, model_name)
    callbacks = get_callbacks(config, model_name=model_name)    

    L.seed_everything(config['ADVANCEDMODEL']['Random_Seed'], workers=True)
    torch.set_float32_matmul_precision('medium')

    train_transform, val_transform = get_transforms(config)
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(tile_dataset['classes'])

    # If single patient, sample by patch; otherwise sample by WSI.
    if tile_dataset['patient'].nunique() == 1:
        sampling = "patch"
    else:
        sampling = "WSI"

    trainer = L.Trainer(devices=n_gpus_literal,
                        strategy="ddp",
                        accelerator="gpu",
                        max_epochs=config['ADVANCEDMODEL']['Max_Epochs'],
                        precision=config['BASEMODEL']['Precision'],
                        callbacks=callbacks,
                        logger=logger,
                        check_val_every_n_epoch=1,
                        log_every_n_steps=5,
                        sync_batchnorm=True)

    data = DataModule(tile_dataset, config=config, train_transform=train_transform,
                      val_transform=val_transform, label_encoder=label_encoder, sampling=sampling)

    getinfo(data, label_encoder) 

    config['DATA']['weights'] = get_class_weights(tile_dataset, label_encoder)

    if config['PROCESSING_TYPE']['Model'] == 'HE+IF':
        model = base_convnet(config, LabelEncoder = label_encoder, n_channels = 4+len(config['CRITERIA']['IF_markers']))
    elif config['PROCESSING_TYPE']['Model'] == 'HE':
        model = HE_mask_cell_classifier(config, LabelEncoder = label_encoder)
    else:
        raise ValueError(f"Undefined model name: {config['PROCESSING_TYPE']['Model']}. Current options are 'HE+IF', or 'HE'.")

    trainer.fit(model, datamodule=data)
    #if isinstance(config['DATA']['Test_Size'], list) 
    #if config['DATA']['Test_Size']>0:
    trainer.test(model, data.test_dataloader())

    # For current easy debugging, extract the validation results
    if trainer.is_global_zero:

        # Get best model according to training
        best_model_path = trainer.checkpoint_callback.best_model_path

        preds = trainer.predict(model, data.val_dataloader(), ckpt_path=best_model_path)        
        probs = torch.cat([prob for prob, _ in preds], dim=0).cpu()
        predicted_labels = torch.argmax(probs, dim=1) # Cat results form all batches
        labels = torch.cat([att for _, att in preds], dim=0).cpu() # Cat results form all batches                
        predicted_class = label_encoder.inverse_transform(predicted_labels)
        true_class      = label_encoder.inverse_transform(labels)

        df = data.val_data.tile_dataset
        df['predicted_class'] = predicted_class
        df['prob_predicted_class'], _ = torch.max(probs, dim=1) # the prob of the dominant class
        val_results_csv_path = f'/home/ubuntu/val_test_results_{model_name}.csv'
        df.to_csv(val_results_csv_path, index=False)
        print("************************************************************")
        print(f'Exported validation results to {val_results_csv_path}.')
        print("************************************************************")

    print('-------------------------')

if __name__ == "__main__":

    if len(sys.argv) > 2:
        n_gpus = ast.literal_eval(sys.argv[2])
    else:
        n_gpus = 1

    config = load_config(sys.argv[1])

    #random_seeds = [22, 69, 1994, 1991, 66]
    random_seeds = [config['ADVANCEDMODEL']['Random_Seed']]

    for idx, random_seed in enumerate(random_seeds):
        config['ADVANCEDMODEL']['Random_Seed'] = random_seed
        print('===============================================================')
        print(f'TRAINING MODEL {idx+1} out of {len(random_seeds)} WITH RANDOM SEED {random_seed}...')
        main(config, n_gpus_literal=n_gpus)
        print('===============================================================')



