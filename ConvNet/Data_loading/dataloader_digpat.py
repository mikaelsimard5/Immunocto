from lightning.pytorch import LightningDataModule
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import KFold
import random
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from Utils import ColourAugment



class DataGenerator(torch.utils.data.Dataset):

    def __init__(self, tile_dataset, config=None, transform=None, target_transform=None, norm_IF=True):

        super().__init__()

        self.config = config
        self.Target_Pixel_Size = self.config['BASEMODEL'].get('target_pixel_size', 0)
        self.Target_Patch_Size = self.config['BASEMODEL'].get('target_patch_size', 0)
        self.transform = transform
        self.target_transform = target_transform
        self.tile_dataset = tile_dataset
        self.inference = config['ADVANCEDMODEL']['inference']

    def __len__(self):
        return len(self.tile_dataset)

    def __getitem__(self, id):

        # Reads .png H&E patches and masks. Expected data input:
        # > H&E are loaded as [H, W, C] arrays of type uint8
        # > masks are loaded as [H, W, C=1] uint8 binary arrays

        patch_HE = np.asarray(Image.open(self.tile_dataset['image_path'].iloc[id])) / 255. # convert 0 to 1

        patch_mask = np.array(Image.open(self.tile_dataset['mask_path'].iloc[id]).convert('L'))
        patch_mask = patch_mask[:, :, np.newaxis].astype(np.float32)

        # convert mask to [0,1] range - otherwise if mask is only zeros, keep as is
        if np.max(patch_mask)>0:
            patch_mask = (patch_mask - np.min(patch_mask)) / (np.max(patch_mask) - np.min(patch_mask))
        

        patches = np.concatenate((patch_HE, patch_mask), axis=2).astype(np.float32)

        # It is assumed that a np array of shape (H, W, C), range 0-1 is passed to the transforms below.
        
        if self.transform:
            for t in self.transform.transforms:
                if isinstance(t, ColourAugment.ColourAugment) or isinstance(t, v2.Normalize):
                    patches[:3] = t(patches[:3]) # only transform the RGB image corresponding to H&E stain
                else:
                    patches = t(patches)

        # Uncomment if you want to export some training images, for debugging purposes.
        # self.export_example_images(patches)

        if self.inference:
            return patches, id
        else:
            label = self.tile_dataset['label'].iloc[id]
            if self.target_transform:
                label = self.target_transform(label)
            return patches, label


    def export_example_images(self, patches):

        # Taking the first 3 channels for RGB image
        P = patches.detach().cpu().numpy().astype(np.float32).transpose(1, 2, 0) # reorder to H, W, C

        # Rescale HE to be 0-255 in uint8.
        he = P[:,:,0:3]
        he = (he - np.min(he))/(np.max(he)-np.min(he))
        he = (255. * he).astype(np.uint8)
        
        msk = P[:, :, -1]    

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(he)
        axs[0].axis('off')
        axs[0].set_title('H & E')

        axs[1].imshow(msk)
        axs[1].axis('off')
        axs[1].set_title('Mask')
        
        plt.tight_layout()
        plt.savefig(f'example_image_{random.randint(10000, 99999)}.png')
        plt.close()



class digpat_datamodule(LightningDataModule):
    def __init__(self, tile_dataset, train_transform=None, val_transform=None, config=None, **kwargs):
        super().__init__()

        self.batch_size = config['BASEMODEL']['batch_size']
        self.num_workers = int(7 * config['ADVANCEDMODEL']['n_gpus'])
        self.tr_s = config['DATA']['train_size']
        self.val_s = config['DATA']['val_size']
        self.test_s = config['DATA']['test_size']
        assert self.tr_s + self.val_s + self.test_s == 1, "Size parameters must sum up to 1"
        self.unique_pts = tile_dataset['patient'].unique()        
        print(f'Number of workers for dataloader: {self.num_workers}')

        # ------------------------------------------------------------------------------------------------
        # Split train+val vs test on a per-patient basis


        # Create train+val, test
        if self.test_s > 0:
            train_val_pt, test_pt = train_test_split(self.unique_pts,
                                                     train_size=(self.val_s + self.test_s),
                                                     shuffle=True,
                                                     random_state=config['ADVANCEDMODEL']['Random_Seed'])
        else:
            train_val_pt = self.unique_pts
            test_pt = []

        # ------------------------------------------------------------------------------------------------
        # Sort train, val patients depending on folds or not

        if "train_fold" in config['DATA']:

            # Retrieve the train fold value from config
            train_fold = config['DATA']['train_fold']

            # Assert that train_fold is between 1 and 5 - kind of hard coded for 5 folds now
            assert train_fold in [1, 2, 3, 4, 5], f"Error: train_fold must be one of [1, 2, 3, 4, 5], but got {train_fold}"

            train_size = self.tr_s / (self.tr_s + self.val_s)
            val_size = 1 - train_size
            print(f"5-fold CV, fold #{config['DATA']['train_fold']}, {train_size} train / {np.round(val_size,2)} val split of remaining data ({self.test_s} used for test)")
            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            # Generate all folds directly
            folds = [
                (
                    [train_val_pt[i] for i in train_index],  # Train paths
                    [train_val_pt[i] for i in val_index]    # Validation paths
                )
                for train_index, val_index in kf.split(train_val_pt)
            ]

            # Access the specific fold
            train_pt, val_pt = folds[train_fold - 1]                       

        else:
            print(f"Did not find any fold information - proceeding with normal splitting according to seed {config['ADVANCEDMODEL']['random_seed']}.")
            train_pt, val_pt = train_test_split(train_val_pt, train_size=train_size / (train_size + val_size),
                                                shuffle=True, random_state=config['ADVANCEDMODEL']['random_seed'])


        # ------------------------------------------------------------------------------------------------
        # Split dataframe based on allocated patients for train/val/test

        tile_dataset_train = tile_dataset[tile_dataset['patient'].isin(train_pt)].reset_index()
        tile_dataset_val = tile_dataset[tile_dataset['patient'].isin(val_pt)].reset_index()
        tile_dataset_test = tile_dataset[tile_dataset['patient'].isin(test_pt)].reset_index()

        self.train_data = DataGenerator(tile_dataset_train, config=config, transform=train_transform, **kwargs)
        self.val_data = DataGenerator(tile_dataset_val, config=config, transform=val_transform, **kwargs)

        if self.test_s > 0:
            self.test_data = DataGenerator(tile_dataset_test, config=config, transform=val_transform, **kwargs)
        else:
            self.test_data = None
        

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=False,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=False,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=False,
                          shuffle=False)


def get_tile_dataset(csv_file="", valid_labels=None):

    # Assumed that the dataframe contains the columns
    # > image_path (absolute path to patch)
    # > mask_path (absolute path to binary cell mask)
    # > label (string or number)

    df = pd.read_csv(csv_file).reset_index()
    expected_columns = {"image_path", "mask_path", "label"}
    assert expected_columns.issubset(df.columns), f"file {csv_file} is missing one of image_path, mask_path, label."

    # filter by removing unwanted labels.
    if valid_labels is not None:
        df = df[df["label"].isin(valid_labels)]

    return df