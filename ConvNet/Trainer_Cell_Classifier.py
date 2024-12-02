# Point of this code:
import argparse
import toml
import torch


from Utils.parser import parse_arguments


def load_config(config_file):
    return toml.load(config_file)


def main(config_file=None, n_gpus=1, gpu_devices=0):
    config = load_config(config_file)
    config['ADVANCEDMODEL']['n_gpus'] = n_gpus

    print("********************************************************************************************************")
    print(f"Available GPUs: {torch.cuda.device_count()}, {config['ADVANCEDMODEL']['n_gpus']} GPUs are used for training.")

    tile_dataset = get_tile_dataset(config, subset = {'prob_tissue_type_tumour': config['BASEMODEL']['Prob_Tumour_Tresh']})
    tile_dataset = tile_dataset.loc[:, ["coords_x", "coords_y", "wsi_path", config['DATA']['Label'], "id"]] # keep only useful info
    config['DATA']['N_Classes'] = len(tile_dataset[config['DATA']['Label']].unique())

    # Print some stats
    counter_diagnosis = tile_dataset.groupby('wsi_path').head(n=1)
    print('Number of WSI per diagnosis:')
    print(counter_diagnosis['diagnosis'].value_counts())
    print(f'Number of unique WSI in tile_dataset is {tile_dataset["wsi_path"].nunique()} after merge.')
    print(f"There are {config['DATA']['N_Classes']} classes in the training dataset.")
    print("********************************************************************************************************")





if __name__ == "__main__":

    config_file, n_gpus, gpu_devices = parse_arguments()
    main(config_file, n_gpus=n_gpus, gpu_devices=gpu_devices)

