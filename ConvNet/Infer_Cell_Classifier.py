import lightning as L
import numpy as np
import pandas as pd
from sklearn import preprocessing
import os
import toml
import torch
from torch.utils.data import DataLoader

from Data_loading.dataloader_digpat import get_tile_dataset, DataGenerator, digpat_datamodule
from models.HE_mask_classifiers import SAM_ConvNet
from Utils.parser import parse_arguments
from Utils.esthetics import generate_model_name
from Utils.training_prep import get_logger, get_callbacks, get_class_weights, get_transforms


def load_config(config_file):
    return toml.load(config_file)


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model    



# train_transform, val_transform = get_transforms(config) with the config of the model.

def main(config_file=None, n_gpus=1, gpu_devices=0):
    config = load_config(config_file)
    config['ADVANCEDMODEL']['n_gpus'] = n_gpus

    print("********************************************************************************************************")
    print(f"Available GPUs: {torch.cuda.device_count()}, {config['ADVANCEDMODEL']['n_gpus']} GPUs are used for inference.")
    print("********************************************************************************************************")
    # Set up trainer, configuration files

    # Load configuration files from the training models
    classifier_configs = [SAM_ConvNet.read_config_from_checkpoint(model_path) for model_path in config['CLASSIFIER']['checkpoint']]

    # Keep the first one to set up configuration file
    trainer_config = classifier_configs[0]

    num_workers = int(7 * config['ADVANCEDMODEL']['n_gpus'])

    trainer = L.Trainer(devices=config['ADVANCEDMODEL']['n_gpus'],
                        accelerator="gpu",
                        strategy="ddp",
                        logger=False,
                        precision=trainer_config['BASEMODEL']['precision'],
                        use_distributed_sampler = False,
                        benchmark=False)        

    L.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('medium')

    # Set up dataset, split accross GPUs
    tile_dataset = get_tile_dataset(config['DATA']['data_file'])
    tiles_per_gpu = len(tile_dataset) // trainer.world_size
    start_idx     = trainer.global_rank * tiles_per_gpu
    end_idx = start_idx + tiles_per_gpu if trainer.global_rank < trainer.world_size - 1 else len(tile_dataset)    
    tile_dataset = tile_dataset[start_idx:end_idx]        
    print(f"The GPU with rank {trainer.global_rank} is processing {len(tile_dataset)} cells.")    

    # Load transforms and model
    _, val_transform = get_transforms(trainer_config)

    data = DataLoader(DataGenerator(tile_dataset, config=config, transform=val_transform),
                      batch_size = config['BASEMODEL']['batch_size'],
                      num_workers = num_workers,
                      pin_memory = False,
                      shuffle = False) 

    # 
    predicted_classes_prob_all_models = list()
    for m, model_path in enumerate(config['CLASSIFIER']['checkpoint']):

        print(f"Now running inference on model {m} out of {len(config['CLASSIFIER']['checkpoint'])}...")

        model_classifier = SAM_ConvNet.load_from_checkpoint(model_path)

        model_classifier.eval()
        model_classifier = freeze_model(model_classifier)

        predictions = trainer.predict(model_classifier, data)
        probabilities = torch.cat([prob for prob, _, in predictions], dim=0).cpu()
        probabilities = probabilities.view(-1, probabilities.shape[-1])

        predicted_classes_prob_all_models.append(probabilities)        

    # Average out the ensemble model predictions
    probabilities = torch.mean(torch.stack(predicted_classes_prob_all_models), dim=0).numpy()
    prob_std = torch.std(torch.stack(predicted_classes_prob_all_models), dim=0).numpy()

    # Export classification and cell masks
    export_dataset = data.dataset.tile_dataset
    cell_names = model_classifier.LabelEncoder.inverse_transform(np.arange(probabilities.shape[1]))
    
    for i, cell_name in enumerate(cell_names): # get probs and std from the ensemble model
        export_dataset.loc[:, cell_name] = (100 * probabilities[:, i]).astype(np.int16)  # for low memory, save int %
        export_dataset.loc[:, cell_name + '_ensemble_std'] = (100 * prob_std[:, i]).astype(np.int16)  # for low memory, save int %

    export_dataset.to_csv(f'temp_dataset_rank_{trainer.global_rank}.csv', index=False)

    trainer.strategy.barrier() # synchronise saves
    torch.cuda.empty_cache()
    del export_dataset

    # Merge into a single dataframe on rank 0.
    if trainer.is_global_zero:
        
        export_filename = config['EXPORT'].get('inference_results', './inference_results/results.csv')
        os.makedirs(os.path.dirname(export_filename), exist_ok=True)

        tile_dataset_dict = {}
        for gpu_id in range(trainer.world_size):
            temp_csv_path = f'temp_dataset_rank_{gpu_id}.csv'
            dataset = pd.read_csv(temp_csv_path).reset_index(drop=True)        
            if not dataset.empty:
                tile_dataset_dict[f"{gpu_id}"] = dataset

            os.remove(temp_csv_path)  # delete temp csv file
            del dataset        


        full_df = pd.concat([value for key, value in tile_dataset_dict.items()], axis=0)
        full_df.to_csv(export_filename, index=False)
        print(f"Inference results exported to :{export_filename}.")

    trainer.strategy.barrier() # synchronise before looping to next patient if there



if __name__ == "__main__":

    config_file, n_gpus, gpu_devices = parse_arguments()
    main(config_file, n_gpus=n_gpus, gpu_devices=gpu_devices)

