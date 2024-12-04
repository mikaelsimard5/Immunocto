import io
import json
import os
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import seaborn as sn
import torch
import torch.nn as nn
import torchvision
from torch.nn.functional import softmax
from torchmetrics.classification import ROC
from torchmetrics.functional import accuracy, f1_score, precision, recall, confusion_matrix
from torchvision import models, transforms
from skimage import measure

# Unified model for cell classification - works with mitotic figures, lymphocites, etc.

class SAM_ConvNet(L.LightningModule):
    def __init__(self, config, LabelEncoder=None):
        super().__init__()

        # -------------------------------------------------------- Base parameters

        self.config = config
        self.LabelEncoder = LabelEncoder
        self.loss_fcn = getattr(torch.nn, self.config["BASEMODEL"]["loss_function"])()
        self.num_classes = config['DATA']['n_classes']
        self.validation_step_outputs = []
        self.test_step_outputs = []        
        self.train_logits, self.train_labels, self.train_loss = [], [], 0  # accumulators for epoch metrics
        self.val_logits, self.val_labels, self.val_loss = [], [], 0  # accumulators for epoch metrics
        self.test_logits, self.test_labels, self.test_loss = [], [], 0  # accumulators for epoch metrics   
        
        # -------------------------------------------------------- Model

        self.activation = getattr(torch.nn, "Identity")()
        backbone = getattr(torchvision.models, self.config['BASEMODEL']['backbone'])
        self.backbone = backbone(weights='DEFAULT')
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, self.num_classes)
        self.mask_encoder = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=True)

        # -------------------------------------------------------- Loss function
        
        if self.config['BASEMODEL']['loss_function'] == 'CrossEntropyLoss':
            if "weights" in self.config['DATA']:
                w = torch.tensor(self.config['DATA']['weights'], dtype=torch.float32)
            else:
                w = torch.ones(self.num_classes, dtype=torch.float32)
            self.loss_fcn = torch.nn.CrossEntropyLoss(weight=w,
                                                      label_smoothing=self.config['REGULARIZATION']['label_smoothing'])

        self.save_hyperparameters()

    def forward(self, data):
        
        # for now input has shape [B, C, H, W] - C is typically 4 (the first 3 are H&E, the last one is mask)
        x = self.backbone.conv1(data[:, 0:3, :, :]) # HE channels

        if data.shape[1] > 3: #meaning if you have more than H&E
            x = x + self.mask_encoder(data[:, -1, :, :].unsqueeze(dim=1)) # so mask keeps shape B, 1, H, W

        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = torch.mean(torch.mean(x, dim=2), dim=2)
        x = self.backbone.fc(x)
        x = self.activation(x)

        return x


    def training_step(self, train_batch, batch_idx):
        image, labels = train_batch
        logits = self.forward(image)
        loss = self.loss_fcn(logits, labels)

        self.train_logits.append(logits.detach())
        self.train_labels.append(labels.detach())
        self.train_loss += loss

        return loss


    def validation_step(self, val_batch, batch_idx):
        image, labels = val_batch
        logits = self.forward(image)
        loss = self.loss_fcn(logits, labels)

        self.val_logits.append(logits.detach())
        self.val_labels.append(labels.detach())
        self.val_loss += loss

        return loss


    def test_step(self, test_batch, batch_idx):
        image, labels = test_batch
        logits = self.forward(image)
        loss = self.loss_fcn(logits, labels)

        self.test_logits.append(logits.detach())
        self.test_labels.append(labels.detach())
        self.test_loss += loss

        return loss


    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        image, index = batch # if inference -> index is index, if not index is label.
        predictions = self.forward(image)  # Unpack logit and attention maps

        # return gathered predictions and wsi index - to sort if multi-gpu processing.
        return softmax(predictions, dim=1), index

    #def predict_step(self, batch, batch_idx, dataloader_idx=0):
    #    output = softmax(self(batch), dim=1)
    #    return output, batch['coords'], batch['id']
    #    # return self.all_gather(output), self.all_gather(batch['coords']), self.all_gather(batch['id'])

    def on_train_epoch_end(self):     

        gathered_logits = self.all_gather(torch.cat(self.train_logits, axis=0).to(self.device)) # [n_gpus, n_examples, n_classes]
        gathered_labels = self.all_gather(torch.cat(self.train_labels, axis=0).to(self.device)) # [n_gpus, n_examples]
        gathered_loss = self.all_gather(torch.stack([self.train_loss]).to(self.device)) # [n_gpus], as accumulated over steps

        # n_examples is typically batch_size * n_steps, obtained via the cat operation; can be less if the last batch
        # is such that len(dataset) modulo batch size != 0.        

        self.get_metrics_and_log_on_rank_zero(gathered_labels, gathered_logits, gathered_loss, log_prefix="train")

        self.train_logits.clear()
        self.train_labels.clear()
        self.train_loss = 0


    def on_validation_epoch_end(self):

        gathered_logits = self.all_gather(torch.cat(self.val_logits, axis=0).to(self.device)) # [n_gpus, n_examples, n_classes]
        gathered_labels = self.all_gather(torch.cat(self.val_labels, axis=0).to(self.device)) # [n_gpus, n_steps, batch_size]
        gathered_loss = self.all_gather(torch.stack([self.val_loss]).to(self.device)) # [n_gpus], as accumulated over steps 

        self.get_metrics_and_log_on_rank_zero(gathered_labels, gathered_logits, gathered_loss, log_prefix="val")

        self.val_logits.clear()
        self.val_labels.clear() 
        self.val_loss = 0

    def on_test_epoch_end(self):     

        gathered_logits = self.all_gather(torch.cat(self.test_logits, axis=0).to(self.device)) # [n_gpus, n_examples, n_classes]
        gathered_labels = self.all_gather(torch.cat(self.test_labels, axis=0).to(self.device)) # [n_gpus, n_examples]
        gathered_loss = self.all_gather(torch.stack([self.test_loss]).to(self.device)) # [n_gpus], as accumulated over steps/batch

        self.get_metrics_and_log_on_rank_zero(gathered_labels, gathered_logits, gathered_loss, log_prefix="test")

        self.test_logits.clear()
        self.test_labels.clear()             
        self.test_loss = 0     


    def get_metrics_and_log_on_rank_zero(self, gathered_labels, gathered_logits, gathered_loss, log_prefix):
        # gathered_labels: labels gathered on all devices following self.all_gather(), of size [n_gpus, n_examples]
        # gathered_logits: logits gathered on all devices following self.all_gather(), of size [n_gpus, n_examples, n_classes]
        # gathered_loss: loss gathered on all devices following self.all_gather(), of size [n_gpus,], as the loss is summed over steps/batches.
        # log_prefix: prefix for logging, typically "train", "val" or "test".

        average_loss = torch.sum(gathered_loss)/torch.numel(gathered_labels)  # sum over gpus, and divide over all examples to get the mean.
        self.log(f'{log_prefix}_loss_epoch', average_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, rank_zero_only=True)

        # Compute only on rank 0
        if self.trainer.is_global_zero:
            # Gather n_gpus/n_steps/batch_size together
            gathered_labels = gathered_labels.view(-1) 
            gathered_probs = softmax(gathered_logits.view(-1, gathered_logits.shape[-1]), dim=1) # collapse all dimensions but the last one (n_classes) together
            gathered_preds = torch.argmax(gathered_probs, dim=1)

            metrics = self.calculate_metrics(gathered_preds, gathered_labels, gathered_probs, log_prefix=log_prefix, log_suffix='epoch')
            self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, rank_zero_only=True)

            if log_prefix=="val" or log_prefix=="test":
                self.generate_confusion_matrix(gathered_preds, gathered_labels, log_prefix)


    def generate_confusion_matrix(self, preds, labels, log_prefix):
        # preds : predicted labels tensor of size (N_examples,)
        # labels: label tensor of size (N_examples,)
        # log_prefix: prefix for logging, typically "val" or "test".

        tb = self.logger.experiment
        le_name_mapping = dict(zip(self.LabelEncoder.classes_, self.LabelEncoder.transform(self.LabelEncoder.classes_)))
        computed_confusion = confusion_matrix(preds, labels, task="multiclass", num_classes=self.config["DATA"]["n_classes"]).detach().cpu().numpy().astype(int)
        df_cm = pd.DataFrame(computed_confusion, index=le_name_mapping.values(), columns=le_name_mapping.values())

        fig, ax = plt.subplots(figsize=(17, 12))
        fig.subplots_adjust(left=0.05, right=.65)
        sn.set(font_scale=1.3)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d', ax=ax, xticklabels=le_name_mapping.values(), yticklabels=le_name_mapping.keys())
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg', bbox_inches='tight')
        buf.seek(0)
        im = Image.open(buf)
        im = transforms.ToTensor()(im)
        tb.add_image(f"{log_prefix}_confusion_matrix", im, global_step=self.current_epoch)
        plt.close()


    def calculate_metrics(self, preds, labels, probs, log_prefix='', log_suffix=''):
        # preds : predicted labels tensor of size (N_examples,)
        # labels: label tensor of size (N_examples,)
        # probs : probability tensor of size (N_examples, N_classes)
        # log_prefix/log_suffix: optionally append names for the metrics towards logging, i.e. for train/val/test or step/epoch

        metrics = {}
        metrics['accuracy_macro'] = accuracy(preds, labels, task="multiclass", average="macro", num_classes=self.num_classes)
        metrics['accuracy_micro'] = accuracy(preds, labels, task="multiclass", average="micro", num_classes=self.num_classes)
        metrics['precision'] = precision(preds, labels, task="multiclass", average="macro", num_classes=self.num_classes)
        metrics['recall'] = recall(preds, labels, task="multiclass", average="macro", num_classes=self.num_classes)
        metrics['f1'] = f1_score(preds, labels, task="multiclass", average="macro", num_classes=self.num_classes)

        # Update name
        log_prefix = log_prefix + '_' if log_prefix != '' else log_prefix
        log_suffix = '_' + log_suffix if log_suffix != '' else log_suffix
        metrics = {log_prefix + k + log_suffix: v for k, v in metrics.items()}

        return metrics


    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.config['OPTIMIZER']['algorithm'])
        optimizer = optimizer(self.parameters(),
                              lr=self.config["OPTIMIZER"]["lr"],
                              eps=self.config["OPTIMIZER"]["eps"],
                              betas=(0.9, 0.999),
                              weight_decay=self.config['REGULARIZATION']['weight_decay'])

        if self.config['SCHEDULER']['type'] == 'stepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config["SCHEDULER"]["lin_step_size"],
                                                        gamma=self.config["SCHEDULER"]["lin_gamma"])
        elif self.config['SCHEDULER']['type'] == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   T_max=self.config['ADVANCEDMODEL']['max_epochs'])
        else:
            raise ValueError('Unknown type of scheduler specified in configuration file.')

        return [optimizer], [scheduler]


    def on_save_checkpoint(self, checkpoint):
        def convert_tensors(obj):
            if isinstance(obj, dict):
                return {key: convert_tensors(value) for key, value in obj.items()}
            elif isinstance(obj, torch.Tensor):
                return obj.tolist()
            else:
                return obj

        serializable_config = convert_tensors(self.config)
        checkpoint['config'] = json.dumps(serializable_config, sort_keys=True, indent=4)

        # Save the entire LabelEncoder object.
        if self.LabelEncoder:
            checkpoint['LabelEncoder'] = pickle.dumps(self.LabelEncoder)


    @classmethod
    def read_config_from_checkpoint(cls, checkpoint_path):
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        config_str = checkpoint.get('config', '{}')
        config = json.loads(config_str)
        return config


    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, *args, **kwargs):

        def convert_lists(obj):
            if isinstance(obj, dict):
                return {key: convert_lists(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return torch.tensor(obj)
            else:
                return obj

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

        # Extract and parse the config and LabelEncoder from the checkpoint
        config_str = checkpoint.get('config', '{}')  # Use an empty dict string as default.
        config = json.loads(config_str)

        if 'LabelEncoder' in checkpoint:
            LabelEncoder = pickle.loads(checkpoint['LabelEncoder'])

        # Create the model with the extracted config and LabelEncoder
        model = cls(config, LabelEncoder=LabelEncoder, *args, **kwargs)

        # Load the state dict
        model.load_state_dict(checkpoint['state_dict'])

        return model    
