# Immunocto

## Description and manuscript
**Immunocto** is a comprehensive database of 2,828,818 immune cells derived from H&E stained whole slide images from a colorectal cancer cohort. The database contains 952,929 CD4+ T cells, 467,395 CD8+ T cells, 480,900 CD20+ B cells and 381,594 CD68 or CD163+ macrophages. It also includes 4,565,636 other cells/objects. See the publication at: https://arxiv.org/abs/2406.02618 for more details.

The **Immunocto V1** database is hosted on **Zenodo**: https://zenodo.org/uploads/11073373.

The database hosted on Zenodo includes `.png` images of size 64x64 pixels (resolution of 0.325 um/pixel), extracted from H&E digital pathology whole slide images, along with their corresponding binary masks obtained from the Segment Anything Model (SAM). Each image in the dataset follows a naming convention: `(class)_(cx)_(cy).png`. Here, `(cx, cy)` denotes the centroid coordinates of each object within the whole slide image.

This repository contains exemplar code to do the following:
1) **Extract additional information from the database**. 
2) **Reproduce a subset of the results in the manuscript**.

## Setting up 

The virtual environment to run the scripts in this repository can be set up as follows:
```bash
# Set up the virtual environment (tested with python 3.9.19)
python -m venv immunocto
source immunocto/bin/activate

# Install required packages
pip install -r requirements.txt
```

## 1. Extract additional information from the database

`./Image_Reader/read_HE_IF.py` shows how to extract additional information on the database. This includes (1) getting the registered immunofluorescence data to H&E, and (2) producing larger context images around each cells. For instance, images of size 256x256 around each cell can be extracted. 
	
More specifically, `./Image_Reader/read_HE_IF.py` illustrates how to open a given region of interest with central coordinates `(cx)_(cy)` and arbitrary patch size directly from the whole slide images, both for H&E and IF channels. 

## 2. Reproduce a subset of the results in the manuscript

We provide code to reproduce a subset of the results provided in the manuscript; generalising to other architectures/databases is trivial from this point. 

For instance, to train the **SAM + ConvNet classifier** on the **Lizard data** and test it on the **Immunocto**, **SegPath** and **Lizard** test datasets to obtain the results of table IV (recall on lympocyte detection), follow the next steps:

### Download the trained models and data

Data and trained models can be accessed here:

https://drive.google.com/drive/folders/1LQVMLhg4g4nzzOMvt4XEd6JW1zjV3E5X?usp=share_link

Please open an issue if the link is not accessible anymore.

The google drive folder holds the sub-folder **trained_models**, which contains the SAM + ConvNet ensemble classifier trained on Lizard (5 models for 5 folds). There is also a zipped folder **data.zip**. After downloading (and unzipping) the models and data, they should be moved to the main repository's folder such that the arborescence is as follows:

```bash
├── Immunocto
│   ├── ConvNet
│   ├── Image_Reader
│   ├── data
│   ├── trained_models
```

### Run inference on test sets and obtain performance metrics
Inference on the three test sets with the trained models can be ran with the following commands:

```bash
python ConvNet/Infer_Cell_Classifier.py --config ConvNet/Configs/test/infer_Lizard_SAM_ConvNet_on_Immunocto.ini --gpus 1

python ConvNet/Infer_Cell_Classifier.py --config ConvNet/Configs/test/infer_Lizard_SAM_ConvNet_on_SegPath.ini --gpus 1

python ConvNet/Infer_Cell_Classifier.py --config ConvNet/Configs/test/infer_Lizard_SAM_ConvNet_on_Lizard.ini --gpus 1
```

The configuration files use a batch size of 512, which can be modified depending on your GPU's VRAM. The code also works with >1 GPUs, although inference is fast (~1 minute on a NVIDIA RTX 3090). The inference script will generate .csv files in ./Analysis/inference_results/, from which the final lymphocyte recall numbers can be obtained by running

```bash
python Analysis/Lymphocyte_Recall_Analysis.py
```
### Re-train models
If desired, the provided models can be re-trained as follows:

```bash
python ConvNet/Trainer_Cell_Classifier.py --config ConvNet/Configs/train/trainer_Lizard_SAM_ConvNet_fold1.ini --gpus 1
```
The above is for fold 1; adjust the configuration file path to `trainer_Lizard_SAM_ConvNet_fold{k}.ini` for the $k^{th}$ fold. 
