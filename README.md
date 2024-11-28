# Immunocto

**Immunocto** is a comprehensive immune cell database derived from H&E stained whole slide images. 

See the publication at: https://arxiv.org/abs/2406.02618.

## Immunocto V1 Database
The **Immunocto V1** database is hosted on **Zenodo**: https://zenodo.org/uploads/11073373.

The database includes `.png` images of size 64x64 pixels (resolution of 0.325 um/pixel), extracted from H&E images, along with their corresponding binary masks obtained from SAM. Each image in the dataset follows a naming convention: `(class)_(cx)_(cy).png`. Here, `(cx, cy)` denotes the centroid coordinates of each object within the whole slide image.

## Use cases of this repository
Use the scripts in this repository for the following:

1. Extract larger context around each cell (bounding boxes larger than 64x64 pixels).
2. Access the corresponding immunofluorescence (IF) data.

## Example Code
To utilize the dataset for the above purposes, you can follow these steps to set up and run the provided code:

```bash
# Set up the virtual environment (tested with python 3.9.19)
python -m venv immunocto
source immunocto/bin/activate

# Install required packages
pip install -r requirements.txt

# Run the script
python read_HE_IF.py
```
`read_HE_IF.py` illustrates how to open a given region of interest with central coordinates `(cx)_(cy)` directly from the whole slide images, both for H&E and IF channels. 
