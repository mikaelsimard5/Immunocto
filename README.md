# Immunocto

**Immunocto** is a comprehensive immune cell database derived from H&E stained whole slide images.

## ImmunoctoV1 Database
The **ImmunoctoV1** database, hosted on **Zenodo**, comprises `.png` images of size 64x64 pixels, extracted from H&E images, along with their corresponding binary masks sourced from SAM. Each image in the dataset follows a naming convention: `(class)_(cx)_(cy).png`. Here, `(cx, cy)` denotes the centroid coordinates of each object within the whole slide image.

## Usage Scenarios
This dataset can be particularly useful for researchers who:

1. Need to extract a larger context around each cell (bounding boxes larger than 64x64 pixels).
2. Wish to access corresponding immunofluorescence (IF) data.

## Example Code
To utilize the dataset for the above purposes, you can follow these steps to set up and run the provided code:

```bash
# Set up the virtual environment
virtualenv immunocto
source ./immunocto/bin/activate

# Install required packages
pip install -r requirements.txt

# Run the script
python read_HE_IF.py
