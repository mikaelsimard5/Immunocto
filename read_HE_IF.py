import boto3
import os
import pandas as pd
import tifffile
import zarr

def download_from_s3(s3_client, bucket_name, object_key, local_dir):
    """Download a single file from an S3 bucket"""
    local_file_name = os.path.join(local_dir, os.path.basename(object_key))
    if not os.path.exists(local_file_name):
        s3_client.download_file(bucket_name, object_key, local_file_name)
    return local_file_name

def download_orion_data(bucket_name="lin-2023-orion-crc", region="us-east-1", prefix="data",
                        data_type="HE", local_dir='./'):
    """ Download data from ORION. If any issues arise, see the documentation: 
        https://github.com/labsyspharm/orion-crc?tab=readme-ov-file
    """
    file_extension = "ome.tif" if data_type == "HE" else "ome.tiff" # see data structure in documentation if unclear
    
    s3_client = boto3.client('s3', region_name=region)
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    filename = next((obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith(file_extension)), None)

    data_file = download_from_s3(s3_client, bucket_name, filename, local_dir) # download main data file

    if data_type == "IF":
        local_file = os.path.join(local_dir, "markers.csv")
        if not os.path.exists(local_file):
            markers_file = download_from_s3(s3_client, bucket_name, os.path.join(prefix, "markers.csv"), local_file)
        return (data_file, markers_file)

    return data_file

def read_tif(wsi_path):
    """ Read ome.tif(f) data using tifffile and zarr """
    store = tifffile.imread(wsi_path, aszarr=True)
    return zarr.open(store, mode='r')

def sort_coords(c, d):
    """Transform coordinates defined by center (c) and box size (d) to bbox limits"""
    x_start = int(c[0] - d[0] // 2)
    x_end = int(c[0] + d[0] // 2)
    y_start = int(c[1] - d[1] // 2)
    y_end = int(c[1] + d[1] // 2)   
    return x_start, x_end, y_start, y_end

def read_he_patch(arr, zoom_level=0, c=(0,0), d=(0,0)):
    """
    Reads an H&E region centred at c, of size d into a np.uint8 array of
    shape [d[0], d[1], C=3].
    """
    x_start, x_end, y_start, y_end = sort_coords(c, d)
    return arr[zoom_level][x_start:x_end, y_start:y_end, :]

def read_IF_patch(arr, zoom_level=0, c=(0,0), d=(0,0), channels_idx=[0]):
    """
    Reads an IF region centred at c, of size d, for channel indices channels_idx
    into a np.float32 array of shape [d[0], d[1], C], where C = len(channels_idx).
    """
    x_start, x_end, y_start, y_end = sort_coords(c, d)
    return arr[zoom_level][channels_idx, x_start:x_end, y_start:y_end].astype(np.float32).transpose(1, 2, 0)

if __name__ == "__main__":

    # This code provides a template to read H&E and IF data from the ORION dataset.

    # Input parameters
    c = (12549, 42168) # central location of the object in the slide
    d = (128, 128) # size of the bounding box to be read around the object
    sample = "CRC01" # samples in the ORION dataset are CRC01, ... , CRC40.
    target_markers = ["Hoechst", "CD45", "CD3e", "CD4", "CD8a", "CD20", "CD163", "CD68"] # chanels to read

    # example 1 - read H&E data
    he_path = download_orion_data(prefix=f"data/{sample}", data_type="HE") # download
    he_arr = read_tif(he_path) # size is [H, W, C]
    he_patch = read_he_patch(he_arr, c=c, d=d)

    print(f"Sucessfully read H&E data of size: {he_patch.shape}")

    # example 2 - read IF data for all channels relevant to Immunocto
    print('Downloading IF data - you have time to grab a coffee...', end = "")
    if_path, markers_file = download_orion_data(prefix=f"data/{sample}", data_type="IF") # download
    print("IF data downloaded!")
    slide_markers = pd.read_csv(markers_file)
    channels_idx = [slide_markers.index(m) for m in target_markers]
    if_arr = read_tif(he_path)
    if_patch = read_he_patch(if_arr, c=c, d=d)

    print(f"Sucessfully read IF data of size: {he_patch.shape}")


