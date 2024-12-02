import boto3
from botocore import UNSIGNED
from botocore.client import Config
import os
from typing import Tuple, List, Union
import pandas as pd
import tifffile
import zarr


def download_from_s3(s3_client: boto3.client, bucket_name: str, object_key: str, local_dir: str) -> str:
    """
    Download a single file from an S3 bucket to a local directory.

    Args:
        s3_client (boto3.client): The S3 client for interacting with AWS.
        bucket_name (str): Name of the S3 bucket.
        object_key (str): Key of the object to download.
        local_dir (str): Local directory where the file will be saved.

    Returns:
        str: Path to the downloaded file.
    """
    local_file_path = os.path.join(local_dir, os.path.basename(object_key))
    if not os.path.exists(local_file_path):
        print(f'Downloading local file {local_file_path}...', end='', flush=True)
        os.makedirs(local_dir, exist_ok=True)
        s3_client.download_file(bucket_name, object_key, local_file_path)
        print('Complete.')
    else:
        print(f'Local file {local_file_path} already exists.')
    return local_file_path


def download_orion_data(
    bucket_name: str = "lin-2023-orion-crc",
    region: str = "us-east-1",
    prefix: str = "data",
    data_type: str = "HE",
    local_dir: str = "./"
) -> Union[str, Tuple[str, str]]:
    """
    Download data from the ORION dataset stored in an S3 bucket.

    Args:
        bucket_name (str): Name of the S3 bucket.
        region (str): AWS region.
        prefix (str): Prefix path within the bucket.
        data_type (str): Type of data to download ('HE' or 'IF').
        local_dir (str): Local directory to store downloaded data.

    Returns:
        Union[str, Tuple[str, str]]: Path to the downloaded file, or tuple containing paths
                                     to the data file and markers file (for 'IF' data).
    """
    s3_client = boto3.client('s3', region_name=region, config=Config(signature_version=UNSIGNED))
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    if 'Contents' not in response:
        raise ValueError(f"No files found in bucket {bucket_name} with prefix {prefix}. Check permissions or path.")

    file_extension = "ome.tif" if data_type == "HE" else "ome.tiff"
    filename = next((obj['Key'] for obj in response['Contents'] if obj['Key'].endswith(file_extension)), None)
    if not filename:
        raise FileNotFoundError(f"No file with extension {file_extension} found in bucket {bucket_name} under prefix {prefix}.")

    data_file = download_from_s3(s3_client, bucket_name, filename, local_dir)

    if data_type == "IF":
        markers_key = os.path.join(prefix, "markers.csv")
        local_markers_file = download_from_s3(s3_client, bucket_name, markers_key, local_dir)
        return data_file, local_markers_file

    return data_file


def read_tif(wsi_path: str) -> zarr.hierarchy.Group:
    """
    Read ome.tif or ome.tiff data as a Zarr array.

    Args:
        wsi_path (str): Path to the .tif(f) file.

    Returns:
        zarr.hierarchy.Group: Zarr representation of the image.
    """
    store = tifffile.imread(wsi_path, aszarr=True)
    return zarr.open(store, mode='r')


def sort_coords(center: Tuple[int, int], size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """
    Convert center coordinates and size to bounding box limits.

    Args:
        center (Tuple[int, int]): Center coordinates (x, y).
        size (Tuple[int, int]): Size of the bounding box (width, height).

    Returns:
        Tuple[int, int, int, int]: Bounding box limits (x_start, x_end, y_start, y_end).
    """
    x_start = center[0] - size[0] // 2
    x_end = center[0] + size[0] // 2
    y_start = center[1] - size[1] // 2
    y_end = center[1] + size[1] // 2
    return x_start, x_end, y_start, y_end


def read_he_patch(arr: zarr.hierarchy.Group, zoom_level: int, center: Tuple[int, int], size: Tuple[int, int]) -> zarr.Array:
    """
    Read a region of H&E data.

    Args:
        arr (zarr.hierarchy.Group): Zarr array representing the image data.
        zoom_level (int): Zoom level to extract from.
        center (Tuple[int, int]): Center coordinates of the patch.
        size (Tuple[int, int]): Size of the patch.

    Returns:
        zarr.Array: Extracted patch as an array.
    """
    x_start, x_end, y_start, y_end = sort_coords(center, size)
    return arr[zoom_level][x_start:x_end, y_start:y_end, :]


def read_if_patch(
    arr: zarr.hierarchy.Group,
    zoom_level: int,
    center: Tuple[int, int],
    size: Tuple[int, int],
    channels_idx: List[int]
) -> zarr.Array:
    """
    Read a region of IF data.

    Args:
        arr (zarr.hierarchy.Group): Zarr array representing the image data.
        zoom_level (int): Zoom level to extract from.
        center (Tuple[int, int]): Center coordinates of the patch.
        size (Tuple[int, int]): Size of the patch.
        channels_idx (List[int]): Indices of the channels to extract.

    Returns:
        zarr.Array: Extracted patch as an array.
    """
    x_start, x_end, y_start, y_end = sort_coords(center, size)
    patch = arr[zoom_level][channels_idx, x_start:x_end, y_start:y_end].astype("float32")
    return patch.transpose(1, 2, 0)


if __name__ == "__main__":
    
    c = (12549, 42168)  # centroid of the patch - corresponds to (cx, cy).
    d = (128, 128) # patch size - adjust depending on the required context

    sample = "CRC01"
    target_markers = ["Hoechst", "CD45", "CD3e", "CD4", "CD8a", "CD20", "CD163", "CD68"]

    he_path = download_orion_data(prefix=f"data/{sample}", data_type="HE")
    he_arr = read_tif(he_path)
    he_patch = read_he_patch(he_arr, zoom_level=0, center=c, size=d)
    print(f"Successfully read H&E data of size: {he_patch.shape}")

    if_path, markers_file = download_orion_data(prefix=f"data/{sample}", data_type="IF")
    slide_markers = pd.read_csv(markers_file)
    channels_idx = [slide_markers.index[slide_markers["Marker"] == m].tolist()[0] for m in target_markers]
    if_arr = read_tif(if_path)
    if_patch = read_if_patch(if_arr, zoom_level=0, center=c, size=d, channels_idx=channels_idx)
    print(f"Successfully read IF data of size: {if_patch.shape}")
