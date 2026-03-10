import os
import rasterio
import numpy as np
import gc
from concurrent.futures import ProcessPoolExecutor

# Set NumPy to ignore warnings for divide-by-zero and invalid operations
np.seterr(divide="ignore", invalid="ignore")

# Define a NoData value for invalid results
NODATA_VALUE = -9999

# Normalize bands to the range [0, 1] based on the data type
def normalize_band(band):
    if band.dtype == 'uint16':  # Assuming 16-bit unsigned integer data
        return band / 65535.0
    elif band.dtype == 'uint8':  # Assuming 8-bit unsigned integer data
        return band / 255.0
    else:
        # If data is already in float format, no need to normalize
        return band

# Function to compute NDVI and NDWI using the band mappings and handle stability
def compute_indices(red_band, nir_band, green_band):
    # Compute NDVI (using Red and NIR)
    ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-10)
    
    # Clamp NDVI values to the range [-1, 1]
    ndvi = np.clip(ndvi, -1, 1)
    
    # Compute NDWI (using Green and NIR)
    ndwi = (green_band - nir_band) / (green_band + nir_band + 1e-10)
    
    # Clamp NDWI values to the range [-1, 1]
    ndwi = np.clip(ndwi, -1, 1)
    
    # Post-process to handle NaN and Inf values
    ndvi = np.nan_to_num(ndvi, nan=NODATA_VALUE, posinf=NODATA_VALUE, neginf=NODATA_VALUE)
    ndwi = np.nan_to_num(ndwi, nan=NODATA_VALUE, posinf=NODATA_VALUE, neginf=NODATA_VALUE)
    
    return ndvi, ndwi

# Function to find all target raster files (ending with '_2193.tif') in the root directory
def find_raster_files(root_folder):
    raster_files = []
    for subdir, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.tif'):  # Target files ending with '_2021.tif' ##CHANGE in the code, not the comment 
                raster_files.append(os.path.join(subdir, file))
    return raster_files

# Function to process the specific file using your band mapping
def process_single_raster(file_path):
    try:
        with rasterio.open(file_path) as src:
            # Use the correct bands according to the mapping provided
            green = src.read(4)         # Band 4 (Green)
            red = src.read(6)           # Band 6 (Red)
            nir = src.read(8)           # Band 8 (Near-infrared)

            # Normalize the bands (if needed)
            green = normalize_band(green)
            red = normalize_band(red)
            nir = normalize_band(nir)

            # Compute NDVI (using Red and NIR) and NDWI (using Green and NIR)
            ndvi, ndwi = compute_indices(red, nir, green)

            # Prepare output file paths as 'NDVI.tif' and 'NDWI.tif' in the corresponding folder
            folder_path, file_name = os.path.split(file_path)
            ndvi_output_path = os.path.join(folder_path, 'NDVI.tif')
            ndwi_output_path = os.path.join(folder_path, 'NDWI.tif')

            # Use the metadata from the source raster to create new rasters (for COGs)
            ndvi_meta = src.meta.copy()
            ndwi_meta = src.meta.copy()

            # Update metadata for single-band output and configure COG options
            ndvi_meta.update({
                'driver': 'COG',  # Set the driver to COG for Cloud-Optimized GeoTIFF
                'dtype': 'float32', 
                'compress': 'LZW',  # Use LZW compression for efficient storage
                'nodata': NODATA_VALUE,
                'count': 1,         # Single-band output
                'blockxsize': 256,   # Typical tiling size
                'blockysize': 256,
                'tiled': True        # Ensure tiling is enabled for COG
            })

            ndwi_meta.update({
                'driver': 'COG',
                'dtype': 'float32',
                'compress': 'LZW',
                'nodata': NODATA_VALUE,
                'count': 1,         # Single-band output
                'blockxsize': 256,
                'blockysize': 256,
                'tiled': True
            })

            # Write NDVI COG as a single-band raster
            with rasterio.open(ndvi_output_path, 'w', **ndvi_meta) as dst:
                dst.write(ndvi.astype(np.float32), 1)

            # Write NDWI COG as a single-band raster
            with rasterio.open(ndwi_output_path, 'w', **ndwi_meta) as dst:
                dst.write(ndwi.astype(np.float32), 1)
        
        return f"Processed and saved as COG: {file_path}"

    except Exception as e:
        return f"Error processing {file_path}: {str(e)}"

# Function to process a batch of raster files
def process_raster_batch(raster_batch):
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_single_raster, raster_batch))
    return results

# Function to batch process the rasters with limited open files and garbage collection
def process_rasters_in_parallel(root_folder, max_workers=28, batch_size=100):
    raster_files = find_raster_files(root_folder)
    
    total_files = len(raster_files)
    print(f"Total raster files to process: {total_files}")
    
    for i in range(0, total_files, batch_size):
        batch_files = raster_files[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1} of {total_files // batch_size + 1}...")

        results = process_raster_batch(batch_files)

        for result in results:
            print(result)

        gc.collect()

# Main function to specify the directory path
if __name__ == '__main__':
    root_folder = '/data/MF_Hutt_Planet_tiles/Spring_23'
    #root_folder = '/data/BHCEP/Wetlands/CPWL/CPWL_Client_Planet_2020_768m_tiles_2193' 
    #root_folder = '/data/BHCEP/Wetlands/Taranaki_Wetlands/Taranaki_Planet_2023_tiled'
    #root_folder = '/data/BHCEP/Wetlands/Taranaki_Wetlands/Taranaki_AOI/Taranaki_AOI_Planet/Taranaki_AOI_2023_tiled'
    #root_folder = '/data/BHCEP/Waikato_wetlands/Swath8B/AOI_BOG_2023/AOI_BOG_Planet_2023'
    #root_folder = '/data/BHCEP/Wetlands/NZ_Merino_Wetlands_detection/MerinoByCatchments/Raglan_Merino/Raglan_Merino_768m_data_tiles_2193'
    process_rasters_in_parallel(root_folder, max_workers=28, batch_size=100)
