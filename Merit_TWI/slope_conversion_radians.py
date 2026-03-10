import os
import numpy as np
import rasterio
from rasterio.enums import Resampling

# Paths to your slope input folder and radians output folder
input_folder = '/data/BHCEP/Data/MeritDEM/calculated_slp'
output_folder = '/data/BHCEP/Data/MeritDEM/calculated_slp/slope_radians'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Function to convert slope to radians
def convert_slope_to_radians(slope_file, output_file, nodata_value=-9999):
    with rasterio.open(slope_file) as src:
        # Read the slope data
        slope_data = src.read(1, resampling=Resampling.bilinear)
        
        # Convert the slope from rise/run to radians
        slope_radians = np.arctan(slope_data)
        
        # Set invalid values (negative) to NoData
        slope_radians[slope_radians < 0] = nodata_value
        
        # Save the radians slope data to a new GeoTIFF
        with rasterio.open(
            output_file, 'w', driver='GTiff', height=slope_radians.shape[0], 
            width=slope_radians.shape[1], count=1, dtype=slope_radians.dtype,
            crs=src.crs, transform=src.transform, nodata=nodata_value
        ) as dst:
            dst.write(slope_radians, 1)

# Iterate through all slope files in the input folder
for slope_file in os.listdir(input_folder):
    if slope_file.endswith(".tif"):  # Check for GeoTIFF files
        input_path = os.path.join(input_folder, slope_file)
        output_path = os.path.join(output_folder, f"radians_{slope_file}")
        
        # Convert slope to radians and save in the output folder
        convert_slope_to_radians(input_path, output_path)
        print(f"Slope converted to radians for {slope_file} and saved to {output_path}")
