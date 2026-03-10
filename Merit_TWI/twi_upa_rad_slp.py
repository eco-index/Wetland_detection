import os
import re
import numpy as np
import rasterio
from rasterio.enums import Resampling

# Define paths to your UPA and Slope folders
upa_folder = '/data/BHCEP/Data/MeritDEM/extracted_upa/upa_s60e150'
slope_folder = '/data/BHCEP/Data/MeritDEM/calculated_slp/slope_radians'
output_folder = '/data/BHCEP/Data/MeritDEM/twi_output'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Function to extract a common identifier (e.g., tile ID) from a file name
def extract_identifier(file_name):
    # Use a regex pattern to match common identifiers in file names, e.g., sXXeYYY
    pattern = r's\d{2}e\d{3}'  # Example: matches s35e170
    match = re.search(pattern, file_name)
    return match.group(0) if match else None

# Function to calculate TWI
def calculate_twi(upa_file, slope_file, twi_file, nodata_value=-9999):
    with rasterio.open(upa_file) as upa_src:
        upa = upa_src.read(1, resampling=Resampling.bilinear)
        upa[upa == upa_src.nodata] = np.nan  # Mask NoData values
        upa_km2 = upa * 1e6  # Convert UPA to square meters from square kilometers (km² to m²)
    
    with rasterio.open(slope_file) as slope_src:
        slope = slope_src.read(1, resampling=Resampling.bilinear)
        slope[slope == slope_src.nodata] = np.nan  # Mask NoData values

    # Ensure slope values are positive and greater than zero to avoid invalid calculations
    slope[slope <= 0] = np.nan

    # Calculate TWI
    twi = np.log(upa_km2 / np.tan(slope))

    # Handle infinite and NaN values resulting from the calculation
    twi[np.isinf(twi) | np.isnan(twi)] = nodata_value

    # Save the TWI to a new GeoTIFF
    with rasterio.open(
        twi_file, 'w', driver='GTiff', height=twi.shape[0], width=twi.shape[1],
        count=1, dtype=twi.dtype, crs=upa_src.crs, transform=upa_src.transform,
        nodata=nodata_value
    ) as dst:
        dst.write(twi, 1)

# Dictionary to store UPA and slope file matches based on common identifiers
upa_files = {}
slope_files = {}

# Populate the dictionary with UPA files and their identifiers
for upa_file in os.listdir(upa_folder):
    if upa_file.endswith(".tif"):
        identifier = extract_identifier(upa_file)
        if identifier:
            upa_files[identifier] = upa_file

# Populate the dictionary with slope files and their identifiers
for slope_file in os.listdir(slope_folder):
    if slope_file.endswith(".tif"):
        identifier = extract_identifier(slope_file)
        if identifier:
            slope_files[identifier] = slope_file

# Now calculate TWI for matched UPA and slope files
for identifier in upa_files:
    if identifier in slope_files:
        upa_path = os.path.join(upa_folder, upa_files[identifier])
        slope_path = os.path.join(slope_folder, slope_files[identifier])
        twi_file = os.path.join(output_folder, f"twi_{identifier}.tif")
        
        # Calculate TWI and save
        calculate_twi(upa_path, slope_path, twi_file)
        print(f"TWI calculated for {identifier} and saved to {twi_file}")
    else:
        print(f"No matching slope file found for UPA {upa_files[identifier]}")
