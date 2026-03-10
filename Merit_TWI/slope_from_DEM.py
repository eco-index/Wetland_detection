import os
import numpy as np
import rasterio
from rasterio.enums import Resampling

# Define paths to your DEM folder and slope output folder
input_folder = '/data/BHCEP/Data/MeritDEM/extracted_elv/elv_s60e150/'
output_folder = '/data/BHCEP/Data/MeritDEM/calculated_slp'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Function to calculate slope
def calculate_slope(dem_file, slope_file):
    with rasterio.open(dem_file) as dem:
        elevation = dem.read(1, resampling=Resampling.bilinear)
        # Replace NoData values with NaN
        elevation[elevation == dem.nodata] = np.nan
        xres, yres = dem.res[0], dem.res[1]  # Pixel size (cell size in x and y direction)

    # Compute slope using finite differences and apply scaling to get slope in degrees
    dzdx = np.gradient(elevation, xres, axis=1)  # Slope in x-direction
    dzdy = np.gradient(elevation, yres, axis=0)  # Slope in y-direction
    slope = np.sqrt(dzdx**2 + dzdy**2)  # Magnitude of gradient (slope)

    # Convert slope to degrees
    slope_degrees = np.arctan(slope) * (180 / np.pi)

    # Save the slope to a new GeoTIFF
    with rasterio.open(
        slope_file, 'w', driver='GTiff', height=slope_degrees.shape[0], width=slope_degrees.shape[1],
        count=1, dtype=slope_degrees.dtype, crs=dem.crs, transform=dem.transform,
        nodata=-9999, compress='DEFLATE'
    ) as dst:
        slope_degrees[np.isnan(slope_degrees)] = -9999  # Set NoData values back to -9999
        dst.write(slope_degrees, 1)

# Iterate through all DEM tiles in the input folder
for dem_tile in os.listdir(input_folder):
    if dem_tile.endswith(".tif"):  # Check for GeoTIFF files
        dem_file = os.path.join(input_folder, dem_tile)
        slope_file = os.path.join(output_folder, f"slope_{dem_tile}")
        
        # Calculate slope and save to the output folder
        calculate_slope(dem_file, slope_file)
        print(f"Slope calculated for {dem_tile} and saved to {slope_file}")
