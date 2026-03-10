import os
import rasterio
from rasterio.merge import merge
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Correct logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def open_raster_file(file_path):
    """Open a single raster file (COG) and return the dataset object."""
    try:
        logging.info(f"Opening COG file: {file_path}")
        src = rasterio.open(file_path)
        return src
    except Exception as e:
        logging.error(f"Error opening {file_path}: {e}")
        return None

def process_files_in_batches(files, batch_size):
    """Process raster files in manageable batches."""
    for i in range(0, len(files), batch_size):
        yield files[i:i + batch_size]

def create_mosaic(batch, output_path, num_workers=4):
    """Create a mosaic from a batch of COG files and save as a COG."""
    src_files_to_mosaic = []
    
    # Read files in parallel within the current batch
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(open_raster_file, cog_file): cog_file for cog_file in batch}
        
        for future in as_completed(futures):
            try:
                result = future.result()
                if result is not None:
                    src_files_to_mosaic.append(result)
            except Exception as e:
                logging.error(f"Error processing a file in the batch: {e}")

    if not src_files_to_mosaic:
        logging.error("No valid COG files could be opened for mosaicking in this batch.")
        return None

    # Ensure CRS and resolution are consistent across tiles (you may need to reproject)
    logging.info("Merging batch into a mosaic...")
    mosaic, out_trans = merge(src_files_to_mosaic, method='first')  # Ensure method='first' to keep only the first non-null pixel

    # Copy metadata from one of the source files
    out_meta = src_files_to_mosaic[0].meta.copy()

    # Update the metadata to reflect the mosaic dimensions, transform, and count
    out_meta.update({
        "driver": "COG",  # Save as COG
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "count": mosaic.shape[0],
        "compress": "LZW"  # Apply LZW compression
    })

    # Save the mosaic as a COG
    logging.info(f"Saving mosaic as COG to {output_path}")
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)
    
    return output_path

def create_mosaic_from_large_dataset(cog_directory, output_dir, final_mosaic_path, num_workers=4, batch_size=500):
    """Create a mosaic from a large dataset of COG files by processing in batches, saving all mosaics as COGs."""
    
    # Find all .tif files in the directory
    cog_files = [os.path.join(cog_directory, f) for f in os.listdir(cog_directory) if f.endswith('.tif')]

    if not cog_files:
        logging.warning(f"No COG files found in the directory: {cog_directory}")
        return

    # Determine if there will be only one batch
    single_batch = len(cog_files) <= batch_size

    if single_batch:
        # Directly create the final mosaic without intermediate steps
        logging.info("Single batch detected, directly creating the final mosaic...")
        create_mosaic(cog_files, final_mosaic_path, num_workers)
    else:
        intermediate_mosaic_paths = []
        
        # Process files in batches and create intermediate mosaics
        for i, batch in enumerate(process_files_in_batches(cog_files, batch_size)):
            batch_output_path = os.path.join(output_dir, f"intermediate_mosaic_{i+1}.tif")
            logging.info(f"Processing batch {i+1} of {len(cog_files) // batch_size + 1}")
            mosaic_path = create_mosaic(batch, batch_output_path, num_workers)
            if mosaic_path:
                intermediate_mosaic_paths.append(mosaic_path)

        # Merge all intermediate mosaics into the final mosaic
        if intermediate_mosaic_paths:
            logging.info("Merging intermediate mosaics into a final mosaic...")
            merge_intermediate_mosaics(intermediate_mosaic_paths, final_mosaic_path, num_workers)

def merge_intermediate_mosaics(intermediate_mosaic_paths, output_final_mosaic, num_workers=4):
    """Merge all intermediate mosaics into a final mosaic and save as a COG."""
    logging.info("Merging intermediate mosaics into a final mosaic...")
    src_files_to_mosaic = [rasterio.open(mosaic) for mosaic in intermediate_mosaic_paths]
    mosaic, out_trans = merge(src_files_to_mosaic, method='first')  # Ensure method='first' is used

    # Copy metadata from one of the source files
    out_meta = src_files_to_mosaic[0].meta.copy()

    # Update the metadata to reflect the mosaic dimensions, transform, and count
    out_meta.update({
        "driver": "COG",  # Save as COG
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "count": mosaic.shape[0],
        "compress": "LZW"  # Apply LZW compression
    })

    # Save the final mosaic as a COG
    logging.info(f"Saving final mosaic as COG to {output_final_mosaic}")
    with rasterio.open(output_final_mosaic, "w", **out_meta) as dest:
        dest.write(mosaic)

if __name__ == "__main__":
    cog_directory = '/data/BHCEP/Data/MeritDEM/twi_output/'  # Replace with the path to your COG files
    output_dir = '/data/BHCEP/Data/MeritDEM/Merit_TWI_Mosaic'  # Directory to store intermediate mosaics
    final_mosaic_path = '/data/BHCEP/Data/MeritDEM/Merit_TWI_Mosaic/Merit_TWI_NZ_mosaic.tif'  # Path for the final mosaic
  
    # Create a mosaic from the large dataset by processing in batches
    create_mosaic_from_large_dataset(cog_directory, output_dir, final_mosaic_path, num_workers=os.cpu_count(), batch_size=500)
