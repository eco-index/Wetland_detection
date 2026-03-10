import os
import glob
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.plot import show
from rasterio.windows import Window, from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio import mask
from shapely.geometry import box
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import List, Tuple, Optional
import gc

def find_geotiff_files(directory: str, pattern: str = "*.tif") -> List[str]:
    """
    Find all GeoTIFF files in a directory
    
    Args:
        directory (str): Directory path to search
        pattern (str): File pattern to match (default: "*.tif")
        
    Returns:
        List[str]: List of GeoTIFF file paths
    """
    # Support both .tif and .tiff extensions
    patterns = [pattern, pattern.replace('.tif', '.tiff')]
    
    files = []
    for pat in patterns:
        search_pattern = os.path.join(directory, pat)
        found_files = glob.glob(search_pattern)
        files.extend(found_files)
    
    # Remove duplicates and sort
    files = sorted(list(set(files)))
    
    print(f"Found {len(files)} GeoTIFF files in {directory}")
    for file in files:
        print(f"  - {os.path.basename(file)}")
    
    return files

def check_compatibility(file_paths: List[str]) -> Tuple[bool, str]:
    """
    Check if GeoTIFF files are compatible for merging
    
    Args:
        file_paths (List[str]): List of file paths to check
        
    Returns:
        Tuple[bool, str]: (is_compatible, error_message)
    """
    if not file_paths:
        return False, "No files provided"
    
    reference_crs = None
    reference_dtype = None
    reference_count = None
    
    for i, file_path in enumerate(file_paths):
        try:
            with rasterio.open(file_path) as src:
                if i == 0:
                    reference_crs = src.crs
                    reference_dtype = src.dtypes[0]
                    reference_count = src.count
                    print(f"Reference file: {os.path.basename(file_path)}")
                    print(f"  CRS: {reference_crs}")
                    print(f"  Data type: {reference_dtype}")
                    print(f"  Band count: {reference_count}")
                else:
                    if src.crs != reference_crs:
                        return False, f"CRS mismatch: {file_path} has {src.crs}, expected {reference_crs}"
                    if src.dtypes[0] != reference_dtype:
                        return False, f"Data type mismatch: {file_path} has {src.dtypes[0]}, expected {reference_dtype}"
                    if src.count != reference_count:
                        return False, f"Band count mismatch: {file_path} has {src.count}, expected {reference_count}"
        except Exception as e:
            return False, f"Error reading {file_path}: {e}"
    
    return True, "All files are compatible"

def get_mosaic_bounds_and_resolution(file_paths: List[str]) -> Tuple[Tuple[float, float, float, float], float, float, object]:
    """
    Get the bounds and resolution for the output mosaic without loading data
    
    Returns:
        Tuple: (bounds, pixel_width, pixel_height, crs)
    """
    min_left, min_bottom = float('inf'), float('inf')
    max_right, max_top = float('-inf'), float('-inf')
    
    pixel_widths = []
    pixel_heights = []
    reference_crs = None
    
    for file_path in file_paths:
        with rasterio.open(file_path) as src:
            bounds = src.bounds
            min_left = min(min_left, bounds.left)
            min_bottom = min(min_bottom, bounds.bottom)
            max_right = max(max_right, bounds.right)
            max_top = max(max_top, bounds.top)
            
            pixel_widths.append(abs(src.transform[0]))
            pixel_heights.append(abs(src.transform[4]))
            
            if reference_crs is None:
                reference_crs = src.crs
    
    # Use the finest resolution
    pixel_width = min(pixel_widths)
    pixel_height = min(pixel_heights)
    
    return (min_left, min_bottom, max_right, max_top), pixel_width, pixel_height, reference_crs

def merge_geotiffs_memory_efficient(file_paths: List[str], output_path: str, 
                                  method: str = 'first', nodata: Optional[float] = None,
                                  compress: str = 'lzw', use_bigtiff: bool = True,
                                  chunk_size: int = 2048) -> bool:
    """
    Memory-efficient merge of multiple GeoTIFF files using windowed processing
    
    Args:
        file_paths (List[str]): List of input GeoTIFF file paths
        output_path (str): Path for output merged file
        method (str): Merge method ('first', 'last', 'min', 'max', 'mean')
        nodata (Optional[float]): NoData value for output
        compress (str): Compression method
        use_bigtiff (bool): Use BigTIFF format for large files
        chunk_size (int): Size of processing chunks in pixels
        
    Returns:
        bool: Success status
    """
    try:
        print(f"Starting memory-efficient merge of {len(file_paths)} files...")
        print(f"Processing in {chunk_size}x{chunk_size} pixel chunks")
        
        # Get output bounds and resolution
        bounds, pixel_width, pixel_height, crs = get_mosaic_bounds_and_resolution(file_paths)
        min_left, min_bottom, max_right, max_top = bounds
        
        # Calculate output dimensions
        output_width = int(np.ceil((max_right - min_left) / pixel_width))
        output_height = int(np.ceil((max_top - min_bottom) / pixel_height))
        
        print(f"Output dimensions: {output_width} x {output_height}")
        print(f"Output bounds: {bounds}")
        
        # Create output transform
        output_transform = rasterio.transform.from_bounds(
            min_left, min_bottom, max_right, max_top, 
            output_width, output_height
        )
        
        # Get reference metadata
        with rasterio.open(file_paths[0]) as src:
            profile = src.profile.copy()
            if nodata is None:
                nodata = src.nodata
        
        # Update profile for output
        profile.update({
            'driver': 'GTiff',
            'height': output_height,
            'width': output_width,
            'transform': output_transform,
            'crs': crs,
            'compress': compress,
            'nodata': nodata,
            'BIGTIFF': 'YES' if use_bigtiff else 'NO',
            'TILED': 'YES',
            'BLOCKXSIZE': min(512, chunk_size),
            'BLOCKYSIZE': min(512, chunk_size)
        })
        
        # Create output file
        with rasterio.open(output_path, 'w', **profile) as dst:
            # Add metadata
            dst.update_tags(
                DESCRIPTION=f'Memory-efficient merged GeoTIFF using {method} method',
                SOURCE_FILES=f'{len(file_paths)} input files',
                PROCESSING='Merged using windowed processing',
                CHUNK_SIZE=str(chunk_size)
            )
            
            # Process in chunks
            total_chunks = 0
            processed_chunks = 0
            
            for row_start in range(0, output_height, chunk_size):
                for col_start in range(0, output_width, chunk_size):
                    total_chunks += 1
            
            print(f"Total chunks to process: {total_chunks}")
            
            for row_start in range(0, output_height, chunk_size):
                for col_start in range(0, output_width, chunk_size):
                    # Calculate chunk boundaries
                    row_end = min(row_start + chunk_size, output_height)
                    col_end = min(col_start + chunk_size, output_width)
                    
                    chunk_height = row_end - row_start
                    chunk_width = col_end - col_start
                    
                    # Create window for this chunk
                    window = Window(col_start, row_start, chunk_width, chunk_height)
                    
                    # Get geographic bounds for this chunk
                    chunk_bounds = rasterio.windows.bounds(window, output_transform)
                    
                    # Initialize output chunk
                    output_chunk = np.full((chunk_height, chunk_width), nodata, dtype=profile['dtype'])
                    valid_data_mask = np.zeros((chunk_height, chunk_width), dtype=bool)
                    values_list = [] if method in ['min', 'max', 'mean'] else None
                    
                    # Read data from all overlapping files
                    for file_path in file_paths:
                        with rasterio.open(file_path) as src:
                            # Check if this file overlaps with current chunk
                            if (src.bounds.right <= chunk_bounds[0] or 
                                src.bounds.left >= chunk_bounds[2] or
                                src.bounds.top <= chunk_bounds[1] or 
                                src.bounds.bottom >= chunk_bounds[3]):
                                continue  # No overlap
                            
                            try:
                                # Create geometry from bounds
                                chunk_geometry = box(*chunk_bounds)
                                
                                # Read the overlapping portion
                                chunk_data, chunk_transform = mask.mask(
                                    src, [chunk_geometry], 
                                    crop=True, nodata=nodata
                                )
                                
                                if chunk_data.size == 0:
                                    continue
                                
                                chunk_data = chunk_data[0]  # Get first band
                                
                                # Reproject to output grid if necessary
                                if chunk_data.shape != (chunk_height, chunk_width):
                                    reprojected = np.full((chunk_height, chunk_width), nodata, dtype=profile['dtype'])
                                    reproject(
                                        chunk_data, reprojected,
                                        src_transform=chunk_transform,
                                        dst_transform=rasterio.windows.transform(window, output_transform),
                                        src_crs=src.crs,
                                        dst_crs=crs,
                                        resampling=Resampling.nearest,
                                        src_nodata=nodata,
                                        dst_nodata=nodata
                                    )
                                    chunk_data = reprojected
                                
                                # Apply merge method
                                valid_mask = chunk_data != nodata
                                
                                if method == 'first':
                                    output_chunk[valid_mask & ~valid_data_mask] = chunk_data[valid_mask & ~valid_data_mask]
                                    valid_data_mask |= valid_mask
                                elif method == 'last':
                                    output_chunk[valid_mask] = chunk_data[valid_mask]
                                    valid_data_mask |= valid_mask
                                elif method in ['min', 'max', 'mean']:
                                    if values_list is None:
                                        values_list = []
                                    values_list.append(chunk_data)
                                    valid_data_mask |= valid_mask
                                
                            except Exception as e:
                                print(f"Warning: Error processing {file_path} for chunk {processed_chunks}: {e}")
                                continue
                    
                    # Apply aggregation methods
                    if method == 'min' and values_list:
                        stacked = np.stack(values_list)
                        with np.errstate(invalid='ignore'):
                            output_chunk = np.nanmin(np.where(stacked == nodata, np.nan, stacked), axis=0)
                        output_chunk = np.where(np.isnan(output_chunk), nodata, output_chunk)
                    elif method == 'max' and values_list:
                        stacked = np.stack(values_list)
                        with np.errstate(invalid='ignore'):
                            output_chunk = np.nanmax(np.where(stacked == nodata, np.nan, stacked), axis=0)
                        output_chunk = np.where(np.isnan(output_chunk), nodata, output_chunk)
                    elif method == 'mean' and values_list:
                        stacked = np.stack(values_list)
                        with np.errstate(invalid='ignore'):
                            output_chunk = np.nanmean(np.where(stacked == nodata, np.nan, stacked), axis=0)
                        output_chunk = np.where(np.isnan(output_chunk), nodata, output_chunk)
                    
                    # Write chunk to output
                    dst.write(output_chunk, 1, window=window)
                    
                    processed_chunks += 1
                    if processed_chunks % 100 == 0:
                        print(f"Processed {processed_chunks}/{total_chunks} chunks ({processed_chunks/total_chunks*100:.1f}%)")
                        gc.collect()  # Force garbage collection
        
        print(f"Successfully created memory-efficient merge: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error during memory-efficient merge: {e}")
        return False

def create_preview(merged_file_path: str, output_preview_path: Optional[str] = None,
                  show_plot: bool = True) -> None:
    """
    Create a preview visualization of the merged GeoTIFF
    
    Args:
        merged_file_path (str): Path to merged GeoTIFF file
        output_preview_path (Optional[str]): Path to save preview image
        show_plot (bool): Whether to display the plot
    """
    try:
        with rasterio.open(merged_file_path) as src:
            # Read first band
            data = src.read(1)
            
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Plot with proper extent
            extent = [src.bounds.left, src.bounds.right, 
                     src.bounds.bottom, src.bounds.top]
            
            # Handle different data types for visualization
            if np.issubdtype(data.dtype, np.integer):
                im = plt.imshow(data, extent=extent, cmap='terrain')
            else:
                # For float data, mask nodata values
                masked_data = np.ma.masked_equal(data, src.nodata) if src.nodata else data
                im = plt.imshow(masked_data, extent=extent, cmap='terrain')
            
            plt.colorbar(im, shrink=0.6, label='Values')
            plt.title(f'Merged GeoTIFF Preview\n{os.path.basename(merged_file_path)}')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            
            # Add statistics
            valid_data = data[data != src.nodata] if src.nodata else data
            stats_text = f'Min: {np.min(valid_data):.2f}\n'
            stats_text += f'Max: {np.max(valid_data):.2f}\n'
            stats_text += f'Mean: {np.mean(valid_data):.2f}\n'
            stats_text += f'Std: {np.std(valid_data):.2f}'
            
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                    verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            if output_preview_path:
                plt.savefig(output_preview_path, dpi=300, bbox_inches='tight')
                print(f"Preview saved to: {output_preview_path}")
            
            if show_plot:
                plt.show()
            else:
                plt.close()
                
    except Exception as e:
        print(f"Error creating preview: {e}")

def main():
    parser = argparse.ArgumentParser(description='Merge multiple GeoTIFF files into a single mosaic')
    parser.add_argument('input_directory', help='Directory containing GeoTIFF files')
    parser.add_argument('output_file', help='Path for output merged GeoTIFF file')
    parser.add_argument('--pattern', default='*.tif', 
                       help='File pattern to match (default: *.tif)')
    parser.add_argument('--method', choices=['first', 'last', 'min', 'max', 'mean'], 
                       default='first',
                       help='Merge method for overlapping areas (default: first)')
    parser.add_argument('--nodata', type=float, 
                       help='NoData value for output file')
    parser.add_argument('--compress', choices=['none', 'lzw', 'deflate', 'packbits'], 
                       default='lzw',
                       help='Compression method (default: lzw)')
    parser.add_argument('--no-bigtiff', action='store_true',
                       help='Disable BigTIFF format (not recommended for large files)')
    parser.add_argument('--chunk-size', type=int, default=2048,
                       help='Processing chunk size in pixels (default: 2048)')
    parser.add_argument('--memory-efficient', action='store_true',
                       help='Use memory-efficient processing (recommended for large datasets)')
    parser.add_argument('--preview', 
                       help='Path to save preview image')
    parser.add_argument('--no-plot', action='store_true', 
                       help='Do not display preview plot')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check file compatibility, do not merge')
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.isdir(args.input_directory):
        print(f"Error: Input directory '{args.input_directory}' not found.")
        return 1
    
    # Find GeoTIFF files
    geotiff_files = find_geotiff_files(args.input_directory, args.pattern)
    
    if not geotiff_files:
        print(f"No GeoTIFF files found in {args.input_directory} matching pattern '{args.pattern}'")
        return 1
    
    # Check compatibility
    is_compatible, message = check_compatibility(geotiff_files)
    print(f"\nCompatibility check: {message}")
    
    if not is_compatible:
        print("Files are not compatible for merging.")
        return 1
    
    if args.check_only:
        print("Compatibility check passed. Files can be merged.")
        return 0
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Merge files
    if args.memory_efficient:
        success = merge_geotiffs_memory_efficient(
            geotiff_files, 
            args.output_file,
            method=args.method,
            nodata=args.nodata,
            compress=args.compress,
            use_bigtiff=not args.no_bigtiff,
            chunk_size=args.chunk_size
        )
    else:
        # Try standard merge first, fall back to memory-efficient if it fails
        try:
            # Open all files to check total size
            total_pixels = 0
            for file_path in geotiff_files:
                with rasterio.open(file_path) as src:
                    total_pixels += src.width * src.height
            
            # Estimate memory usage (assuming float32, with overhead)
            estimated_memory_gb = (total_pixels * 4 * 2) / (1024**3)  # 2x overhead factor
            
            if estimated_memory_gb > 8:  # If estimated >8GB, use memory-efficient method
                print(f"Large dataset detected ({estimated_memory_gb:.1f}GB estimated). Using memory-efficient processing.")
                success = merge_geotiffs_memory_efficient(
                    geotiff_files, 
                    args.output_file,
                    method=args.method,
                    nodata=args.nodata,
                    compress=args.compress,
                    use_bigtiff=not args.no_bigtiff,
                    chunk_size=args.chunk_size
                )
            else:
                # Use standard rasterio merge for smaller datasets
                print("Using standard merge method for smaller dataset.")
                src_files_to_mosaic = [rasterio.open(fp) for fp in geotiff_files]
                
                mosaic, out_trans = merge(src_files_to_mosaic, method=args.method, nodata=args.nodata)
                
                out_meta = src_files_to_mosaic[0].meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_trans,
                    "compress": args.compress,
                    "nodata": args.nodata,
                    "BIGTIFF": "NO" if args.no_bigtiff else "YES",
                    "TILED": "YES",
                    "BLOCKXSIZE": 512,
                    "BLOCKYSIZE": 512
                })
                
                with rasterio.open(args.output_file, "w", **out_meta) as dest:
                    dest.write(mosaic)
                    dest.update_tags(
                        DESCRIPTION=f'Merged GeoTIFF using {args.method} method',
                        SOURCE_FILES=f'{len(geotiff_files)} input files'
                    )
                
                for src in src_files_to_mosaic:
                    src.close()
                
                success = True
                print(f"Successfully merged files to: {args.output_file}")
                
        except MemoryError:
            print("Memory error with standard merge. Falling back to memory-efficient method...")
            success = merge_geotiffs_memory_efficient(
                geotiff_files, 
                args.output_file,
                method=args.method,
                nodata=args.nodata,
                compress=args.compress,
                use_bigtiff=not args.no_bigtiff,
                chunk_size=args.chunk_size
            )
    
    if not success:
        return 1
    
    # Create preview if requested
    if args.preview or not args.no_plot:
        create_preview(
            args.output_file,
            args.preview,
            show_plot=not args.no_plot
        )
    
    print("\nMerge completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main())