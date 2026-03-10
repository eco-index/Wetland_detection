import os
import re
import glob
import time
import logging
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

# ==========================
# CONFIG: set these paths
# ==========================
base_directory_1 = '/data/MF_2023_stacks/Stacks_resampled_cogs_sorted'   # can be flat or tiled
base_directory_2 = '/data/MF_Hutt_lcdb_tiles'                            # can be flat or tiled
base_directory_3 = '/data/MF_Hutt_wetland_tiles'                         # can be flat or tiled

output_directory = '/data/MF_2023_lcdb_stacks'
output_directory_stacks = os.path.join(output_directory, 'Stacks_resampled_cogs_sorted')
os.makedirs(output_directory_stacks, exist_ok=True)

log_file = os.path.join(output_directory, 'Stack_processing.log')
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================
# Helpers: discovery & grouping
# ==========================
import sys
TILE_ID_REGEX = re.compile(r'(x\d+y\d+)', re.IGNORECASE)

def has_subdirs(d: str) -> bool:
    try:
        return any(os.path.isdir(os.path.join(d, n)) for n in os.listdir(d))
    except FileNotFoundError:
        return False

def glob_tifs_ci(path: str):
    return glob.glob(os.path.join(path, '*.tif')) + glob.glob(os.path.join(path, '*.TIF'))

def extract_tile_id_from_name(name: str):
    m = TILE_ID_REGEX.search(name)
    return m.group(1) if m else None

def collect_tiles_from_dir(base_dir: str):
    """
    Returns: dict[tile_id] -> list[file_paths]
    For tiled dirs: uses subfolder name as tile_id (e.g., x289y541).
    For flat dirs: groups files by tile_id parsed from filename.
    """
    mapping = {}
    if not os.path.isdir(base_dir):
        print(f"Warning: base dir not found: {base_dir}")
        return mapping

    if has_subdirs(base_dir):
        # Each subfolder is a tile ID
        for name in os.listdir(base_dir):
            tile_dir = os.path.join(base_dir, name)
            if os.path.isdir(tile_dir):
                tile_id = name
                files = glob_tifs_ci(tile_dir)
                if files:
                    mapping[tile_id] = sorted(files)
    else:
        # Flat: parse tile ID from filenames
        for f in glob_tifs_ci(base_dir):
            base = os.path.basename(f)
            tid = extract_tile_id_from_name(base)
            if tid:
                mapping.setdefault(tid, []).append(f)
    return mapping

# ==========================
# Band sorting & naming
# ==========================
band_name_mapping = {
    'NDVI': 'NDVI',
    'NDWI': 'NDWI',
    'resampled_masked_ndvi8b_cog': 'NDVI',
    'resampled_masked_ndwi8b_cog': 'NDWI'
}

def sort_bands(band_data_list, band_name_list):
    bands = list(zip(band_data_list, band_name_list))

    def get_sort_key(band):
        _, name = band
        name_lower = name.lower()

        # Planet composite bands priority 1–8
        for idx in range(1, 9):
            if f"cpwl_client_2024_planet_band_{idx}" in name_lower:
                return idx
            if f"composite_file_format_band_{idx}" in name_lower:
                return idx
            if f"spring23_mosaic_mf_band_{idx}" in name_lower:
                return idx

        # Known indices 100+
        order = 100
        for key in band_name_mapping.keys():
            if key.lower() in name_lower:
                return order
            order += 1

        # Other vars 200+
        if "nz_fabdem" in name_lower or "resampled_masked_fabdem8b" in name_lower or "nz_fabdem__2193" in name_lower or "lidar_mosaic_hutt" in name_lower:
            return 200
        if "twi" in name_lower or "merit_twi_nz" in name_lower:
            return 201
        if "nz_ksat" in name_lower or "resampled_masked_ksat8b" in name_lower:
            return 202
        if "band_24" in name_lower:
            return 203
        if "band_21" in name_lower:
            return 204
        if "band_27" in name_lower:
            return 205
        if "nz_hysog" in name_lower or "resampled_masked_hysog8b" in name_lower:
            return 206
        if "wetland_lcdb18_core_edge_single" in name_lower or "wetland18_edge_pairwisedisso" in name_lower:
            return 207

        return float('inf')

    return sorted(bands, key=get_sort_key)

def get_renamed_band_name(original_name: str):
    name_lower = original_name.lower()
    map8 = {
        "composite_file_format_band_1": "Coastal blue",
        "composite_file_format_band_2": "Blue",
        "composite_file_format_band_3": "Green I",
        "composite_file_format_band_4": "Green",
        "composite_file_format_band_5": "Yellow",
        "composite_file_format_band_6": "Red",
        "composite_file_format_band_7": "Red Edge",
        "composite_file_format_band_8": "Near infrared",
        "cpwl_client_2024_planet_band_1": "Coastal blue",
        "cpwl_client_2024_planet_band_2": "Blue",
        "cpwl_client_2024_planet_band_3": "Green I",
        "cpwl_client_2024_planet_band_4": "Green",
        "cpwl_client_2024_planet_band_5": "Yellow",
        "cpwl_client_2024_planet_band_6": "Red",
        "cpwl_client_2024_planet_band_7": "Red Edge",
        "cpwl_client_2024_planet_band_8": "Near infrared",
        "spring23_mosaic_mf_band_1": "Coastal blue",
        "spring23_mosaic_mf_band_2": "Blue",
        "spring23_mosaic_mf_band_3": "Green I",
        "spring23_mosaic_mf_band_4": "Green",
        "spring23_mosaic_mf_band_5": "Yellow",
        "spring23_mosaic_mf_band_6": "Red",
        "spring23_mosaic_mf_band_7": "Red Edge",
        "spring23_mosaic_mf_band_8": "Near infrared",
    }
    for key, val in map8.items():
        if key in name_lower:
            return val

    if "nz_fabdem" in name_lower or "resampled_masked_fabdem8b" in name_lower or "nz_fabdem__2193" in name_lower or "lidar_mosaic_hutt" in name_lower:
        return "DEM"
    if "twi" in name_lower or "merit_twi_nz" in name_lower:
        return "TWI"
    if "nz_ksat" in name_lower or "resampled_masked_ksat8b" in name_lower:
        return "Ksat"
    if "nz_hysog" in name_lower or "resampled_masked_hysog8b" in name_lower:
        return "HySOG"
    if "wetland_lcdb18_core_edge_single" in name_lower or "wetland18_edge_pairwisedisso" in name_lower:
        return "LCID"
    if "band_24" in name_lower:
        return "RZSM"
    if "band_21" in name_lower:
        return "PSM"
    if "band_27" in name_lower:
        return "SSM"

    for key, value in band_name_mapping.items():
        if key.lower() in name_lower:
            return value

    return original_name

# ==========================
# IO & alignment
# ==========================
def align_to_reference(file_path, ref_crs, ref_transform, ref_height, ref_width, categorical=False):
    """Read & reproject all bands to the reference grid."""
    out_bands = []
    names = []
    try:
        with rasterio.open(file_path) as src:
            resampling = Resampling.nearest if categorical else Resampling.bilinear
            src_nodata = src.nodata
            dst_dtype = src.dtypes[0]
            if np.issubdtype(np.dtype(dst_dtype), np.integer):
                dst_nodata = 0 if src_nodata is None else src_nodata
            else:
                dst_nodata = float(src_nodata) if src_nodata is not None else -9999.0

            base = os.path.basename(file_path).rsplit('.', 1)[0]
            for i in range(1, src.count + 1):
                dst = np.full((ref_height, ref_width), dst_nodata, dtype=dst_dtype)
                reproject(
                    source=rasterio.band(src, i),
                    destination=dst,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    src_nodata=src_nodata,
                    dst_nodata=dst_nodata,
                    resampling=resampling,
                    init_dest_nodata=True,
                )
                out_bands.append(dst)
                # keep original-like name for sorting/renaming detection
                names.append(f"{base}_Band_{i}")
    except Exception as e:
        logging.error(f"Error aligning {file_path}: {e}")
        print(f"Error aligning {file_path}: {e}")
    return out_bands, names

def choose_stack_nodata(dtype_name: str):
    """Pick a single dataset-level nodata for the stack."""
    if np.issubdtype(np.dtype(dtype_name), np.integer):
        return 0
    # float
    return -9999.0

# ==========================
# Core stack creation (COG)
# ==========================
def save_raster_stack(tile_id, files_1, files_2, files_3, is_flat1, is_flat2, is_flat3):
    """
    Build a stack for a tile using files from each source.
    For flat sources, final band descriptions will not include filename/tile_id.
    Writes output as COG.
    """
    start_time = time.time()
    print(f"Starting tile {tile_id}")

    # Pick a reference file (prefer dirs 2,3, then 1)
    ref_file = None
    for lst in (files_2, files_3, files_1):
        if lst:
            ref_file = lst[0]
            break
    if not ref_file:
        logging.info(f"No files for tile {tile_id}; skipping.")
        print(f"No files for {tile_id}; skipping.")
        return

    try:
        with rasterio.open(ref_file) as ref:
            ref_crs = ref.crs
            ref_transform = ref.transform
            ref_height, ref_width = ref.height, ref.width

        all_band_data = []
        all_band_names = []
        all_band_meta  = []  # track source/flat & band index

        def ingest(file_list, is_flat, src_tag, categorical):
            for fp in file_list:
                bands, names = align_to_reference(fp, ref_crs, ref_transform, ref_height, ref_width, categorical=categorical)
                for bi, (arr, name) in enumerate(zip(bands, names), start=1):
                    all_band_data.append(arr)
                    all_band_names.append(name)  # used for sorting/renaming detection
                    all_band_meta.append({'is_flat': is_flat, 'src': src_tag, 'band_idx': bi})

        ingest(files_1, is_flat1, 'S1', False)  # continuous by default
        ingest(files_2, is_flat2, 'S2', True)   # often categorical
        ingest(files_3, is_flat3, 'S3', True)   # often categorical

        if not all_band_data:
            logging.info(f"No bands collected for {tile_id}; skipping.")
            print(f"No bands collected for {tile_id}; skipping.")
            return

        # Sort while keeping meta aligned
        sorted_pairs = sort_bands(all_band_data, all_band_names)
        sorted_data = []
        sorted_names = []
        sorted_meta = []
        used = [False]*len(all_band_names)
        for data, name in sorted_pairs:
            for k, (n, u) in enumerate(zip(all_band_names, used)):
                if not u and n == name:
                    used[k] = True
                    sorted_data.append(all_band_data[k])
                    sorted_names.append(all_band_names[k])
                    sorted_meta.append(all_band_meta[k])
                    break

        if not sorted_data:
            logging.info(f"No sorted bands for {tile_id}; skipping.")
            print(f"No sorted bands for {tile_id}; skipping.")
            return

        # Stack and choose output dtype/nodata
        stack = np.stack(sorted_data, axis=0)
        out_dtype = stack.dtype.name
        out_nodata = choose_stack_nodata(out_dtype)

        stack_save_path = os.path.join(output_directory_stacks, f"{tile_id}_stackplus_cog.tif")
        if os.path.exists(stack_save_path):
            logging.info(f"Exists; skipping {stack_save_path}")
            print(f"Exists; skipping {stack_save_path}")
            return

        # COG profile (rasterio COG driver)
        profile = {
            'driver': 'COG',
            'height': stack.shape[1],
            'width': stack.shape[2],
            'count': stack.shape[0],
            'dtype': out_dtype,
            'crs': ref_crs,
            'transform': ref_transform,
            'compress': 'DEFLATE',
            'blocksize': 256,
            'overview_resampling': Resampling.nearest,  # safe for mixed/categorical
            'BIGTIFF': 'IF_SAFER',
            'nodata': out_nodata,
        }

        with rasterio.open(stack_save_path, 'w', **profile) as dst:
            for idx in range(stack.shape[0]):
                dst.write(stack[idx], idx + 1)
                orig_name = sorted_names[idx]
                meta = sorted_meta[idx]
                # Pretty name by mapping
                pretty = get_renamed_band_name(orig_name)
                if meta['is_flat']:
                    # For flat sources, strip filename/tile_id unless we mapped to a semantic name
                    if pretty == orig_name:
                        pretty = f"{meta['src']}_Band_{meta['band_idx']}"
                dst.set_band_description(idx + 1, pretty)

        logging.info(f"Saved COG stack: {stack_save_path}")
        print(f"Saved COG stack: {stack_save_path}")

    except Exception as e:
        logging.error(f"Error stacking {tile_id}: {e}")
        print(f"Error stacking {tile_id}: {e}")
    finally:
        try:
            del all_band_data, all_band_names, sorted_data, sorted_names, sorted_meta, stack
        except Exception:
            pass
        gc.collect()
        print(f"Tile {tile_id} done in {time.time() - start_time:.2f}s")

# ==========================
# Work planning
# ==========================
def build_work_items(dir1, dir2, dir3):
    map1 = collect_tiles_from_dir(dir1)
    map2 = collect_tiles_from_dir(dir2)
    map3 = collect_tiles_from_dir(dir3)

    ids1 = set(map1.keys())
    ids2 = set(map2.keys())
    ids3 = set(map3.keys())

    # prefer 3-way intersection; fallback to 2&3
    tile_ids = sorted(ids1 & ids2 & ids3)
    if not tile_ids:
        tile_ids = sorted(ids2 & ids3)
        print(f"No 3-way intersection; using 2&3 intersection with size {len(tile_ids)}")

    is_flat1 = not has_subdirs(dir1)
    is_flat2 = not has_subdirs(dir2)
    is_flat3 = not has_subdirs(dir3)

    print(f"Tiles: dir1={len(ids1)} dir2={len(ids2)} dir3={len(ids3)} -> queued={len(tile_ids)}")
    print(f"Layout: dir1 flat={is_flat1}, dir2 flat={is_flat2}, dir3 flat={is_flat3}")

    work = []
    for tid in tile_ids:
        files1 = map1.get(tid, [])
        files2 = map2.get(tid, [])
        files3 = map3.get(tid, [])
        if not (files2 and files3):
            continue
        work.append((tid, files1, files2, files3, is_flat1, is_flat2, is_flat3))
    return work

def process_work_in_batches(work_items, batch_size=100, max_workers=24):
    total = len(work_items)
    if total == 0:
        print("No tiles to process.")
        return
    for i in range(0, total, batch_size):
        batch = work_items[i:i + batch_size]
        print(f"Batch {i//batch_size + 1}/{(total + batch_size - 1)//batch_size} (size {len(batch)})")
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(save_raster_stack, tid, f1, f2, f3, fl1, fl2, fl3)
                       for (tid, f1, f2, f3, fl1, fl2, fl3) in batch]
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    logging.error(f"Error in batch item: {e}")
                    print(f"Error in batch item: {e}")

# ==========================
# Main
# ==========================
if __name__ == "__main__":
    work = build_work_items(base_directory_1, base_directory_2, base_directory_3)
    print(f"Total tiles queued: {len(work)}")
    process_work_in_batches(work, batch_size=100, max_workers=24)
    logging.info("Raster stacks creation complete.")
    print("Raster stacks creation complete.")
