import tarfile
import os

def extract_tar_file(tar_path, extract_path):
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(path=extract_path)
        print(f"Extracted {tar_path} to {extract_path}")

# Paths to your .tar file
#dir_tar_path = '/data/BHCEP/Data/MeritDEM/dir_s60e150.tar'
#elv_tar_path = '/data/BHCEP/Data/MeritDEM/elv_s60e150.tar'
upstream_pixels_tar_path = '/data/BHCEP/Data/MeritDEM/upg_s60e150.tar'

# Paths to extract
#dir_extract_path = '/data/BHCEP/Data/MeritDEM/extracted_dir'
#elv_extract_path = '/data/BHCEP/Data/MeritDEM/extracted_elv'
upstream_pixels_extract_path = '/data/BHCEP/Data/MeritDEM/extracted_upg'
# Extracting the tar files
#extract_tar_file(dir_tar_path, dir_extract_path)
#extract_tar_file(elv_tar_path, elv_extract_path)
extract_tar_file(upstream_pixels_tar_path, upstream_pixels_extract_path)