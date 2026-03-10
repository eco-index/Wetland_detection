import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import shutil
from tqdm import tqdm

base_dir = '/mnt/zfs_slow_2/Data/Planet'

planet_extents = pd.read_csv(os.path.join(base_dir, 'planet_boundaries.csv'))
grids = gpd.read_file('Waikato 2025/30m_grid_waikato_test.shp')
grids = grids.to_crs('epsg:32760')

images = []

for i, row in planet_extents.iterrows(): #loop through planet image bounds
	for j, box in grids.iterrows():
		#print(box['geometry'].bounds) #left, bottom,right, top
		#check for lack of overlap
		if box['geometry'].bounds[2] < row['left'] or row['right'] < box['geometry'].bounds[0]:
			continue
		if box['geometry'].bounds[1] > row['top'] or row['bottom'] > box['geometry'].bounds[3]:
			continue

		images.append(row['file_name'])

images = list(set(images))

for img in tqdm(images):
	#print(img)
	shutil.copyfile(os.path.join(base_dir, '202504/0000', img), os.path.join(base_dir, '202504/Waikato_2025', img))