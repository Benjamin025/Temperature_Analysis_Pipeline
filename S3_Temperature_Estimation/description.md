# Sentinel-3 SLSTR LST Processing Notebook

# Jupyter-friendly Python script cells. Paste into notebook cells or save as .py and run in Jupyter.

"""
Notebook purpose:

- Download Sentinel-3 SLSTR Level-2 LST products from Copernicus Data Space API
- Preprocess (QA mask, Kelvin->C)
- Compute monthly mean/min/max for baseline (2016-2020) and analysis (2021-2024)
- Compute monthly anomalies relative to baseline
- Produce and save plots and CSV outputs

Notes:

- Requires a Copernicus Data Space account (https://dataspace.copernicus.eu)
- Tested for a local Jupyter environment
  """

# %% [markdown]

# 0. Install dependencies (run once)

# %%

# Run in a notebook cell

!pip install copernicus-dataspace-client xarray rioxarray netCDF4 shapely pandas matplotlib dask[complete]

# %% [markdown]

# 1. Imports and config

# %%

import os
from copernicus_dataspace_client import CDSEClient
import xarray as xr
import rioxarray
import numpy as np
import pandas as pd
from shapely.geometry import box, mapping
import matplotlib.pyplot as plt
from datetime import datetime

# Create folders

os.makedirs('data/sentinel3', exist_ok=True)
os.makedirs('outputs', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# %% [markdown]

# 2. User configuration

# - Set AOI: either a bounding box or path to a GeoJSON

# - Set baseline and analysis periods

# %%

# Example AOI bounding box (min_lon, min_lat, max_lon, max_lat)

# Replace with your AOI. Example covers part of East Africa.

aoi_bbox = (-7.0, -5.0, 42.0, 16.0)

# Alternative: use a GeoJSON file path

aoi_geojson_path = None # 'my_aoi.geojson'

# Temporal windows (adjust if you prefer)

baseline_start = '2016-01-01'
baseline_end = '2020-12-31'
analysis_start = '2021-01-01'
analysis_end = '2024-12-31'

# Which collections to query (S3A and S3B are both available)

collection_names = ['S3A_SL_2_LST___', 'S3B_SL_2_LST___']

# Max products to download per query (set None to download all results)

max_products = None

# %% [markdown]

# 3. Copernicus Data Space authentication

# Put your username/password interactively or set environment variables

# %%

CDS_USERNAME = os.getenv('CDS_USERNAME')
CDS_PASSWORD = os.getenv('CDS_PASSWORD')
if not CDS_USERNAME or not CDS_PASSWORD:
print('Please input Copernicus Data Space credentials (they will not be stored).')
CDS_USERNAME = input('Username: ').strip()
CDS_PASSWORD = input('Password: ').strip()

client = CDSEClient(username=CDS_USERNAME, password=CDS_PASSWORD)
print('✅ Authenticated to Copernicus Data Space')

# %% [markdown]

# 4. Helper: build AOI geometry for queries

# %%

def get_aoi_geometry():
if aoi_geojson_path:
import json
with open(aoi_geojson_path, 'r') as f:
gj = json.load(f)
return gj
else:
minx, miny, maxx, maxy = aoi_bbox
geom = mapping(box(minx, miny, maxx, maxy))
return geom

aoi_geom = get_aoi_geometry()

# %% [markdown]

# 5. Search & download function

# This searches Sentinel-3 LST products for a date range and AOI and downloads NetCDF files.

# %%

from tqdm.notebook import tqdm

def search_and_download(collection, start_date, end_date, aoi_geom, out_dir='data/sentinel3', limit=None):
print(f'Searching {collection} from {start_date} to {end_date}...')
results = client.search(
collection=collection,
start_date=start_date + 'T00:00:00Z',
end_date=end_date + 'T23:59:59Z',
geometry=aoi_geom,
limit=limit
)
print(f'Found {len(results)} products')
downloaded = []
for r in tqdm(results):
prod_id = r.get('id') or r.get('identifier') or r.get('title')
try:
client.download(prod_id, target_dir=out_dir)
downloaded.append(prod_id)
except Exception as e:
print('Download failed for', prod_id, e)
return downloaded

# %% [markdown]

# 6. Download baseline and analysis products (example)

# NOTE: For large AOIs/time ranges, this can download many files. Use limits or smaller windows for testing.

# %%

# Example: download baseline period (2016-2020)

for col in collection_names:
search_and_download(col, baseline_start, baseline_end, aoi_geom, out_dir='data/sentinel3', limit=max_products)

# Example: download analysis period (2021-2024)

for col in collection_names:
search_and_download(col, analysis_start, analysis_end, aoi_geom, out_dir='data/sentinel3', limit=max_products)

print('✅ Downloads (or attempted downloads) completed. Check data/sentinel3 for files.')

# %% [markdown]

# 7. Reading and preprocessing Sentinel-3 SLSTR NetCDF files

# - Find all .nc files in the data folder

# - Open with xarray, convert Kelvin->C, mask by quality flag (if available)

# %%

import glob

nc_files = glob.glob('data/sentinel3/\*_/_.nc', recursive=True) + glob.glob('data/sentinel3/\*.nc')
nc_files = sorted(list(set(nc_files)))
print(f'Found {len(nc_files)} NetCDF files')

# Example reading function: adjust variable names depending on the file

def open_and_process_nc(path, aoi_geom=None):
ds = xr.open_dataset(path, decode_times=True, mask_and_scale=True) # Inspect variables to find LST and quality flags
print('File:', path)
print(ds)

    # Common variable names (may differ by product): 'l2p_flags', 'sea_surface_temperature', 's3_slstr_LST', or 'LST'
    # Inspect and choose the appropriate var name. We'll try common candidates.
    candidates = ['LST', 'lst', 'sea_surface_temperature', 'surface_temperature', 'land_surface_temperature']
    varname = None
    for c in candidates:
        if c in ds.variables:
            varname = c
            break
    if varname is None:
        # fallback: pick the first data variable not coords
        dv = [v for v in ds.data_vars.keys()]
        if dv:
            varname = dv[0]
        else:
            raise ValueError('No data variables found in ' + path)

    # select variable
    arr = ds[varname]

    # Convert to Celsius if values look like Kelvin (> 100)
    if float(arr.mean().values) > 100:
        arr = arr - 273.15

    # Attempt to mask by QA/confidence variable if available
    qa_candidates = ['confidence_level', 'l2p_flags', 'quality_level', 'quality_flag']
    qa = None
    for q in qa_candidates:
        if q in ds.variables:
            qa = ds[q]
            break

    if qa is not None:
        # Simple mask: keep qa == 0 (adjust based on product docs)
        try:
            arr = arr.where(qa == 0)
        except Exception:
            pass

    # Clip to AOI if provided (requires geo info in dataset)
    # Many SLSTR files include geolocation; rioxarray can be used to clip with a bounding box
    try:
        arr = arr.rio.write_crs('EPSG:4326', allow_override=True)
        if aoi_geom is not None:
            minx, miny, maxx, maxy = aoi_bbox
            arr = arr.rio.clip_box(minx, miny, maxx, maxy)
    except Exception:
        pass

    return arr

# %% [markdown]

# 8. Aggregate monthly statistics from processed files

# %%

# We'll build a list of xarray DataArrays with time coordinates, then concat and groupby month

processed = []
for f in nc_files:
try:
arr = open_and_process_nc(f, aoi_geom=aoi_geom) # Ensure a time dimension exists; if not, try to construct from filename or attributes
if 'time' not in arr.dims: # try to get time from global attributes (quick heuristic)
attrs = arr.attrs
if 'time_coverage_start' in attrs:
t0 = np.datetime64(attrs['time_coverage_start'])
arr = arr.expand_dims(time=[t0])
else: # skip if no time
continue
processed.append(arr)
except Exception as e:
print('Failed processing', f, e)

print('Processed arrays:', len(processed))

# Concatenate along time

if len(processed) == 0:
raise RuntimeError('No processed data arrays found. Check variable names and files.')

ds_all = xr.concat(processed, dim='time')

# Group by year and month and compute mean/min/max over spatial dims

spatial_dims = [d for d in ds_all.dims if d not in ('time',)]
print('Spatial dims detected:', spatial_dims)

monthly = ds_all.groupby('time.month').mean(dim=spatial_dims)

# monthly now contains long-term monthly means across all times found

# If you need per-month-per-year stats, group by time.year and time.month

per_year_month = ds_all.groupby([ds_all.time.dt.year, ds_all.time.dt.month]).reduce(np.nanmean)

# Convert per_year_month to a pandas DataFrame for baseline and analysis splitting

records = []
for t in per_year_month['time']:
pass # Some files may not have time coordinate mapping; alternative approach below

# Alternative: resample by month and compute stats per month per year

monthly_stats = ds_all.groupby('time.year').apply(lambda x: x.groupby('time.month').reduce(np.nanmean))

# The above grouping approaches may need adjustments depending on dataset structure

print('Monthly aggregation complete (may need checks).')

# %% [markdown]

# 9. Build baseline (2016-2020) and analysis (2021-2024) tables

# %%

# Flatten per-time statistics: easier approach using resample by 1M and bounding box mean

# Compute monthly mean over AOI for each calendar month

monthly_per_time = ds_all.mean(dim=spatial_dims).to_series()
monthly_per_time.index = pd.to_datetime(monthly_per_time.index)
monthly_df = monthly_per_time.rename('lst_c').reset_index()
monthly_df['year'] = monthly_df['time'].dt.year
monthly_df['month'] = monthly_df['time'].dt.month

# Compute monthly mean/min/max per year-month

agg = monthly_df.groupby(['year', 'month'])['lst_c'].agg(['mean','min','max']).reset_index()

# Baseline climatology: average of months across baseline years

baseline_df = agg[(agg['year'] >= 2016) & (agg['year'] <= 2020)]
baseline_clim = baseline_df.groupby('month').agg({'mean':'mean','min':'mean','max':'mean'}).reset_index()
baseline_clim = baseline_clim.rename(columns={'mean':'baseline_mean','min':'baseline_min','max':'baseline_max'})

# Analysis years

analysis_df = agg[(agg['year'] >= 2021) & (agg['year'] <= 2024)].copy()

# Merge with baseline

analysis_with_baseline = analysis_df.merge(baseline_clim, on='month', how='left')
analysis_with_baseline['mean_anomaly'] = analysis_with_baseline['mean'] - analysis_with_baseline['baseline_mean']
analysis_with_baseline['min_anomaly'] = analysis_with_baseline['min'] - analysis_with_baseline['baseline_min']
analysis_with_baseline['max_anomaly'] = analysis_with_baseline['max'] - analysis_with_baseline['baseline_max']

# Save CSVs

baseline_clim.to_csv('outputs/sentinel3_baseline_2016_2020.csv', index=False)
analysis_with_baseline.to_csv('outputs/sentinel3_analysis_2021_2024.csv', index=False)
print('✅ Saved baseline and analysis CSVs in outputs/')

# %% [markdown]

# 10. Plotting: baseline band and yearly LST (absolute) and anomalies (per year)

# %%

# Plot one figure per year (absolute vs baseline)

for year in sorted(analysis*with_baseline['year'].unique()):
dfy = analysis_with_baseline[analysis_with_baseline['year'] == year]
fig, ax = plt.subplots(figsize=(10,6))
ax.fill_between(baseline_clim['month'], baseline_clim['baseline_min'], baseline_clim['baseline_max'], color='lightgray', alpha=0.4, label='Baseline range (2016-2020)')
ax.plot(baseline_clim['month'], baseline_clim['baseline_mean'], color='k', linewidth=2, label='Baseline mean (2016-2020)')
ax.plot(dfy['month'], dfy['mean'], marker='o', label=f'{year} observed')
ax.set_xticks(range(1,13)); ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
ax.set_ylabel('LST (°C)')
ax.set_title(f'Monthly LST vs Baseline — {year}')
ax.legend()
plt.tight_layout()
outpath = f'plots/LST_vs_baseline*{year}.png'
fig.savefig(outpath, dpi=200)
plt.close(fig)
print('Saved', outpath)

# Plot anomalies per year

for year in sorted(analysis*with_baseline['year'].unique()):
dfy = analysis_with_baseline[analysis_with_baseline['year'] == year]
fig, ax = plt.subplots(figsize=(10,6))
ax.axhline(0, color='k', linestyle='--')
ax.plot(dfy['month'], dfy['mean_anomaly'], marker='o', label=f'{year} anomaly')
ax.set_xticks(range(1,13)); ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
ax.set_ylabel('LST anomaly (°C)')
ax.set_title(f'Monthly LST Anomalies — {year} vs 2016-2020 baseline')
ax.legend()
plt.tight_layout()
outpath = f'plots/LST_anomalies*{year}.png'
fig.savefig(outpath, dpi=200)
plt.close(fig)
print('Saved', outpath)

print('✅ All plots saved in ./plots/')

# %% [markdown]

# 11. Next steps and notes

# - QA flags and exact variable names differ by product; inspect the printed dataset layout and adjust var names in open_and_process_nc

# - For large AOIs and long periods, consider subsetting on the server side or processing in chunks

# - Optionally upload processed NetCDF/GeoTIFF to Earth Engine for spatial visualization

print('Notebook cells complete. Review outputs folder and plots.')
