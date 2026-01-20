# Land Surface Temperature Analysis Notebook using MODIS (MOD11A1)

## 1. Overview

This notebook performs Land Surface Temperature (LST) analysis over a user-defined Area of Interest (AOI) using the MODIS MOD11A1 dataset through Google Earth Engine (GEE).

The workflow includes:

- Data extraction from MODIS LST Daily 1 km product

- Monthly aggregation of temperature statistics (mean, min, max)

- Baseline climatology computation (2002–2020)

- Temperature anomaly analysis for recent years (2021–2024)

- Trend analysis and visualization

- Export of results to CSV and PNG

## 2. Objectives

Compute a baseline climatology (2002–2020) to represent long-term mean conditions.

Derive recent period statistics (2021–2024) for temperature monitoring.

Calculate monthly anomalies relative to the baseline.

Analyze temperature trends and visualize changes.

## 3. Dataset Description

| Property                | Details                                                                  |
| ----------------------- | ------------------------------------------------------------------------ |
| **Dataset Name**        | MODIS Land Surface Temperature and Emissivity Daily Global 1km (MOD11A1) |
| **Earth Engine ID**     | `MODIS/061/MOD11A1`                                                      |
| **Variable Used**       | `LST_Day_1km`                                                            |
| **Spatial Resolution**  | 1 km                                                                     |
| **Temporal Resolution** | Daily                                                                    |
| **Scale Factor**        | 0.02                                                                     |
| **Unit Conversion**     | ( LST(°C) = (LST_Day_1km \times 0.02) - 273.15 )                         |
| **Data Source**         | NASA LP DAAC (via Google Earth Engine)                                   |
| **Temporal Coverage**   | 2000 – present                                                           |

## 4. Study Area

`N/B for this analysis a portion of the rift valley region in Kenya was used.`

Defined by user-uploaded shapefile or CSV with coordinates.

AOI represents the geographic extent over which LST is aggregated.

All computations (mean, min, max) are spatially averaged across this AOI.

## 5. Methodology

### Step 1 — Data Loading and Preprocessing

Load MODIS/061/MOD11A1 from Earth Engine.

Select the LST_Day_1km band.

Apply scale factor and convert from Kelvin to Celsius.

Add year and month metadata to each image.

### Step 2 — Baseline Climatology (2002–2020)

Filter data between 2002-01-01 and 2020-12-31.

Compute monthly mean, minimum, and maximum LST over AOI.

Aggregate by month across all baseline years.

Output: baseline monthly climatology (12-month average).

### Step 3 — Recent Years (2021–2024)

Apply same processing for each month and year (2021–2024).

Compute monthly mean, min, and max for observed data.

### Step 4 — Temperature Anomalies

Calculate monthly anomalies as:

- Anomalyy,m​=ObservedMeany,m​−BaselineMeanm​

- Positive anomalies → warmer than normal

- Negative anomalies → cooler than normal

## Step 5 — Trend Analysis

Perform linear regression on anomalies across time (2021–2024).

Assess direction and magnitude of temperature trends.

## Step 6 — Visualization

Plot baseline and recent years on the same graph.

Plot anomalies relative to baseline (zero reference).

Export plots to PNG for reporting.

## 6. Output Variables

| Variable        | Description                            |
| --------------- | -------------------------------------- |
| `year`          | Year of observation                    |
| `month`         | Month number (1–12)                    |
| `mean_obs`      | Observed mean LST (°C)                 |
| `min_obs`       | Observed minimum LST (°C)              |
| `max_obs`       | Observed maximum LST (°C)              |
| `mean_baseline` | Long-term monthly mean (°C, 2002–2020) |
| `min_baseline`  | Long-term monthly minimum (°C)         |
| `max_baseline`  | Long-term monthly maximum (°C)         |
| `mean_anomaly`  | Deviation from baseline mean (°C)      |
| `min_anomaly`   | Deviation from baseline min (°C)       |
| `max_anomaly`   | Deviation from baseline max (°C)       |

## 7. Result and Interpretation

Baseline Climatology (2002–2020)

Typical annual LST cycle observed with lower values during June – August and peaks around February – April and October – December.

Baseline mean LST generally ranges from 20°C to 32°C, depending on AOI elevation and location.

Observed Period (2021–2024)
Year Observed Behavior
2021 Close to baseline with slight warming (March – May) and cooling (July – August).
2022 Predominantly positive anomalies, especially Feb – Apr, suggesting warmer conditions.
2023 Strongest warming observed (+1.5°C to +2.5°C anomalies in March – May).
2024 Mixed pattern near-normal early months, warming trend toward Sept – Dec.

Temperature Anomalies

- Positive anomalies dominate most months since 2021.

- Gradual warming trend visible across the 4-year analysis period.

- Warmest deviations align with dry seasons, suggesting reduced surface moisture and vegetation cooling.

## 8. Visualization Summary

Baseline vs. Observed Temperatures

Gray shaded region = baseline range (min–max).

Black line = baseline mean (2002–2020).

Colored lines = monthly means for 2021–2024.

![monthly_temperature_2021.png](/plots_yearly/monthly_temperature_2021.png)

Anomaly Plot

Zero line = baseline mean reference.

Points above zero → warmer-than-normal months.

Points below zero → cooler-than-normal months.

![monthly_temperature_anomaly_2021.png](/plots_yearly/Anomalies/Temperature_Anomalies_2021.png)

All figures are saved in the local plots/ folder as .png files.

## 9. Key Insights

The AOI exhibits a warming trend relative to the 2002–2020 climatology.

Most recent years (2022–2024) show positive LST anomalies.

Dry-season months display the strongest positive deviations.

The trend analysis confirms increasing surface temperature, consistent with broader regional climate change signals.
