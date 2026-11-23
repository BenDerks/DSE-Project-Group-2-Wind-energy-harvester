
import os
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


surface_geo_url = (
    "https://arco.datastores.ecmwf.int/cadl-arco-geo-002/"
    "arco/reanalysis_era5_single_levels/sfc/geoChunked.zarr"
)

#API key
if "CDS_API_KEY" not in os.environ:
    raise ValueError(
        "CDS_API_KEY is not set. Set it in your VS Code terminal first."
    )

ds = xr.open_zarr(
    surface_geo_url,
    consolidated=True,
    storage_options={
        "headers": {
            "Authorization": f"Bearer {os.environ['CDS_API_KEY']}"
        }
    }
)


# country

countries = gpd.read_file("ne_110m_admin_0_countries.shp")

country_border = countries[countries["ADMIN"] == "Bangladesh"]

if country_border.empty:
    raise ValueError("Bangladesh not found in the shapefile.")

border = country_border.geometry.iloc[0]

min_lon, min_lat, max_lon, max_lat = border.bounds

# Check ERA5 longitude format
uses_360 = float(ds.longitude.max()) > 180

if uses_360:
    data_min_lon = min_lon % 360
    data_max_lon = max_lon % 360
else:
    data_min_lon = min_lon
    data_max_lon = max_lon

# --------------------------------------------------
# 3. Select data from 2016 to 2025
# --------------------------------------------------

country_data = ds[
    ["u10", "v10", "u100", "v100"]
].sel(
    time=slice("2016-01-01", "2025-12-31T23:00:00"),
    latitude=slice(min_lat, max_lat),
    longitude=slice(data_min_lon, data_max_lon)
)

total_hours = country_data.sizes["time"]

print("Total hours:", total_hours)

# --------------------------------------------------
# 4. Identify land grid points
# --------------------------------------------------

land_points = []

for lat in country_data.latitude.values:
    for lon in country_data.longitude.values:

        geographic_lon = float(lon)

        if uses_360 and geographic_lon > 180:
            geographic_lon -= 360

        if border.covers(Point(geographic_lon, float(lat))):
            land_points.append((float(lat), float(lon)))

print("Land grid points:", len(land_points))

if not land_points:
    raise ValueError("No land grid points found for Bangladesh.")

# --------------------------------------------------
# 5. Calculate wind speed
# --------------------------------------------------

def get_wind_speed(point, height):

    lat, lon = land_points[point]

    u = country_data[f"u{height}"].sel(
        latitude=lat,
        longitude=lon
    )

    v = country_data[f"v{height}"].sel(
        latitude=lat,
        longitude=lon
    )

    wind_speed = np.sqrt(u**2 + v**2)

    return wind_speed


# --------------------------------------------------
# 6. Find longest consecutive period
# --------------------------------------------------

def longest_period(condition):

    longest = 0
    current = 0
    end_index = -1

    for i, value in enumerate(condition):

        if value:
            current += 1

            if current > longest:
                longest = current
                end_index = i

        else:
            current = 0

    start_index = end_index - longest + 1

    return longest, start_index, end_index



def analyse_height(height):

    longest_low = 0
    longest_high = 0

    low_location = None
    high_location = None

    high_mean = np.nan
    high_std = np.nan

    total_high_at_location = 0
    total_low_hours = 0
    total_high_hours_all_points = 0

    speed_sum = 0.0
    speed_squared_sum = 0.0
    valid_count = 0

    for point in range(len(land_points)):

        values = get_wind_speed(
            point, height
        ).compute().values

        valid_values = values[np.isfinite(values)]

        speed_sum += np.sum(valid_values)
        speed_squared_sum += np.sum(valid_values**2)
        valid_count += len(valid_values)

        # Low wind: below 3 m/s
        low_wind = values < 3

        total_low_hours += np.sum(low_wind)

        low_duration, low_start, low_end = longest_period(
            low_wind
        )

        if low_duration > longest_low:

            longest_low = low_duration
            low_location = land_points[point]

        # High wind: above 6 m/s
        high_wind = values > 6

        point_high_hours = int(np.sum(high_wind))

        total_high_hours_all_points += point_high_hours

        high_duration, high_start, high_end = longest_period(
            high_wind
        )

        if high_duration > longest_high:

            longest_high = high_duration
            high_location = land_points[point]

            period_speeds = values[
                high_start:high_end + 1
            ]

            high_mean = float(np.mean(period_speeds))
            high_std = float(np.std(period_speeds))

            total_high_at_location = point_high_hours

        print(
            f"{height} m: point {point + 1}/{len(land_points)}",
            end="\r",
            flush=True
        )

    if valid_count == 0:
        raise ValueError("No valid wind-speed observations found.")

    mean_speed = speed_sum / valid_count

    variance = (
        speed_squared_sum / valid_count
        - mean_speed**2
    )

    std_speed = np.sqrt(max(variance, 0))

    avg_low_hours = total_low_hours / len(land_points)

    low_percentage = (
        100 * total_low_hours
        / (total_hours * len(land_points))
    )

    avg_high_hours = (
        total_high_hours_all_points
        / len(land_points)
    )

    return {
        "Mean wind speed": mean_speed,
        "Std wind speed": std_speed,
        "Total hours": total_hours,
        "Average low-wind hours": avg_low_hours,
        "Low-wind percentage": low_percentage,
        "Longest consecutive low-wind hours": longest_low,
        "Low-wind location": low_location,
        "Longest consecutive high-wind hours": longest_high,
        "High-wind location": high_location,
        "Mean during longest high-wind period": high_mean,
        "Std during longest high-wind period": high_std,
        "Total high-wind hours at selected location": total_high_at_location,
        "Average high-wind hours per grid point": avg_high_hours
    }


print("\nAnalysing Bangladesh at 100 m...")
result100 = analyse_height(100)

print("\nAnalysing Bangladesh at 10 m...")
result10 = analyse_height(10)


results = []

for height, result in [
    (100, result100),
    (10, result10)
]:

    results.append({
        "Country": "Bangladesh",
        "Height (m)": height,
        **result,
        "Land grid points": len(land_points)
    })

df = pd.DataFrame(results)

df.to_excel(
    "bangladesh_wind_analysis.xlsx",
    index=False
)

print("\nAnalysis complete!")
print(df.to_string(index=False))
print("\nResults saved to bangladesh_wind_analysis.xlsx")
