import streamlit as st
import tempfile
import os
import zipfile
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.mask import mask
from shapely.geometry import Polygon
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt

# -------------------------
# CONFIG
# -------------------------
PIXEL_SIZE = 0.05
N_CLASSES = 15

st.set_page_config(page_title="TSF Analysis Tool", layout="wide")

st.title("📊 TSF Freeboard & Deposition Analysis")

# -------------------------
# FILE INPUTS
# -------------------------
prev_file = st.file_uploader("Upload Previous Survey CSV (X,Y,Z)", type="csv")
new_file = st.file_uploader("Upload New Survey CSV (X,Y,Z)", type="csv")
string_file = st.file_uploader("Upload Boundary String CSV (X,Y)", type="csv")

water_level = st.number_input("Water Level", value=1250.0)

# -------------------------
# FUNCTIONS
# -------------------------
def load_xyz(file):
    df = pd.read_csv(file)
    return df["X"].values, df["Y"].values, df["Z"].values

def load_xy(file):
    df = pd.read_csv(file)
    return df[["X", "Y"]].values

def create_raster(x, y, z, output_path):
    interp = LinearNDInterpolator(list(zip(x, y)), z)

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    xi = np.arange(xmin, xmax, PIXEL_SIZE)
    yi = np.arange(ymin, ymax, PIXEL_SIZE)
    xi, yi = np.meshgrid(xi, yi)

    zi = interp(xi, yi)

    transform = from_origin(xmin, ymax, PIXEL_SIZE, PIXEL_SIZE)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=zi.shape[0],
        width=zi.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:32735",
        transform=transform,
        nodata=np.nan
    ) as dst:
        dst.write(zi.astype("float32"), 1)

    return output_path

def clip_raster(raster_path, polygon, output_path):
    with rasterio.open(raster_path) as src:
        out_image, out_transform = mask(src, [polygon], crop=True)
        out_meta = src.meta.copy()

    out_meta.update({
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })

    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(out_image)

    return output_path

def save_raster(data, ref, output_path):
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs=ref.crs,
        transform=ref.transform,
        nodata=np.nan
    ) as dst:
        dst.write(data.astype("float32"), 1)

    return output_path

def freeboard_map(raster_path, water_level, output_path):
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        result = water_level - data
        return save_raster(result, src, output_path)

def deposition_map(new_path, old_path, output_path):
    with rasterio.open(new_path) as n, rasterio.open(old_path) as o:
        result = n.read(1) - o.read(1)
        return save_raster(result, n, output_path)

def save_plot(raster_path, output_png):
    with rasterio.open(raster_path) as src:
        data = src.read(1)

    plt.figure(figsize=(6, 4))
    plt.imshow(data, cmap="viridis")
    plt.colorbar()
    plt.title(os.path.basename(raster_path))
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()

# -------------------------
# PROCESS BUTTON
# -------------------------
if st.button("Run Analysis"):

    if not (prev_file and new_file and string_file):
        st.error("Please upload all required files.")
    else:
        with tempfile.TemporaryDirectory() as tmp:

            st.info("Processing...")

            # Save uploaded files
            prev_path = os.path.join(tmp, "prev.csv")
            new_path = os.path.join(tmp, "new.csv")
            string_path = os.path.join(tmp, "string.csv")

            with open(prev_path, "wb") as f:
                f.write(prev_file.read())
            with open(new_path, "wb") as f:
                f.write(new_file.read())
            with open(string_path, "wb") as f:
                f.write(string_file.read())

            # Load data
            px, py, pz = load_xyz(prev_path)
            nx, ny, nz = load_xyz(new_path)
            string_xy = load_xy(string_path)

            # Interpolate
            prev_raster = create_raster(px, py, pz, f"{tmp}/prev.tif")
            new_raster = create_raster(nx, ny, nz, f"{tmp}/new.tif")

            # Polygon
            polygon = Polygon(string_xy)

            # Clip
            prev_clip = clip_raster(prev_raster, polygon, f"{tmp}/prev_clip.tif")
            new_clip = clip_raster(new_raster, polygon, f"{tmp}/new_clip.tif")

            # Analysis
            freeboard = freeboard_map(new_clip, water_level, f"{tmp}/freeboard.tif")
            deposition = deposition_map(new_clip, prev_clip, f"{tmp}/deposition.tif")

            # Visuals
            free_png = f"{tmp}/freeboard.png"
            dep_png = f"{tmp}/deposition.png"

            save_plot(freeboard, free_png)
            save_plot(deposition, dep_png)

            # Show previews
            st.subheader("Results Preview")
            col1, col2 = st.columns(2)

            with col1:
                st.image(free_png, caption="Freeboard Map")

            with col2:
                st.image(dep_png, caption="Deposition Map")

            # Zip results
            zip_path = f"{tmp}/results.zip"
            with zipfile.ZipFile(zip_path, "w") as z:
                for file in os.listdir(tmp):
                    if file != "results.zip":
                        z.write(os.path.join(tmp, file), file)

            # Download button
            with open(zip_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Results ZIP",
                    data=f,
                    file_name="TSF_results.zip",
                    mime="application/zip"
                )

            st.success("✅ Analysis complete!")