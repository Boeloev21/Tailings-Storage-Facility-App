import streamlit as st
import tempfile
import os
import zipfile
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt
from matplotlib.path import Path

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
def load_xyz(file_path):
    df = pd.read_csv(file_path)
    return df["X"].values, df["Y"].values, df["Z"].values

def load_xy(file_path):
    df = pd.read_csv(file_path)
    return df[["X", "Y"]].values

# -------------------------
# GRID INTERPOLATION (TIN-like)
# -------------------------
def create_grid(x, y, z):
    interp = LinearNDInterpolator(list(zip(x, y)), z)

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    xi = np.arange(xmin, xmax, PIXEL_SIZE)
    yi = np.arange(ymin, ymax, PIXEL_SIZE)

    xi, yi = np.meshgrid(xi, yi)
    zi = interp(xi, yi)

    return xi, yi, zi

# -------------------------
# POLYGON MASK (no rasterio needed)
# -------------------------
def mask_grid(xi, yi, zi, polygon_xy):
    poly_path = Path(polygon_xy)

    points = np.vstack((xi.flatten(), yi.flatten())).T
    mask = poly_path.contains_points(points).reshape(xi.shape)

    zi_masked = np.where(mask, zi, np.nan)

    return zi_masked

# -------------------------
# ANALYSIS
# -------------------------
def freeboard_map(grid, water_level):
    return water_level - grid

def deposition_map(new_grid, old_grid):
    return new_grid - old_grid

# -------------------------
# PLOTTING
# -------------------------
def save_plot(data, path, title):
    plt.figure(figsize=(7, 5))
    plt.imshow(data, cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

# -------------------------
# PROCESS
# -------------------------
if st.button("Run Analysis"):

    if not (prev_file and new_file and string_file):
        st.error("Please upload all required files.")
    else:
        with tempfile.TemporaryDirectory() as tmp:

            st.info("Processing...")

            # Save uploads
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

            # Interpolate grids
            px_i, py_i, prev_grid = create_grid(px, py, pz)
            nx_i, ny_i, new_grid = create_grid(nx, ny, nz)

            # Mask with polygon
            prev_grid = mask_grid(px_i, py_i, prev_grid, string_xy)
            new_grid = mask_grid(nx_i, ny_i, new_grid, string_xy)

            # Analysis
            freeboard = freeboard_map(new_grid, water_level)
            deposition = deposition_map(new_grid, prev_grid)

            # Outputs
            free_png = os.path.join(tmp, "freeboard.png")
            dep_png = os.path.join(tmp, "deposition.png")

            save_plot(freeboard, free_png, "Freeboard Map")
            save_plot(deposition, dep_png, "Deposition Map")

            # Preview
            st.subheader("Results Preview")

            col1, col2 = st.columns(2)

            with col1:
                st.image(free_png, caption="Freeboard Map")

            with col2:
                st.image(dep_png, caption="Deposition Map")

            # ZIP outputs
            zip_path = os.path.join(tmp, "results.zip")

            with zipfile.ZipFile(zip_path, "w") as z:
                z.write(free_png, "freeboard.png")
                z.write(dep_png, "deposition.png")

            # Download
            with open(zip_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Results ZIP",
                    data=f,
                    file_name="TSF_results.zip",
                    mime="application/zip"
                )

            st.success("✅ Analysis complete!")
