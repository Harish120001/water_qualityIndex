import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import contextlib, io, zipfile, tempfile, os

import pydeck as pdk
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin

# ---------------- CONFIG ----------------
st.set_page_config(page_title="HydroGIS Pro", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>
.main {background-color: #f4f6f9;}
h1 {text-align:center; color:#0A3D62;}
.card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.1);
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- CONSTANTS ----------------
BIS_LIMITS = {"pH": 8.5, "SO4": 200, "NO3": 45, "F": 1.5, "Cl": 250,
              "TDS": 500, "Na": 100, "Ca": 75, "Mg": 30, "K": 10, "HCO3": 200}

BIS_WEIGHTS = {"pH": 4, "SO4": 5, "NO3": 5, "F": 5, "Cl": 5,
               "TDS": 5, "Na": 4, "Ca": 3, "Mg": 3, "K": 2, "HCO3": 1}

IDEAL_VALUES = {"pH": 7}
EQ_WT = {"Ca": 20.04, "Mg": 12.15, "Na": 23, "K": 39.1, "HCO3": 61, "CO3": 30}

# ---------------- TRUE IDW ----------------
def idw(x, y, z, xi, yi, power=2):
    grid = np.zeros_like(xi)
    for i in range(xi.shape[0]):
        for j in range(xi.shape[1]):
            d = np.sqrt((x - xi[i,j])**2 + (y - yi[i,j])**2)
            if np.any(d == 0):
                grid[i,j] = z[d == 0][0]
            else:
                w = 1 / d**power
                grid[i,j] = np.sum(w * z) / np.sum(w)
    return grid

# ---------------- CLASSIFICATION ----------------
def classify_array(arr):
    classified = np.zeros_like(arr)

    classified[arr <= 25] = 1
    classified[(arr > 25) & (arr <= 50)] = 2
    classified[(arr > 50) & (arr <= 75)] = 3
    classified[(arr > 75) & (arr <= 100)] = 4
    classified[arr > 100] = 5

    return classified

# ---------------- PROCESS ----------------
def process(df):
    name_col = next((c for c in df.columns if "sample" in c.lower()), df.columns[0])
    df = df.rename(columns={name_col: "Sample_ID"})

    total_w = sum(BIS_WEIGHTS.values())

    def wqi(row):
        val = 0
        for p in BIS_LIMITS:
            if pd.isna(row[p]): continue
            ideal = IDEAL_VALUES.get(p, 0)

            if p == "pH":
                qi = ((row[p] - 7)/(8.5-7))*100
            else:
                qi = ((row[p]-ideal)/(BIS_LIMITS[p]-ideal))*100

            val += qi * (BIS_WEIGHTS[p]/total_w)
        return abs(val)

    df["WQI"] = df.apply(wqi, axis=1)

    for i, ew in EQ_WT.items():
        if i in df.columns:
            df[f"{i}_meq"] = df[i]/ew

    denom = (df["Ca_meq"] + df["Mg_meq"]).replace(0,np.nan)

    df["SAR"] = df["Na_meq"]/np.sqrt(denom/2)
    k = df.get("K_meq", pd.Series(0,index=df.index))

    df["Na%"] = ((df["Na_meq"]+k)/(df["Ca_meq"]+df["Mg_meq"]+df["Na_meq"]+k))*100
    df["RSC"] = (df.get("HCO3_meq",0)+df.get("CO3_meq",0))-(df["Ca_meq"]+df["Mg_meq"])
    df["Kelly"] = df["Na_meq"]/denom
    df["MH"] = (df["Mg_meq"]/denom)*100

    return df

# ---------------- INPUT ----------------
st.sidebar.header("Input")
mode = st.sidebar.radio("Mode",["Manual","CSV"])

if mode=="Manual":
    data = {p:st.sidebar.number_input(p,float(BIS_LIMITS[p])) for p in BIS_LIMITS}
    df = pd.DataFrame({"Sample_ID":["S1"],**{k:[v] for k,v in data.items()},"CO3":[0]})
else:
    f = st.sidebar.file_uploader("Upload CSV")
    if f: df = pd.read_csv(f)
    else: st.stop()

df = process(df)

st.title("🌊 HydroGIS Pro Dashboard")

# ---------------- MULTI PARAMETER MAP ----------------
st.subheader("🗺️ Multi-Parameter IDW Mapping + Classification")

if "lat" in df.columns and "lon" in df.columns:

    param = st.selectbox("Select Parameter", df.select_dtypes(include=np.number).columns)

    if len(df) < 3:
        st.warning("Need at least 3 points")
    else:
        x,y,z = df["lon"].values, df["lat"].values, df[param].values

        gx,gy = np.meshgrid(
            np.linspace(x.min(),x.max(),100),
            np.linspace(y.min(),y.max(),100)
        )

        gz = idw(x,y,z,gx,gy)

        # -------- NORMAL MAP --------
        fig, ax = plt.subplots()
        c = ax.contourf(gx,gy,gz,levels=15)
        ax.scatter(x,y,c=z)
        plt.colorbar(c,label=param)
        ax.set_title(f"{param} IDW Map")
        st.pyplot(fig)

        # -------- CLASSIFIED MAP --------
        classified = classify_array(gz)

        fig2, ax2 = plt.subplots()
        cmap = plt.cm.get_cmap('RdYlGn_r',5)
        im = ax2.imshow(classified, cmap=cmap,
                        extent=[x.min(),x.max(),y.min(),y.max()],
                        origin='lower')

        ax2.set_title("Classified Map")
        st.pyplot(fig2)

        # -------- EXPORT GEOTIFF --------
        if st.button(f"Export {param} GeoTIFF"):
            transform = from_origin(x.min(),y.max(),
                                    (x.max()-x.min())/100,
                                    (y.max()-y.min())/100)

            with tempfile.NamedTemporaryFile(delete=False,suffix=".tif") as tmp:
                with rasterio.open(tmp.name,'w',
                    driver='GTiff',
                    height=gz.shape[0],
                    width=gz.shape[1],
                    count=1,
                    dtype=gz.dtype,
                    crs='EPSG:4326',
                    transform=transform) as dst:
                    dst.write(gz,1)

                with open(tmp.name,"rb") as f:
                    st.download_button("Download GeoTIFF",f,f"{param}.tif")

else:
    st.warning("Add lat/lon columns")

# ---------------- DATA ----------------
st.dataframe(df)