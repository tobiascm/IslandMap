"""Render the Iceland GPX track onto a 3D shaded-relief satellite map.

The basemap is built from two georeferenced XYZ tile layers fetched for the
exact same bounding box, so they are pixel-aligned and the GPX track overlays
precisely (coastlines stay true to the real data):

* a real Digital Elevation Model (AWS Terrain "terrarium" tiles) -> hillshade
* Esri World Imagery satellite tiles, draped over the hillshade for a 3D look

Everything is computed in EPSG:3857, so the projected track coordinates and the
basemap share one coordinate system and line up to the pixel.
"""
import gpxpy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LightSource
import contextily as cx
from pyproj import Transformer

GPX_PATH = "island_track.gpx"
OUTPUT_PATH = "island_track.png"

# --- look & feel -----------------------------------------------------------
ZOOM = 9                 # tile zoom level (higher = sharper, slower download)
BLEND_MODE = "overlay"   # "overlay" | "soft" | "hsv"
VERT_EXAG = 30           # vertical exaggeration of the 3D relief
LIGHT_AZIMUTH = 315      # light direction (NW)
LIGHT_ALTITUDE = 45      # light angle above the horizon
TERRARIUM = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"

# --- read the track --------------------------------------------------------
with open(GPX_PATH, encoding="utf-8") as f:
    gpx = gpxpy.parse(f)

lats, lons = [], []
for track in gpx.tracks:
    for segment in track.segments:
        for point in segment.points:
            lats.append(point.latitude)
            lons.append(point.longitude)

transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
xs, ys = transformer.transform(lons, lats)

margin_x = (max(xs) - min(xs)) * 0.08
margin_y = (max(ys) - min(ys)) * 0.05
w, e = min(xs) - margin_x, max(xs) + margin_x
s, n = min(ys) - margin_y, max(ys) + margin_y

# --- fetch DEM + satellite for an identical extent -------------------------
print("Downloading elevation tiles ...")
img_dem, ext = cx.bounds2img(w, s, e, n, zoom=ZOOM, source=TERRARIUM, ll=False)
print("Downloading satellite tiles ...")
img_sat, ext_sat = cx.bounds2img(w, s, e, n, zoom=ZOOM,
                                 source=cx.providers.Esri.WorldImagery, ll=False)
assert np.allclose(ext, ext_sat), "DEM and satellite extents differ"

# decode terrarium elevation to metres
r = img_dem[:, :, 0].astype("f8")
g = img_dem[:, :, 1].astype("f8")
b = img_dem[:, :, 2].astype("f8")
elev_raw = (r * 256 + g + b / 256) - 32768
sea = elev_raw < 0                      # crisp coastline straight from the DEM
elevation = np.clip(elev_raw, 0, None)  # only the land drives the hillshade

rgb = img_sat[:, :, :3].astype("f8") / 255.0

# real-world pixel size (metres) for a physically correct hillshade
left, right, bottom, top = ext
dx = (right - left) / rgb.shape[1]
dy = (top - bottom) / rgb.shape[0]

light = LightSource(azdeg=LIGHT_AZIMUTH, altdeg=LIGHT_ALTITUDE)
shaded = light.shade_rgb(rgb, elevation, blend_mode=BLEND_MODE,
                         vert_exag=VERT_EXAG, dx=dx, dy=dy)

# paint the whole sea one flat colour (sampled from deep water) so the
# satellite's shallow-water tones no longer look odd along the coast
ocean_color = np.median(rgb[elev_raw < -200], axis=0)
shaded[sea] = ocean_color

# --- plot ------------------------------------------------------------------
fig_height = 14
fig_width = fig_height * (e - w) / (n - s)
fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=200)

ax.imshow(shaded, extent=(left, right, bottom, top), origin="upper")
ax.plot(xs, ys, color="#ff2d2d", linewidth=2.2,
        path_effects=[pe.Stroke(linewidth=4, foreground="white", alpha=0.6),
                      pe.Normal()])

ax.set_xlim(w, e)
ax.set_ylim(s, n)
ax.set_aspect("equal")
ax.set_axis_off()
ax.set_title("Islandreise 2026", fontsize=20, fontweight="bold", pad=12)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
print(f"Saved {OUTPUT_PATH}")
