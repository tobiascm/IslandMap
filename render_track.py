"""Render the Iceland GPX track onto a basemap and export as PNG."""
import gpxpy
import matplotlib.pyplot as plt
import contextily as cx
from pyproj import Transformer

GPX_PATH = "island_track.gpx"
OUTPUT_PATH = "island_track.png"

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
xlim = (min(xs) - margin_x, max(xs) + margin_x)
ylim = (min(ys) - margin_y, max(ys) + margin_y)

width = xlim[1] - xlim[0]
height = ylim[1] - ylim[0]
fig_height = 14
fig_width = fig_height * width / height

fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=200)
ax.plot(xs, ys, color="#e3120b", linewidth=1.8, alpha=0.9)

ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_aspect("equal")

cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik, zoom="auto")

ax.set_axis_off()
ax.set_title("Islandreise 2026", fontsize=20, fontweight="bold", pad=12)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
print(f"Saved {OUTPUT_PATH}")
