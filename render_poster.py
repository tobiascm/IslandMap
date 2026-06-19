"""Render the Iceland trip as a 16:9 poster.

Left: a panel listing the 11 travel days with their highlights.
Right: the 3D shaded-relief satellite map (see render_track.py) with

* the GPS track coloured per day (arrival/departure days greyed out),
* a numbered badge (1-11) at the start of every day,
* one coloured dot per visited activity, with labels; tight clusters of
  stops (e.g. Reykjavik) are merged into a single collected label.

Track days come straight from the GPS timestamps (08.-18.06. = day 1-11),
so the colouring needs no guessing. Activities, dates and descriptions come
from data/Islandreise_2026_Aktivitaeten_Wegpunkte.gpx; the per-day highlight
lists come from the "Kurzfassung" table in the highlights markdown.
"""
import os
import re
import math
import textwrap
from datetime import date
from collections import defaultdict

import gpxpy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LightSource
import contextily as cx
from pyproj import Transformer
from adjustText import adjust_text

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
TRACK_GPX = "data/island_track.gpx"
ACT_GPX = "data/Islandreise_2026_Aktivitaeten_Wegpunkte.gpx"
OUTPUT = "output/island_poster.png"

ZOOM = 8
BG = "#0c1626"            # poster background (dark navy)
CARD = "#172943"          # day-card fill
DAY1 = date(2026, 6, 8)   # day 1; day 11 = 18.06.; 07. & 19.06. are travel days

# 11 visually distinct day colours (day 1 .. day 11)
PALETTE = ["#ff3b30", "#ff9500", "#ffd60a", "#a3e635", "#34c759",
           "#00d0c0", "#32ade6", "#5b8cff", "#a855f7", "#ff4dd2", "#ff7a8a"]

# per-day highlight lines for the side panel (from the markdown "Kurzfassung")
HIGHLIGHTS = {
    1: "Reykjadalur · Kerið · Geysir/Strokkur · Gullfoss",
    2: "Seljalandsfoss · Gljúfrabúi · Skógafoss",
    3: "Fjaðrárgljúfur · Dverghamrar · Lómagnúpur · Skaftafell",
    4: "Hofskirkja · Jökulsárlón · Diamond Beach",
    5: "Vestrahorn · Djúpivogur · Gufufoss · Seyðisfjörður",
    6: "Stuðlagil · Vök Baths · Möðrudalur",
    7: "Hverir · Krafla/Leirhnjúkur · Dettifoss · Goðafoss",
    8: "Whale Watching · Beach Baths · Tröllaskagi · Reykjafoss",
    9: "Kolugljúfur · Grábrók · Hraunfossar/Barnafoss",
    10: "Glymur · Reykjavík-Stadtrundgang",
    11: "Esja · Reykjanes · Grindavík · Lavafeld · Küstentour",
}

CLUSTER_KM = 4.0          # merge same-day stops closer than this
WGS84_TO_MERC = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def text_color_for(hex_color):
    """Black on light badges, white on dark ones."""
    r = int(hex_color[1:3], 16) / 255
    g = int(hex_color[3:5], 16) / 255
    b = int(hex_color[5:7], 16) / 255
    return "black" if (0.299 * r + 0.587 * g + 0.114 * b) > 0.6 else "white"


def haversine_km(a, b):
    (la1, lo1), (la2, lo2) = a, b
    r = 6371.0
    p1, p2 = math.radians(la1), math.radians(la2)
    dlat = math.radians(la2 - la1)
    dlon = math.radians(lo2 - lo1)
    h = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def short_name(name):
    return re.sub(r"^Tag\s*\d+\s*[–-]\s*", "", name).strip()


def day_from_date(d):
    return (d - DAY1).days + 1


# --------------------------------------------------------------------------
# basemap (cached): 3D shaded-relief satellite with a flat sea
# --------------------------------------------------------------------------
def build_basemap(w, s, e, n):
    cache = f"data/_basemap_z{ZOOM}.npz"
    if os.path.exists(cache):
        d = np.load(cache)
        return d["img"], tuple(d["ext"])

    terr = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
    print("Downloading elevation tiles ...")
    dem, ext = cx.bounds2img(w, s, e, n, zoom=ZOOM, source=terr, ll=False)
    print("Downloading satellite tiles ...")
    sat, ext2 = cx.bounds2img(w, s, e, n, zoom=ZOOM,
                              source=cx.providers.Esri.WorldImagery, ll=False)
    assert np.allclose(ext, ext2)

    r = dem[:, :, 0].astype("f8")
    g = dem[:, :, 1].astype("f8")
    b = dem[:, :, 2].astype("f8")
    elev_raw = (r * 256 + g + b / 256) - 32768
    sea = elev_raw < 0
    elevation = np.clip(elev_raw, 0, None)

    rgb = sat[:, :, :3].astype("f8") / 255.0
    left, right, bottom, top = ext
    dx = (right - left) / rgb.shape[1]
    dy = (top - bottom) / rgb.shape[0]

    light = LightSource(azdeg=315, altdeg=45)
    shaded = light.shade_rgb(rgb, elevation, blend_mode="overlay",
                             vert_exag=30, dx=dx, dy=dy)
    shaded[sea] = np.median(rgb[elev_raw < -200], axis=0)

    img = (np.clip(shaded, 0, 1) * 255).astype("uint8")
    np.savez_compressed(cache, img=img, ext=np.array(ext))
    return img, tuple(ext)


# --------------------------------------------------------------------------
# read track grouped by day
# --------------------------------------------------------------------------
def read_track():
    with open(TRACK_GPX, encoding="utf-8") as f:
        gpx = gpxpy.parse(f)
    runs = []          # (day_or_None, xs, ys) contiguous same-day runs
    day_start = {}     # day -> (x, y) first point chronologically
    for trk in gpx.tracks:
        for seg in trk.segments:
            cur_day, cx_, cy_ = None, [], []
            for p in seg.points:
                d = day_from_date(p.time.date())
                day = d if 1 <= d <= 11 else None
                x, y = WGS84_TO_MERC.transform(p.longitude, p.latitude)
                if day is not None and day not in day_start:
                    day_start[day] = (x, y)
                if day != cur_day and cx_:
                    runs.append((cur_day, cx_, cy_))
                    cx_, cy_ = [], []
                cur_day = day
                cx_.append(x)
                cy_.append(y)
            if cx_:
                runs.append((cur_day, cx_, cy_))
    return runs, day_start


# --------------------------------------------------------------------------
# read activities, cluster tight same-day groups
# --------------------------------------------------------------------------
def read_activities():
    with open(ACT_GPX, encoding="utf-8") as f:
        gpx = gpxpy.parse(f)
    byday = defaultdict(list)
    for w in gpx.waypoints:
        m = re.search(r"(\d{4})-(\d{2})-(\d{2})", w.comment or "")
        d = day_from_date(date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
        byday[d].append((short_name(w.name), w.latitude, w.longitude))

    labels = []        # (day, label_text, x, y, n_members)
    for day, items in byday.items():
        parent = list(range(len(items)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a = (items[i][1], items[i][2])
                b = (items[j][1], items[j][2])
                if haversine_km(a, b) < CLUSTER_KM:
                    parent[find(i)] = find(j)

        groups = defaultdict(list)
        for i, it in enumerate(items):
            groups[find(i)].append(it)
        for members in groups.values():
            lat = sum(m[1] for m in members) / len(members)
            lon = sum(m[2] for m in members) / len(members)
            x, y = WGS84_TO_MERC.transform(lon, lat)
            if len(members) > 1 and any("Reykjavík" in m[0] for m in members):
                label = "Reykjavík"
            else:
                names = [m[0].split(" / ")[0].split(" – ")[0] for m in members]
                label = " · ".join(names)
            labels.append((day, label, x, y, len(members)))
    return labels


# --------------------------------------------------------------------------
# draw
# --------------------------------------------------------------------------
def main():
    runs, day_start = read_track()
    activities = read_activities()

    allx = [x for _, xs, _ in runs for x in xs]
    ally = [y for _, _, ys in runs for y in ys]
    mx = (max(allx) - min(allx)) * 0.06
    my = (max(ally) - min(ally)) * 0.04
    w, e = min(allx) - mx, max(allx) + mx
    s, n = min(ally) - my, max(ally) + my

    img, ext = build_basemap(w, s, e, n)
    left, right, bottom, top = ext

    fig = plt.figure(figsize=(19.2, 10.8), dpi=150, facecolor=BG)
    ax = fig.add_axes([0.235, 0.015, 0.755, 0.97])
    ax.imshow(img, extent=(left, right, bottom, top), origin="upper", zorder=0)
    ax.set_xlim(w, e)
    ax.set_ylim(s, n)
    ax.set_aspect("equal")
    ax.set_axis_off()

    # track: arrival/departure first (grey), then coloured days on top
    for day, xs, ys in runs:
        if day is None:
            ax.plot(xs, ys, color="#8a93a3", lw=1.2, alpha=0.55, zorder=1)
    for day, xs, ys in runs:
        if day is None:
            continue
        col = PALETTE[day - 1]
        ax.plot(xs, ys, color="white", lw=5.0, alpha=0.55, solid_capstyle="round",
                zorder=2)
        ax.plot(xs, ys, color=col, lw=3.2, solid_capstyle="round", zorder=3)

    # activity dots
    for day, _, x, y, _ in activities:
        ax.scatter([x], [y], s=70, color=PALETTE[day - 1], edgecolors="white",
                   linewidths=1.1, zorder=5)

    # activity labels with leader lines (auto de-overlapped)
    cx_mid = (w + e) / 2
    texts = []
    for day, label, x, y, members in activities:
        wrapped = textwrap.fill(label, width=19) if members > 1 else label
        ha = "left" if x >= cx_mid else "right"
        txt = ax.text(x, y, wrapped, fontsize=13, color="white", ha=ha, va="center",
                      zorder=7,
                      bbox=dict(boxstyle="round,pad=0.28", fc="#0d1726ee",
                                ec=PALETTE[day - 1], lw=1.0))
        txt.set_path_effects([pe.withStroke(linewidth=0.5, foreground="black")])
        texts.append(txt)
    adjust_text(texts, ax=ax,
                expand=(1.25, 1.6), force_text=(0.4, 0.6),
                arrowprops=dict(arrowstyle="-", color="white", lw=0.7, alpha=0.8))

    # numbered day badges on top
    for day, (x, y) in sorted(day_start.items()):
        col = PALETTE[day - 1]
        ax.scatter([x], [y], s=430, color=col, edgecolors="white", linewidths=1.8,
                   zorder=8)
        ax.text(x, y, str(day), fontsize=12, fontweight="bold",
                color=text_color_for(col), ha="center", va="center", zorder=9)

    # ----- side panel -----
    pan = fig.add_axes([0.0, 0.0, 0.235, 1.0])
    pan.set_xlim(0, 1)
    pan.set_ylim(1, 0)            # y inverted: 0 = top
    pan.set_axis_off()
    pan.add_patch(plt.Rectangle((0, 0), 1, 1, color=BG, zorder=0))

    pan.text(0.07, 0.045, "ISLAND 2026", fontsize=30, fontweight="bold",
             color="white", ha="left", va="center")

    top0, bot0 = 0.08, 0.992
    ch = (bot0 - top0) / 11
    for day in range(1, 12):
        ty = top0 + (day - 1) * ch
        cy = ty + ch / 2
        col = PALETTE[day - 1]
        pan.add_patch(FancyBboxPatch(
            (0.04, ty + 0.004), 0.92, ch - 0.008,
            boxstyle="round,pad=0,rounding_size=0.015",
            fc=CARD, ec=col, lw=1.1, alpha=0.95, zorder=1,
            mutation_aspect=0.35))
        pan.scatter([0.115], [cy], s=540, color=col, edgecolors="white",
                    linewidths=1.6, zorder=3)
        pan.text(0.115, cy, str(day), fontsize=13, fontweight="bold",
                 color=text_color_for(col), ha="center", va="center", zorder=4)
        pan.text(0.205, cy - 0.015, f"TAG {day}", fontsize=14, fontweight="bold",
                 color="white", ha="left", va="center", zorder=4)
        wrapped = textwrap.fill(HIGHLIGHTS[day], width=30)
        pan.text(0.205, cy + 0.013, wrapped, fontsize=13, color="#c3cee0",
                 ha="left", va="center", zorder=4, linespacing=1.15)

    fig.savefig(OUTPUT, dpi=150, facecolor=BG)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
