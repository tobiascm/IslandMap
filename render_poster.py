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
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cairosvg
import io
from PIL import Image

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
    11: "Þverfellshorn · Grindavík · Lavafeld · Küstentour",
}

# a few waypoint names need a more specific display name than their raw
# "Tag NN – ..." text (e.g. the actual hike target, not the wider area)
NAME_OVERRIDES = {
    "Esja / Þverfellshorn über Steinn": "Þverfellshorn",
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
# overnight (camp) location per trip day, derived from the track timeline:
# the spot where the track rests through the night (densest cluster of points
# between 22:00 and 06:00 Iceland time = UTC). The night AFTER trip-day N is
# attributed to day N.
# --------------------------------------------------------------------------
def read_overnights():
    with open(TRACK_GPX, encoding="utf-8") as f:
        gpx = gpxpy.parse(f)
    nights = defaultdict(list)
    for trk in gpx.tracks:
        for seg in trk.segments:
            for p in seg.points:
                t = p.time
                if t.hour >= 22 or t.hour < 6:
                    ev = t.date() if t.hour >= 22 else date.fromordinal(
                        t.date().toordinal() - 1)
                    nights[day_from_date(ev)].append((p.latitude, p.longitude))

    camps = {}         # trip_day -> (x, y) in EPSG:3857
    for day, pts in nights.items():
        if not (1 <= day <= 11):
            continue
        # densest point: the one with the most neighbours within ~400 m
        best, best_n = pts[0], -1
        for a in pts:
            cnt = sum(1 for b in pts if haversine_km(a, b) < 0.4)
            if cnt > best_n:
                best, best_n = a, cnt
        # average the tight cluster around that point for a stable centre
        near = [b for b in pts if haversine_km(best, b) < 0.4]
        lat = sum(p[0] for p in near) / len(near)
        lon = sum(p[1] for p in near) / len(near)
        camps[day] = WGS84_TO_MERC.transform(lon, lat)
    return camps


CAMP_NAMES = {
    1: "Flúðir Camping",
    2: "Vík Campsite",
    3: "Svínafell Camping",
    4: "Myllulækur Campsite",
    5: "Seyðisfjörður Campsite",
    6: "Möðrudalur / Fjalladýrð Camping",
    7: "Hamrar Camping",
    8: "Varmahlíð Camping",
    9: "Borgarnes Camping",
    10: "Mosskógar Camping",
    11: "Sandgerði Camping",
}


# --------------------------------------------------------------------------
# basemap (cached): 3D shaded-relief satellite with a flat sea
# --------------------------------------------------------------------------
def build_basemap(w, s, e, n, zoom=ZOOM, tag="iceland"):
    cache = (f"data/_basemap_z{zoom}.npz" if tag == "iceland"
             else f"data/_basemap_{tag}_z{zoom}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["img"], tuple(d["ext"])

    terr = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
    print(f"Downloading elevation tiles ({tag}, z{zoom}) ...")
    dem, ext = cx.bounds2img(w, s, e, n, zoom=zoom, source=terr, ll=False)
    print(f"Downloading satellite tiles ({tag}, z{zoom}) ...")
    sat, ext2 = cx.bounds2img(w, s, e, n, zoom=zoom,
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
    deep = rgb[elev_raw < -200]
    ocean = (np.median(deep, axis=0) if len(deep)
             else np.array([22, 58, 73]) / 255.0)   # inland tiles have no sea
    shaded[sea] = ocean

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
                names = [NAME_OVERRIDES.get(m[0], m[0].split(" / ")[0].split(" – ")[0])
                          for m in members]
                label = " · ".join(names)
            labels.append((day, label, x, y, len(members)))
    return labels


# map axes geometry (also used to fit the per-day zoom box to the right aspect)
MAP_BOX = [0.235, 0.015, 0.755, 0.97]
FIG_W, FIG_H = 19.2, 10.8
MAP_ASPECT = (MAP_BOX[2] * FIG_W) / (MAP_BOX[3] * FIG_H)

# labels in cramped corners of the *overview* map get a fixed manual nudge
# (projected metres) since adjust_text alone can't pull them apart; these are
# only applied on the full poster, not the zoomed-in per-day variants
MANUAL_LABEL_NUDGE = {
    "Reykjanes-Küstentour · Gunnuhver": (-35055, -5172),
    "Grindavík": (-61240, -60329),
    "Lavafeld bei Sýlingarfell": (-62705, -111777),
}
MANUAL_LABEL_HA = {
    "Reykjanes-Küstentour · Gunnuhver": "left",
    "Grindavík": "left",
    "Lavafeld bei Sýlingarfell": "left",
}


CAMP_ICON_SVG = "data/campingsymbol.svg"
_camp_icon_cache = None


def load_camp_icon():
    """Rasterise the camping-symbol SVG to an RGBA array (white silhouette, transparent bg)."""
    global _camp_icon_cache
    if _camp_icon_cache is None:
        png_bytes = cairosvg.svg2png(url=CAMP_ICON_SVG, output_width=200, output_height=200)
        img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
        arr = np.array(img).astype(np.float64) / 255.0
        # recolor the black silhouette to dark navy, keep the white outline/alpha as-is
        black = (arr[..., 0] < 0.3) & (arr[..., 1] < 0.3) & (arr[..., 2] < 0.3) & (arr[..., 3] > 0)
        arr[black, 0:3] = np.array([13, 23, 38]) / 255.0
        _camp_icon_cache = arr
    return _camp_icon_cache


def fit_bbox(xs, ys, margin, aspect):
    """Bounding box around the points, padded and stretched to `aspect` (w/h)."""
    w, e = min(xs), max(xs)
    s, n = min(ys), max(ys)
    dx, dy = (e - w), (n - s)
    w -= dx * margin; e += dx * margin
    s -= dy * margin; n += dy * margin
    dx, dy = (e - w), (n - s)
    if dx / dy < aspect:                      # too narrow -> widen
        extra = (dy * aspect - dx) / 2
        w -= extra; e += extra
    else:                                     # too short -> heighten
        extra = (dx / aspect - dy) / 2
        s -= extra; n += extra
    return w, s, e, n


# --------------------------------------------------------------------------
# draw one poster: overview (focus_day=None) or a single zoomed-in day
# --------------------------------------------------------------------------
def render(runs, day_start, activities, camps=None, focus_day=None,
           out=OUTPUT, day_zoom=10):
    camps = camps or {}
    if focus_day is None:
        allx = [x for _, xs, _ in runs for x in xs]
        ally = [y for _, _, ys in runs for y in ys]
        mx = (max(allx) - min(allx)) * 0.06
        my = (max(ally) - min(ally)) * 0.04
        w, e = min(allx) - mx, max(allx) + mx
        s, n = min(ally) - my, max(ally) + my
        img, ext = build_basemap(w, s, e, n)
    else:
        fx = [x for day, xs, _ in runs if day == focus_day for x in xs]
        fy = [y for day, _, ys in runs if day == focus_day for y in ys]
        for day, _, x, y, _ in activities:
            if day == focus_day:
                fx.append(x); fy.append(y)
        if focus_day in camps:                     # keep the camp site in frame
            cx_, cy_ = camps[focus_day]
            fx.append(cx_); fy.append(cy_)
        w, s, e, n = fit_bbox(fx, fy, margin=0.22, aspect=MAP_ASPECT)
        img, ext = build_basemap(w, s, e, n, zoom=day_zoom, tag=f"day{focus_day}")
    left, right, bottom, top = ext

    # zoomed-in day images get a much chunkier track and bigger map text
    if focus_day is None:
        halo_lw, track_lw, lbl_fs, dot_s = 5.0, 3.2, 13, 70
    else:
        halo_lw, track_lw, lbl_fs, dot_s = 10.0, 6.8, 19, 130

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=150, facecolor=BG)
    ax = fig.add_axes(MAP_BOX)
    ax.imshow(img, extent=(left, right, bottom, top), origin="upper", zorder=0)
    ax.set_xlim(w, e)
    ax.set_ylim(s, n)
    ax.set_aspect("equal")
    ax.set_axis_off()

    # track: travel days (grey) first, then coloured days; in focus mode the
    # non-focus days recede into faint context lines
    for day, xs, ys in runs:
        if day is None:
            ax.plot(xs, ys, color="#8a93a3", lw=1.2, alpha=0.5, zorder=1)
    for day, xs, ys in runs:
        if day is None:
            continue
        col = PALETTE[day - 1]
        if focus_day is not None and day != focus_day:
            ax.plot(xs, ys, color="#6b7689", lw=2.0, alpha=0.45, zorder=1)
            continue
        ax.plot(xs, ys, color="white", lw=halo_lw, alpha=0.55,
                solid_capstyle="round", zorder=2)
        ax.plot(xs, ys, color=col, lw=track_lw, solid_capstyle="round", zorder=3)

    # activity dots (non-focus days dimmed in focus mode)
    for day, _, x, y, _ in activities:
        if focus_day is not None and day != focus_day:
            ax.scatter([x], [y], s=45, color="#6b7689", edgecolors="white",
                       linewidths=0.8, alpha=0.5, zorder=4)
        else:
            ax.scatter([x], [y], s=dot_s, color=PALETTE[day - 1], edgecolors="white",
                       linewidths=1.3, zorder=5)

    # activity labels with leader lines (auto de-overlapped)
    cx_mid = (w + e) / 2
    texts = []
    for day, label, x, y, members in activities:
        if focus_day is not None and day != focus_day:
            continue
        wrapped = textwrap.fill(label, width=19) if len(label) > 19 else label
        ha = "left" if x >= cx_mid else "right"
        if focus_day is None and label in MANUAL_LABEL_NUDGE:
            dx, dy = MANUAL_LABEL_NUDGE[label]
            ha = MANUAL_LABEL_HA.get(label, ha)
            ann = ax.annotate(wrapped, xy=(x, y), xytext=(x + dx, y + dy),
                              fontsize=lbl_fs, color="white", ha=ha, va="center", zorder=7,
                              bbox=dict(boxstyle="round,pad=0.28", fc="#0d1726ee",
                                        ec=PALETTE[day - 1], lw=1.0),
                              arrowprops=dict(arrowstyle="-", color="white", lw=0.7,
                                              alpha=0.8))
            ann.set_path_effects([pe.withStroke(linewidth=0.5, foreground="black")])
            continue
        txt = ax.text(x, y, wrapped, fontsize=lbl_fs, color="white", ha=ha,
                      va="center", zorder=7,
                      bbox=dict(boxstyle="round,pad=0.28", fc="#0d1726ee",
                                ec=PALETTE[day - 1], lw=1.0))
        txt.set_path_effects([pe.withStroke(linewidth=0.5, foreground="black")])
        texts.append(txt)
    adjust_text(texts, ax=ax,
                expand=(1.25, 1.6), force_text=(0.4, 0.6),
                arrowprops=dict(arrowstyle="-", color="white", lw=0.7, alpha=0.8))

    # numbered day badges (focus mode: only the focused day's start)
    for day, (x, y) in sorted(day_start.items()):
        if focus_day is not None and day != focus_day:
            continue
        col = PALETTE[day - 1]
        ax.scatter([x], [y], s=430, color=col, edgecolors="white", linewidths=1.8,
                   zorder=8)
        ax.text(x, y, str(day), fontsize=12, fontweight="bold",
                color=text_color_for(col), ha="center", va="center", zorder=9)

    # overnight marker for the focused day: a tent/teepee pictogram (filled
    # triangle + crossed poles sticking out the top), drawn in map units with a
    # white halo so it reads clearly on the satellite imagery
    if focus_day is not None and focus_day in camps:
        cxp, cyp = camps[focus_day]
        icon = load_camp_icon()
        zoom_factor = 0.34 if focus_day is not None else 0.18
        oi = OffsetImage(icon, zoom=zoom_factor)
        ab = AnnotationBbox(oi, (cxp, cyp), frameon=False, zorder=9, pad=0)
        ax.add_artist(ab)
        name = CAMP_NAMES.get(focus_day)
        if name:
            ax.annotate(name, (cxp, cyp), xytext=(0, -28),
                        textcoords="offset points", ha="center", va="top",
                        fontsize=lbl_fs, fontweight="bold", color="white",
                        zorder=9,
                        path_effects=[pe.withStroke(linewidth=3.5, foreground="#0d1726")])

    # focus-day title chip on the map
    if focus_day is not None:
        col = PALETTE[focus_day - 1]
        ax.text(0.025, 0.965, f"TAG {focus_day}", transform=ax.transAxes,
                fontsize=34, fontweight="bold", color="white",
                ha="left", va="top", zorder=10,
                bbox=dict(boxstyle="round,pad=0.4", fc="#0d1726dd", ec=col, lw=2.2))

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
        dim = focus_day is not None and day != focus_day
        pan.add_patch(FancyBboxPatch(
            (0.04, ty + 0.004), 0.92, ch - 0.008,
            boxstyle="round,pad=0,rounding_size=0.015",
            fc=CARD, ec=("#33425c" if dim else col),
            lw=(1.0 if dim else (2.6 if focus_day == day else 1.1)),
            alpha=(0.35 if dim else 0.97), zorder=1,
            mutation_aspect=0.35))
        pan.scatter([0.115], [cy], s=540, color=col, edgecolors="white",
                    linewidths=1.6, alpha=(0.4 if dim else 1.0), zorder=3)
        pan.text(0.115, cy, str(day), fontsize=13, fontweight="bold",
                 color=text_color_for(col), ha="center", va="center",
                 alpha=(0.5 if dim else 1.0), zorder=4)
        pan.text(0.205, cy - 0.015, f"TAG {day}", fontsize=14, fontweight="bold",
                 color=("#7b87a0" if dim else "white"),
                 ha="left", va="center", zorder=4)
        wrapped = textwrap.fill(HIGHLIGHTS[day], width=30)
        pan.text(0.205, cy + 0.013, wrapped, fontsize=13,
                 color=("#69748c" if dim else "#c3cee0"),
                 ha="left", va="center", zorder=4, linespacing=1.15)

    fig.savefig(out, dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"Saved {out}")


def main(days=None):
    runs, day_start = read_track()
    activities = read_activities()
    camps = read_overnights()
    render(runs, day_start, activities, camps, focus_day=None, out=OUTPUT)
    for d in (days or []):
        render(runs, day_start, activities, camps, focus_day=d,
               out=f"output/tag_{d:02d}.png")


if __name__ == "__main__":
    import sys
    days = [int(a) for a in sys.argv[1:]]
    main(days=days)
