---
deeplabcut:
  last_content_updated: '2026-07-14'
  last_metadata_updated: '2026-07-14'
  ignore: false
---

# Deriving fish swimming kinematics and choosing a reliable observation duration

This recipe shows how to turn DeepLabCut pose tracks of a swimming fish into
common **swimming-kinematics metrics** — tail-beat frequency, swimming speed,
body curvature, and a day/night (diel) index — and how to decide **how long you
need to record** before those metrics are stable, using the Spearman–Brown
reliability index.

The methods below are generic with respect to species and metric: they operate on
any single-animal DeepLabCut output where landmarks are placed along the body
midline.

## 1. A midline landmark scheme for fish

Placing landmarks along the body midline (plus the tail and, optionally, fins)
makes the geometry below straightforward. A scheme that works well for an
elongated fish is:

| Group | Landmarks                               | Used for                                         |
| ----- | --------------------------------------- | ------------------------------------------------ |
| Head  | `Head1`, `Head2`                        | Heading reference line (for tail-beat detection) |
| Body  | `Body1` … `Body5`                       | Body curvature (circle fit), displacement/speed  |
| Tail  | `Tail1` … `Tail4`                       | Tail-beat frequency                              |
| Fins  | e.g. `PectoralL`, `PectoralR`, `Dorsal` | optional context                                 |

Label and train as usual (see the
[standard DeepLabCut user guide](../standardDeepLabCut_UserGuide.md)), then run
`deeplabcut.analyze_videos` to produce the per-frame predictions used below.

## 2. Loading the predictions

```python
import numpy as np
import pandas as pd

# The .h5 written by deeplabcut.analyze_videos (single-animal project)
df = pd.read_hdf("my_videoDLC_resnet50_FishJul14shuffle1_1000000.h5")

# Drop the top "scorer" column level so columns are (bodypart, coord)
scorer = df.columns.get_level_values("scorer")[0]
df = df[scorer]

FPS = 30.0            # video frame rate
MM_PER_PX = 2.0       # spatial calibration (mm per pixel), from a known reference
BODY_LENGTH_MM = 490  # mean body length of the target fish

def xy(bodypart, p_cutoff=0.6):
    """(N, 2) array of x, y for a bodypart; low-confidence points -> NaN."""
    sub = df[bodypart]
    pts = sub[["x", "y"]].to_numpy(dtype=float, copy=True)
    pts[sub["likelihood"].to_numpy() < p_cutoff] = np.nan
    return pts
```

Always condition on `likelihood` (the `p_cutoff`) before computing kinematics, so
that uncertain detections don't leak into the metrics.

## 3. Swimming speed (absolute and body-length-normalized)

Sum the frame-to-frame Euclidean displacement of a body landmark over a time
window, convert pixels to millimetres, and divide by the window duration:

```python
def swimming_speed(bodypart, mm_per_px=MM_PER_PX, fps=FPS):
    """Returns (absolute speed mm/s, body-length-normalized speed BL/s)."""
    p = xy(bodypart)
    steps = np.linalg.norm(np.diff(p, axis=0), axis=1)   # px per frame
    path_mm = np.nansum(steps) * mm_per_px               # total path length (mm)
    dt = (len(p) - 1) / fps                               # window duration (s)
    v = path_mm / dt
    return v, v / BODY_LENGTH_MM
```

Absolute speed (mm/s) is right for movement in physical space; the
body-length-normalized speed (BL/s) lets you compare individuals or species of
different sizes.

## 4. Tail-beat frequency (TBF)

Define a heading reference line from `Head1` to `Head2`. Every time a tail
landmark crosses that line, one half-cycle has occurred. Count the crossings `N`
over a window and divide by two to convert crossings to full beats:

```python
def _side_of_line(p, a, b):
    """Signed side (+/-1) of points p relative to the a->b line."""
    return np.sign((b[:, 0] - a[:, 0]) * (p[:, 1] - a[:, 1])
                   - (b[:, 1] - a[:, 1]) * (p[:, 0] - a[:, 0]))

def tail_beat_frequency(tail="Tail1", fps=FPS):
    a, b, p = xy("Head1"), xy("Head2"), xy(tail)
    s = _side_of_line(p, a, b)
    s = s[np.isfinite(s) & (s != 0)]          # drop NaNs / on-line points
    n_crossings = int(np.count_nonzero(np.diff(s) != 0))
    dt = (len(p) - 1) / fps
    return (n_crossings / 2) / dt              # Hz
```

Typical analysis windows range from 1 minute (to catch transient burst swimming)
to 1 hour (for sustained averages).

## 5. Body curvature

Fit a circle to the `Body1`…`Body5` landmarks in a frame; the curvature is the
inverse of the fitted radius, `κ = 1/R`:

```python
def fit_circle_radius(points):
    """Algebraic (Kåsa) circle fit -> radius. points: (n, 2)."""
    pts = points[np.isfinite(points).all(axis=1)]
    if len(pts) < 3:
        return np.nan
    x, y = pts[:, 0], pts[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    cx, cy, c = np.linalg.lstsq(A, x**2 + y**2, rcond=None)[0]
    return np.sqrt(c + cx**2 + cy**2)

def body_curvature(frame_index, mm_per_px=MM_PER_PX):
    pts = np.array([xy(f"Body{i}")[frame_index] for i in range(1, 6)])
    R_px = fit_circle_radius(pts)
    return np.nan if not np.isfinite(R_px) else 1.0 / (R_px * mm_per_px)  # mm^-1
```

> **Note:** the reference study uses the **HyperLS** algebraic circle fit
> (Kanatani et al., 2011), which reduces the bias of the plain algebraic fit shown
> here. Swap `fit_circle_radius` for a HyperLS implementation when you need the
> lower-bias estimate; the exact code is in the reference repository.

## 6. Diel (day/night) index

Split each metric by a day/night boundary and take the ratio of the nighttime to
daytime mean. A diel index `> 1` means the metric is larger at night:

```python
def diel_index(values, timestamps, day_start=6, day_end=20):
    """values and timestamps (datetime64) are aligned 1-D arrays."""
    hour = pd.DatetimeIndex(timestamps).hour
    is_day = (hour >= day_start) & (hour < day_end)
    return np.nanmean(values[~is_day]) / np.nanmean(values[is_day])
```

## 7. How long must you record? — Spearman–Brown reliability

Long-term metrics vary day to day, so a short clip may not represent the animal.
Model reliability as a function of the number of averaged days `k` with the
**Spearman–Brown** formula, using the day-to-day **intraclass correlation
coefficient (ICC)** as the single-day reliability `ρ`:

```python
def icc_one_way(daily_values):
    """
    One-way random-effects ICC(1,1) from Shrout & Fleiss (1979).
    daily_values: list of 1-D arrays, one per day (e.g. hourly metric values).
    """
    groups = [np.asarray(d, float) for d in daily_values]
    all_vals = np.concatenate(groups)
    grand = all_vals.mean()
    g, N = len(groups), len(all_vals)
    n_i = np.array([len(d) for d in groups])
    day_means = np.array([d.mean() for d in groups])

    ssb = np.sum(n_i * (day_means - grand) ** 2)
    ssw = np.sum([np.sum((d - d.mean()) ** 2) for d in groups])
    ms_between = ssb / (g - 1)
    ms_within = ssw / (N - g)
    k_bar = n_i.mean()
    return (ms_between - ms_within) / (ms_between + (k_bar - 1) * ms_within)

def spearman_brown(icc, k):
    return (k * icc) / (1 + (k - 1) * icc)

def min_days_for_reliability(icc, target=0.8, max_days=365):
    """Smallest number of averaged days whose reliability reaches `target`."""
    for k in range(1, max_days + 1):
        if spearman_brown(icc, k) >= target:
            return k
    return None
```

Use it on pilot data (e.g. hourly values grouped by day) to get an evidence-based
recording length per metric:

```python
icc = icc_one_way(daily_hourly_values)     # from your pilot recording
print("single-day ICC:", round(icc, 3))
print("days to reach reliability 0.8:", min_days_for_reliability(icc, target=0.8))
```

A common convention treats reliability `> 0.8` as "very good" (Ursachi et al.,
2015). The required duration typically differs markedly across metrics — highly
variable or slowly fluctuating metrics need longer records to stabilize — which is
exactly why it is worth estimating rather than assuming. The thresholds you obtain
are context-specific (they depend on species, individual, life stage, and
environment), but the workflow that produces them is general.

## Source

Method adapted from Hwang et al. (2026), *Ecological Informatics* 95, 103760
([DOI](https://doi.org/10.1016/j.ecoinf.2026.103760),
[reference code](https://github.com/Denny-Hwang/FishML_Observation_to_Replication)).
Please also [cite DeepLabCut](../citation.md).
