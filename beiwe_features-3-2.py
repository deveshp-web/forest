#!/usr/bin/env python3
"""
beiwe_features.py
=================
Reproducible behavioral feature extraction from the public Beiwe sample dataset
(Emedom-Nnamdi et al., 2022; Zenodo DOI 10.5281/zenodo.6471045).

This pipeline is written for a *secondary analysis*. It does not collect data.
It reads the raw Beiwe export directory you download from Zenodo and produces:

  1. A data-coverage table and figure (how much data actually exists per
     participant per day). Report this honestly; it is a real finding.
  2. A GPS-obfuscation diagnostic. The public dataset's coordinates are
     de-identified with Beiwe's "noise GPS" feature, so raw point-to-point
     distance and speed are dominated by injected noise rather than travel.
     This figure QUANTIFIES that noise, which is the methodological core of
     the paper.
  3. Mobility features that survive obfuscation better than raw distance:
     time-at-home, radius of gyration, location entropy, significant places.
  4. A validation of algorithmic time-at-home against the participants'
     OWN self-reported hours at home (the only self-report item in this
     dataset), via correlation, mean absolute error, and a Bland-Altman plot.
  5. Phone-usage features from the power_state stream (screen-on duration,
     nocturnal usage), which are NOT location-obfuscated.

USAGE
-----
    # Always start here. Confirms directory layout and column names.
    python beiwe_features.py inspect --data-dir ./data

    # Then run the full pipeline.
    python beiwe_features.py run --data-dir ./data --out ./output

DEPENDENCIES
------------
    pip install pandas numpy matplotlib scipy scikit-learn

NOTE ON SCHEMA
--------------
Beiwe's raw export layout is:
    <data-dir>/<participant_id>/<stream>/<YYYY-MM-DD HH_MM_SS+00_00>.csv
Column names have varied slightly across Beiwe versions. Run `inspect` FIRST
and reconcile the printed columns against the COLUMN_MAP below before
trusting any output. Do not report a number this script prints if you have
not verified the column it came from.
"""

import argparse
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

# The public sample study ran 2022-03-21 to 2022-03-28 in US Eastern time.
STUDY_TZ = "America/New_York"

# Candidate column names, in priority order. Verify with `inspect`.
COLUMN_MAP = {
    "time": ["UTC time", "timestamp", "UTC_time"],
    "lat": ["latitude", "lat"],
    "lon": ["longitude", "lon", "long"],
    "accuracy": ["accuracy", "horizontal_accuracy"],
    "event": ["event", "state"],
    "answer": ["answer", "response"],
    "question_text": ["question text", "question_text"],
}

# Home cluster radius (metres). Points within this of the home centroid
# count as "at home".
HOME_RADIUS_M = 100.0

# Home-detection window (local hours). Wang et al. (2021, HOPES) define home
# as the GPS location where a participant spent the most time overnight,
# using 21:00-06:00. Adopting their window rather than an arbitrary one makes
# this choice citable. Wraps midnight; see in_window().
HOME_WINDOW = (21, 6)

# Presumed-stationary window (local hours), used ONLY for the obfuscation
# diagnostic. This is deliberately narrower than HOME_WINDOW: at 21:00 many
# people are still out, so including that hour would contaminate an estimate
# that depends on the participant genuinely not moving.
STATIONARY_WINDOW = (1, 5)

# Segments longer than this (seconds) are treated as data gaps and are not
# integrated into time-at-home totals.
MAX_GAP_S = 600

# Point-to-point speeds above this (km/h) are treated as GPS error, not travel.
MAX_PLAUSIBLE_SPEED_KMH = 200.0

# Noise levels (metres, per-axis standard deviation) for the sensitivity
# analysis, and how many random draws to average at each level.
SENSITIVITY_SIGMAS_M = [0.0, 10.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0]
SENSITIVITY_SEEDS = 5

# Cap on points used for the entropy clustering during sensitivity sweeps,
# purely to keep runtime reasonable. Set to None to use every point.
SENSITIVITY_MAX_POINTS = 6000

# Cap on points fed into any DBSCAN clustering step (home estimation,
# significant-places entropy). A week-long, low-mobility trace can have tens
# of thousands of fixes almost all mutually within HOME_RADIUS_M of each
# other, which makes neighbor-list construction blow up in memory regardless
# of ball_tree vs brute force. A systematic (every-Nth-row) subsample
# preserves the time-ordered spatial distribution while keeping clustering
# tractable.
ENTROPY_MAX_POINTS = 8000


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------

def pick_column(df, key):
    """Return the first matching column name for a logical field, else None."""
    for candidate in COLUMN_MAP[key]:
        if candidate in df.columns:
            return candidate
    return None


def in_window(hours, window):
    """
    Boolean mask for an hour-of-day window, handling midnight wrap.

    (1, 5)   -> 01:00 through 05:00
    (21, 6)  -> 21:00 through 06:00, spanning midnight
    """
    start, end = window
    if start <= end:
        return (hours >= start) & (hours <= end)
    return (hours >= start) | (hours <= end)


def is_overnight(hours):
    """Boolean mask for the wrapping 21:00-06:00 window (Wang et al. 2021)."""
    import numpy as _np
    h = _np.asarray(hours)
    return (h >= NIGHT_START_H) | (h < NIGHT_END_H)


def haversine_m(lat1, lon1, lat2, lon2):
    """Vectorised great-circle distance in metres."""
    R = 6_371_000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def load_stream(participant_dir, stream):
    """Concatenate every hourly CSV for one participant and one data stream."""
    stream_dir = participant_dir / stream
    if not stream_dir.is_dir():
        return pd.DataFrame()

    frames = []
    for path in sorted(stream_dir.rglob("*.csv")):
        try:
            chunk = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001
            print(f"    ! could not read {path.name}: {exc}", file=sys.stderr)
            continue
        if not chunk.empty:
            frames.append(chunk)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    time_col = pick_column(df, "time")
    if time_col is None:
        print(f"    ! no recognisable time column in {stream}", file=sys.stderr)
        return pd.DataFrame()

    df["t_utc"] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    df = df.dropna(subset=["t_utc"]).sort_values("t_utc").reset_index(drop=True)
    df["t_local"] = df["t_utc"].dt.tz_convert(STUDY_TZ)
    df["date"] = df["t_local"].dt.date
    return df


def find_participants(data_dir):
    """Beiwe participant IDs are 8-character alphanumeric directory names."""
    candidates = [
        p for p in sorted(Path(data_dir).iterdir())
        if p.is_dir() and not p.name.startswith(".")
    ]
    # A participant directory contains stream subdirectories, not CSVs directly.
    return [p for p in candidates if any(c.is_dir() for c in p.iterdir())]


# ----------------------------------------------------------------------------
# Inspect mode
# ----------------------------------------------------------------------------

def cmd_inspect(args):
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        sys.exit(f"Not a directory: {data_dir}")

    participants = find_participants(data_dir)
    print(f"\nData directory: {data_dir.resolve()}")
    print(f"Participants found: {len(participants)}\n")

    for pdir in participants:
        streams = sorted(c.name for c in pdir.iterdir() if c.is_dir())
        print(f"  {pdir.name}")
        for stream in streams:
            files = list((pdir / stream).rglob("*.csv"))
            if not files:
                print(f"      {stream:<18} 0 files")
                continue
            sample = pd.read_csv(files[0], nrows=3)
            print(f"      {stream:<18} {len(files):>5} files   "
                  f"columns: {list(sample.columns)}")
        print()

    print("Reconcile the column names above against COLUMN_MAP in this script "
          "before running `run`.\n")


# ----------------------------------------------------------------------------
# GPS feature extraction
# ----------------------------------------------------------------------------

def locate_home(gps, lat_col, lon_col):
    """
    Estimate the home coordinate as the densest cluster of overnight points.

    Follows the definition in Wang et al. (2021): home is where the
    participant spent the most time overnight (21:00-06:00 local). DBSCAN is
    run on the haversine metric so epsilon is a true distance rather than a
    degree approximation.
    """
    from sklearn.cluster import DBSCAN

    night = gps[in_window(gps["t_local"].dt.hour, HOME_WINDOW)]
    if len(night) < 10:
        night = gps  # fall back to all points if nighttime coverage is absent
    if night.empty:
        return None

    if len(night) > ENTROPY_MAX_POINTS:
        # Subsample night itself (not just a copy of its columns) so labels
        # from DBSCAN stay aligned with the frame used below for the mean.
        stride = int(np.ceil(len(night) / ENTROPY_MAX_POINTS))
        night = night.iloc[::stride]
    coords = np.radians(night[[lat_col, lon_col]].to_numpy())
    eps = HOME_RADIUS_M / 6_371_000.0
    # algorithm="ball_tree" avoids a full O(n^2) pairwise distance matrix on
    # dense traces (brute force on ~30k points needs ~7.4 GB).
    labels = DBSCAN(eps=eps, min_samples=5, metric="haversine",
                     algorithm="ball_tree").fit_predict(coords)

    valid = labels[labels >= 0]
    if valid.size == 0:
        return float(night[lat_col].median()), float(night[lon_col].median())

    biggest = np.bincount(valid).argmax()
    members = night[labels == biggest]
    return float(members[lat_col].mean()), float(members[lon_col].mean())


def gps_noise_diagnostic(gps, lat_col, lon_col, home):
    """
    Quantify coordinate obfuscation, and distinguish its mechanism.

    Wang et al. (2021) describe Beiwe/HOPES location obfuscation as a random
    displacement from the origin that is unique per participant. If the public
    sample dataset was obfuscated that way, the transform is a single constant
    offset per participant: absolute position is destroyed but all relative
    geometry -- distance, speed, radius of gyration, time at home -- is
    preserved. If instead noise is drawn independently per fix, relative
    geometry is degraded too.

    These two mechanisms are empirically separable. During the presumed-
    stationary window a participant is almost certainly not moving, so
    point-to-point displacement then reflects noise rather than travel:
      - tens of metres  -> consistent with a constant per-participant offset
                           plus ordinary GPS error; mobility features usable
      - hundreds of metres -> consistent with per-fix noise; distance- and
                           speed-based features not interpretable as travel

    Returns a DataFrame with the displacement series and, where the GPS
    stream carries an accuracy column, the device's own reported horizontal
    accuracy for the same fixes. The ratio of the two is the sharpest
    available discriminator: ordinary positioning error produces displacement
    comparable to reported accuracy, whereas injected per-fix noise produces
    displacement far exceeding what the device claims. Do not caption the
    resulting figure until you have looked at this distribution.
    """
    night = gps[in_window(gps["t_local"].dt.hour, STATIONARY_WINDOW)].copy()
    if len(night) < 2:
        return pd.DataFrame(columns=["displacement_m", "reported_accuracy_m"])

    d = haversine_m(
        night[lat_col].to_numpy()[:-1], night[lon_col].to_numpy()[:-1],
        night[lat_col].to_numpy()[1:], night[lon_col].to_numpy()[1:],
    )

    out = pd.DataFrame({"displacement_m": d})
    acc_col = pick_column(night, "accuracy")
    if acc_col is not None:
        acc = pd.to_numeric(night[acc_col], errors="coerce").to_numpy()
        # Pair each displacement with the accuracy of its starting fix.
        out["reported_accuracy_m"] = acc[:-1]
    else:
        out["reported_accuracy_m"] = np.nan
    return out


def daily_gps_features(gps, lat_col, lon_col, home):
    """
    Per-day mobility features.

    time_at_home_h is computed by INTEGRATING over intervals between
    consecutive fixes rather than counting samples, because Beiwe duty-cycles
    the GPS sensor. Intervals longer than MAX_GAP_S are excluded and reported
    as missing coverage instead of being silently extrapolated.
    """
    rows = []
    for date, day in gps.groupby("date"):
        day = day.sort_values("t_local")
        lat = day[lat_col].to_numpy()
        lon = day[lon_col].to_numpy()
        # tz-aware Series.to_numpy() returns object-dtype Timestamps, not
        # datetime64. Casting through datetime64[ns] keeps dt as a real
        # float64 array so later division-by-zero is a numpy warning (which
        # np.errstate can suppress) rather than a Python ZeroDivisionError.
        t = day["t_local"].to_numpy(dtype="datetime64[ns]")

        if len(day) < 2:
            continue

        dt = (t[1:] - t[:-1]) / np.timedelta64(1, "s")
        dt = dt.astype("float64")
        step_m = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])

        # Distance-weighted speed, with implausible segments removed.
        with np.errstate(divide="ignore", invalid="ignore"):
            seg_kmh = np.where(dt > 0, (step_m / dt) * 3.6, np.nan)
        usable = (dt > 0) & (dt <= MAX_GAP_S) & (seg_kmh <= MAX_PLAUSIBLE_SPEED_KMH)

        total_m = float(np.nansum(step_m[usable]))
        total_s = float(np.nansum(dt[usable]))
        mean_kmh = (total_m / total_s) * 3.6 if total_s > 0 else np.nan

        # Time at home: integrate interval durations where the interval START
        # is within the home radius.
        if home is not None:
            at_home = haversine_m(lat, lon, home[0], home[1]) <= HOME_RADIUS_M
            counted = (dt <= MAX_GAP_S)
            home_s = float(np.sum(dt[counted & at_home[:-1]]))
        else:
            home_s = np.nan

        # Radius of gyration about the day's centroid.
        clat, clon = float(np.mean(lat)), float(np.mean(lon))
        rg_m = float(np.sqrt(np.mean(haversine_m(lat, lon, clat, clon) ** 2)))

        rows.append({
            "date": date,
            "n_fixes": len(day),
            "coverage_h": total_s / 3600.0,
            "total_distance_km_raw": float(np.nansum(step_m)) / 1000.0,
            "total_distance_km_filtered": total_m / 1000.0,
            "mean_speed_kmh_weighted": mean_kmh,
            "time_at_home_h": home_s / 3600.0,
            "radius_of_gyration_m": rg_m,
        })

    return pd.DataFrame(rows)


def location_entropy(gps, lat_col, lon_col):
    """Shannon entropy over time spent in significant places (nats)."""
    from sklearn.cluster import DBSCAN

    pts = gps[[lat_col, lon_col]]
    if len(pts) > ENTROPY_MAX_POINTS:
        stride = int(np.ceil(len(pts) / ENTROPY_MAX_POINTS))
        pts = pts.iloc[::stride]
    coords = np.radians(pts.to_numpy())
    eps = HOME_RADIUS_M / 6_371_000.0
    labels = DBSCAN(eps=eps, min_samples=5, metric="haversine",
                     algorithm="ball_tree").fit_predict(coords)
    valid = labels[labels >= 0]
    if valid.size == 0:
        return np.nan, 0
    counts = np.bincount(valid)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p))), int(len(counts))


# ----------------------------------------------------------------------------
# Survey + power_state
# ----------------------------------------------------------------------------

def extract_home_survey(participant_dir):
    """
    Pull the self-reported hours-at-home answers.

    The question text actually delivered is "How much time (in hours) do you
    think you spent AWAY from home yesterday?" (slider, 0-24) -- the inverse
    of what earlier drafts of this pipeline assumed. reported_home_h is
    computed as 24 - reported_away_h.

    Beiwe logs one row per slider interaction (present/changed/unpresent on
    iOS; an undifferentiated sequence of drag values with no event column on
    Android), not one row per submitted answer. Taking every row as a
    separate answer overcounts responses several-fold and mixes in
    transient, not-yet-submitted slider positions. This groups rows into
    delivery sessions (a >10 minute gap starts a new session) and keeps only
    the final value of each session: the 'unpresent' event on iOS, or the
    last drag value chronologically when no event column exists (Android).
    """
    df = load_stream(participant_dir, "survey_timings")
    if df.empty or "question text" not in df.columns:
        return pd.DataFrame(columns=["t_local", "date", "reported_home_h"])

    ans_col = pick_column(df, "answer")
    if ans_col is None:
        return pd.DataFrame(columns=["t_local", "date", "reported_home_h"])

    mask = df["question text"].astype(str).str.contains(
        "away from home", case=False, na=False)
    df = df[mask].copy()
    if df.empty:
        return pd.DataFrame(columns=["t_local", "date", "reported_home_h"])

    df["ans_num"] = pd.to_numeric(df[ans_col], errors="coerce")
    df = df.sort_values("t_local")
    gap = df["t_local"].diff().dt.total_seconds().fillna(9999)
    df["session"] = (gap > 600).cumsum()

    rows = []
    for _, grp in df.groupby("session"):
        final = grp[grp["event"] == "unpresent"] if "event" in grp.columns else pd.DataFrame()
        if final.empty:
            final = grp[grp["ans_num"].notna()].tail(1)
        if final.empty:
            continue
        r = final.iloc[-1]
        if pd.isna(r["ans_num"]):
            continue
        rows.append({"t_local": r["t_local"], "date": r["date"],
                     "reported_home_h": 24.0 - r["ans_num"]})

    return pd.DataFrame(rows, columns=["t_local", "date", "reported_home_h"])


def screen_features(participant_dir):
    """
    Device-usage features from the power_state stream.

    Implements the power-state specification in Wang et al. (2021),
    supplementary section 5.1, item 7: per day, the number of power-down
    signals and the max/min/std/mean duration of each screen-on session,
    plus total seconds with the screen on and total power events. Using their
    published operational definitions rather than inventing thresholds makes
    every feature here citable.

    This stream carries no location information, so it is unaffected by
    coordinate obfuscation.
    """
    df = load_stream(participant_dir, "power_state")
    if df.empty:
        return pd.DataFrame()

    ev_col = pick_column(df, "event")
    if ev_col is None:
        return pd.DataFrame()

    df["ev"] = df[ev_col].astype(str).str.lower()
    on = df["ev"].str.contains("screen turned on|unlocked", regex=True)
    off = df["ev"].str.contains("screen turned off|locked", regex=True)
    powerdown = df["ev"].str.contains("power|shut", regex=True) & ~(on | off)

    events = df[on | off | powerdown].copy()
    events["kind"] = np.where(on[on | off | powerdown], "on",
                      np.where(off[on | off | powerdown], "off", "powerdown"))

    rows = []
    for date, day in events.groupby("date"):
        day = day.sort_values("t_local").reset_index(drop=True)

        # Pair each screen-on with the next screen-off to get sessions.
        sessions = []
        open_at = None
        for _, r in day.iterrows():
            if r["kind"] == "on" and open_at is None:
                open_at = r["t_local"]
            elif r["kind"] == "off" and open_at is not None:
                delta = (r["t_local"] - open_at).total_seconds()
                if 0 < delta <= MAX_GAP_S * 6:
                    sessions.append(delta)
                open_at = None

        s = np.asarray(sessions, dtype=float)
        night = day[in_window(day["t_local"].dt.hour, STATIONARY_WINDOW)]

        rows.append({
            "date": date,
            "screen_on_h": float(s.sum()) / 3600.0 if s.size else 0.0,
            "n_screen_sessions": int(s.size),
            "n_power_events": int(len(day)),
            "n_powerdowns": int((day["kind"] == "powerdown").sum()),
            "session_max_s": float(s.max()) if s.size else np.nan,
            "session_min_s": float(s.min()) if s.size else np.nan,
            "session_mean_s": float(s.mean()) if s.size else np.nan,
            "session_std_s": float(s.std(ddof=1)) if s.size > 1 else np.nan,
            "nocturnal_screen_ons": int((night["kind"] == "on").sum()),
        })

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------

def make_figures(features, noise, validation, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 200, "savefig.dpi": 300, "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    # Figure 1 -- data coverage
    if not features.empty:
        pivot = features.pivot_table(index="participant", columns="date",
                                     values="coverage_h", aggfunc="sum")
        fig, ax = plt.subplots(figsize=(7, 2.6))
        im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis",
                       vmin=0, vmax=24)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(c) for c in pivot.columns], rotation=45,
                           ha="right", fontsize=7)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=7)
        ax.set_title("GPS coverage per participant-day (hours of usable data)")
        fig.colorbar(im, ax=ax, label="hours")
        fig.tight_layout()
        fig.savefig(figdir / "fig1_coverage.png")
        plt.close(fig)

    # Figure 2 -- obfuscation magnitude, with the accuracy comparison that
    # discriminates injected noise from ordinary positioning error.
    if not noise.empty:
        disp = noise["displacement_m"].dropna().to_numpy()
        acc = noise["reported_accuracy_m"].dropna().to_numpy()
        has_acc = acc.size > 0

        fig, axes = plt.subplots(1, 2 if has_acc else 1,
                                 figsize=(8.4 if has_acc else 5, 3.2),
                                 squeeze=False)
        ax = axes[0][0]
        ax.hist(disp, bins=60, color="#3b6ea5", edgecolor="white", linewidth=0.3)
        med = float(np.median(disp))
        ax.axvline(med, color="#b5432f", linestyle="--", linewidth=1.2,
                   label=f"median = {med:.0f} m")
        ax.set_xlabel("Point-to-point displacement, stationary window (m)")
        ax.set_ylabel("Count")
        ax.set_title("Apparent displacement while presumed stationary")
        ax.legend(frameon=False, fontsize=8)

        if has_acc:
            ax2 = axes[0][1]
            n = min(disp.size, acc.size)
            ax2.scatter(acc[:n], disp[:n], s=6, alpha=0.35, color="#3b6ea5",
                        edgecolor="none")
            lim = max(float(np.nanpercentile(acc[:n], 99)),
                      float(np.nanpercentile(disp[:n], 99)), 1.0)
            ax2.plot([0, lim], [0, lim], color="#888", linestyle="--",
                     linewidth=1, label="displacement = reported accuracy")
            ax2.set_xlim(0, lim); ax2.set_ylim(0, lim)
            ax2.set_xlabel("Device-reported horizontal accuracy (m)")
            ax2.set_ylabel("Observed displacement (m)")
            ax2.set_title("Displacement vs. what the device claims")
            ax2.legend(frameon=False, fontsize=7)

        fig.suptitle("Points far above the diagonal indicate injected noise "
                     "rather than ordinary positioning error", fontsize=8.5)
        fig.tight_layout()
        fig.savefig(figdir / "fig2_gps_obfuscation.png")
        plt.close(fig)

    # Figure 3 & 4 -- validation against self-report
    if not validation.empty and validation["reported_home_h"].notna().sum() >= 3:
        v = validation.dropna(subset=["reported_home_h", "time_at_home_h"])
        if len(v) >= 3:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.scatter(v["reported_home_h"], v["time_at_home_h"],
                       s=34, color="#3b6ea5", alpha=0.85, edgecolor="white")
            lim = [0, 24]
            ax.plot(lim, lim, color="#888", linestyle="--", linewidth=1,
                    label="perfect agreement")
            ax.set_xlim(lim); ax.set_ylim(lim)
            ax.set_xlabel("Self-reported hours at home")
            ax.set_ylabel("GPS-derived hours at home")
            ax.set_title("Algorithmic vs. self-reported time at home")
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            fig.savefig(figdir / "fig3_timeathome_scatter.png")
            plt.close(fig)

            mean = (v["reported_home_h"] + v["time_at_home_h"]) / 2
            diff = v["time_at_home_h"] - v["reported_home_h"]
            bias, sd = diff.mean(), diff.std(ddof=1)
            fig, ax = plt.subplots(figsize=(5, 3.4))
            ax.scatter(mean, diff, s=34, color="#3b6ea5", alpha=0.85,
                       edgecolor="white")
            ax.axhline(bias, color="#b5432f", linewidth=1.2,
                       label=f"bias = {bias:+.2f} h")
            ax.axhline(bias + 1.96 * sd, color="#b5432f", linestyle="--",
                       linewidth=1, label="±1.96 SD")
            ax.axhline(bias - 1.96 * sd, color="#b5432f", linestyle="--",
                       linewidth=1)
            ax.set_xlabel("Mean of the two methods (h)")
            ax.set_ylabel("GPS − self-report (h)")
            ax.set_title("Bland–Altman: time at home")
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            fig.savefig(figdir / "fig4_bland_altman.png")
            plt.close(fig)

    # Figure 5 -- per-participant daily features
    if not features.empty:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
        for pid, grp in features.groupby("participant"):
            grp = grp.sort_values("date")
            x = range(len(grp))
            axes[0].plot(x, grp["time_at_home_h"], marker="o", ms=3,
                         linewidth=1, label=pid)
            axes[1].plot(x, grp["radius_of_gyration_m"], marker="o", ms=3,
                         linewidth=1, label=pid)
        axes[0].set_ylabel("Hours at home"); axes[0].set_xlabel("Study day")
        axes[1].set_ylabel("Radius of gyration (m)"); axes[1].set_xlabel("Study day")
        axes[1].legend(frameon=False, fontsize=6, ncol=2)
        fig.suptitle("Daily mobility features by participant")
        fig.tight_layout()
        fig.savefig(figdir / "fig5_daily_features.png")
        plt.close(fig)

    # Figure 6 -- screen-on time
    if "screen_on_h" in features.columns and features["screen_on_h"].notna().any():
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        for pid, grp in features.groupby("participant"):
            grp = grp.sort_values("date")
            ax.plot(range(len(grp)), grp["screen_on_h"], marker="s", ms=3,
                    linewidth=1, label=pid)
        ax.set_xlabel("Study day"); ax.set_ylabel("Screen-on hours")
        ax.set_title("Daily screen-on duration (power_state stream, not obfuscated)")
        ax.legend(frameon=False, fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(figdir / "fig6_screen_on.png")
        plt.close(fig)

    print(f"  figures -> {figdir}")


def make_trace_figure(traces, outdir):
    """
    Figure 7 -- per-participant coordinate traces.

    This is the honest replacement for a 'path walked' plot. Because the
    public dataset's coordinates are obfuscated, a trace plot cannot be
    captioned as travel. It CAN illustrate the obfuscation directly:
    nighttime fixes are drawn separately, so the reader sees apparent
    scatter during hours when the participant was almost certainly
    stationary. Axes are shown as metres of offset from each participant's
    own home centroid, never as absolute coordinates, so no location is
    disclosed.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not traces:
        return
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    n = len(traces)
    ncol = min(3, n)
    nrow = int(math.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.1 * nrow),
                             squeeze=False)

    for ax in axes.flat:
        ax.set_visible(False)

    for idx, (pid, dx, dy, is_night) in enumerate(traces):
        ax = axes[idx // ncol][idx % ncol]
        ax.set_visible(True)
        ax.plot(dx, dy, color="#c9d6e3", linewidth=0.5, zorder=1)
        ax.scatter(dx[~is_night], dy[~is_night], s=3, color="#3b6ea5",
                   alpha=0.6, zorder=2, label="daytime")
        ax.scatter(dx[is_night], dy[is_night], s=3, color="#b5432f",
                   alpha=0.7, zorder=3, label="presumed stationary")
        ax.scatter([0], [0], marker="x", s=45, color="black", zorder=4,
                   label="home centroid")
        ax.set_title(pid, fontsize=8)
        ax.set_xlabel("metres east of home", fontsize=7)
        ax.set_ylabel("metres north of home", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_aspect("equal", adjustable="datalim")
        if idx == 0:
            ax.legend(frameon=False, fontsize=5.5, loc="upper left")

    fig.suptitle("Coordinate traces relative to each participant's home centroid\n"
                 "Red points fall in the 21:00-06:00 window; their "
                 "spread reflects obfuscation, not travel", fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "fig7_obfuscated_traces.png")
    plt.close(fig)
    print(f"  trace figure -> {figdir / 'fig7_obfuscated_traces.png'}")


# ----------------------------------------------------------------------------
# Noise sensitivity sweep
# ----------------------------------------------------------------------------

def _jitter(gps, lat_col, lon_col, sigma_m, rng):
    """
    Return a copy of `gps` with independent Gaussian noise added to each fix.

    Noise is specified in metres and converted to degrees locally, so the
    east-west component is scaled by cos(latitude). Each fix is perturbed
    independently, which is the per-point mechanism from Figure 3 rather
    than a single rigid shift.
    """
    out = gps.copy()
    if sigma_m <= 0:
        return out
    lat = out[lat_col].to_numpy()
    lon = out[lon_col].to_numpy()
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * np.cos(np.radians(np.clip(lat, -89.9, 89.9)))
    m_per_deg_lon = np.where(np.abs(m_per_deg_lon) < 1.0, 1.0, m_per_deg_lon)
    out[lat_col] = lat + rng.normal(0.0, sigma_m, lat.size) / m_per_deg_lat
    out[lon_col] = lon + rng.normal(0.0, sigma_m, lon.size) / m_per_deg_lon
    return out


def _summarise(gps, lat_col, lon_col):
    """Collapse one trace to the scalar features tracked by the sweep."""
    home = locate_home(gps, lat_col, lon_col)
    feats = daily_gps_features(gps, lat_col, lon_col, home)

    sub = gps
    if SENSITIVITY_MAX_POINTS and len(gps) > SENSITIVITY_MAX_POINTS:
        step = int(np.ceil(len(gps) / SENSITIVITY_MAX_POINTS))
        sub = gps.iloc[::step]
    ent, nplaces = location_entropy(sub, lat_col, lon_col)

    if feats.empty:
        return None
    return {
        "home": home,
        "time_at_home_h": float(feats["time_at_home_h"].mean()),
        "radius_of_gyration_m": float(feats["radius_of_gyration_m"].mean()),
        "total_distance_km": float(feats["total_distance_km_filtered"].mean()),
        "mean_speed_kmh": float(feats["mean_speed_kmh_weighted"].mean()),
        "location_entropy": float(ent) if ent == ent else np.nan,
        "n_significant_places": float(nplaces),
    }


def cmd_sensitivity(args):
    """
    Measure how each feature degrades as known noise is added.

    IMPORTANT INTERPRETATION LIMIT. The baseline here is the trace as
    published, which has already been through the platform's own privacy
    protection. This sweep therefore adds noise on top of noise. It
    characterises the SHAPE of degradation -- which features fall apart
    first, and how fast -- not absolute metre thresholds at which a feature
    becomes unusable. Report it that way.
    """
    data_dir = Path(args.data_dir)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    participants = find_participants(data_dir)
    if not participants:
        sys.exit(f"No participant directories found in {data_dir}")

    metrics = ["time_at_home_h", "radius_of_gyration_m", "total_distance_km",
               "mean_speed_kmh", "location_entropy", "n_significant_places"]
    rows = []

    for pdir in participants:
        pid = pdir.name
        gps = load_stream(pdir, "gps")
        if gps.empty:
            print(f"[{pid}] no GPS; skipped")
            continue
        lat_col = pick_column(gps, "lat")
        lon_col = pick_column(gps, "lon")
        if lat_col is None or lon_col is None:
            sys.exit(f"Could not identify lat/lon columns: {list(gps.columns)}")

        base = _summarise(gps, lat_col, lon_col)
        if base is None:
            print(f"[{pid}] too few fixes; skipped")
            continue
        print(f"[{pid}] baseline computed over {len(gps):,} fixes")

        for sigma in SENSITIVITY_SIGMAS_M:
            n_seeds = 1 if sigma == 0 else SENSITIVITY_SEEDS
            for seed in range(n_seeds):
                rng = np.random.default_rng(hash((pid, sigma, seed)) % (2**32))
                res = _summarise(_jitter(gps, lat_col, lon_col, sigma, rng),
                                 lat_col, lon_col)
                if res is None:
                    continue
                row = {"participant": pid, "sigma_m": sigma, "seed": seed}
                for m in metrics:
                    b = base[m]
                    row[m] = res[m]
                    # Ratio to the participant's own zero-noise value.
                    row[m + "_ratio"] = (res[m] / b
                                         if b not in (0, None) and b == b
                                         else np.nan)
                if base["home"] and res["home"]:
                    row["home_displacement_m"] = float(haversine_m(
                        base["home"][0], base["home"][1],
                        res["home"][0], res["home"][1]))
                else:
                    row["home_displacement_m"] = np.nan
                rows.append(row)
            print(f"    sigma {sigma:>6.0f} m  done")

    if not rows:
        sys.exit("Sensitivity sweep produced no rows.")

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "noise_sensitivity.csv", index=False)

    print("\n--- Median feature value as a fraction of its zero-noise value ---")
    hdr = "  sigma_m  " + "".join(f"{m.split('_')[0][:9]:>11}" for m in metrics)
    print(hdr)
    for sigma in SENSITIVITY_SIGMAS_M:
        sl = df[df["sigma_m"] == sigma]
        line = f"  {sigma:>7.0f}  "
        for m in metrics:
            line += f"{sl[m + '_ratio'].median():>11.2f}"
        print(line)

    print("\n--- Median home-estimate displacement from baseline (m) ---")
    for sigma in SENSITIVITY_SIGMAS_M:
        sl = df[df["sigma_m"] == sigma]
        print(f"  {sigma:>7.0f} m noise  ->  "
              f"{sl['home_displacement_m'].median():>8.1f} m")

    _sensitivity_figure(df, metrics, outdir)
    print(f"\nTable -> {outdir}/noise_sensitivity.csv")
    print("Report this as the SHAPE of degradation, not as absolute "
          "thresholds: the baseline trace is already privacy-protected.\n")


def _sensitivity_figure(df, metrics, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 200, "savefig.dpi": 300, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    labels = {
        "time_at_home_h": "Time at home",
        "radius_of_gyration_m": "Radius of gyration",
        "total_distance_km": "Total distance",
        "mean_speed_kmh": "Mean speed",
        "location_entropy": "Location entropy",
        "n_significant_places": "Significant places",
    }
    colors = ["#2E5496", "#2E6B4F", "#B5432F", "#8A6D3B", "#6A4C93", "#3E8FA8"]
    sig = np.array(SENSITIVITY_SIGMAS_M, dtype=float)
    x = np.where(sig == 0, 5.0, sig)  # place the zero point on the log axis

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7),
                             gridspec_kw={"width_ratios": [1.55, 1]})

    ax = axes[0]
    for m, c in zip(metrics, colors):
        med, lo, hi = [], [], []
        for s_ in sig:
            v = df[df["sigma_m"] == s_][m + "_ratio"].dropna()
            med.append(v.median() if len(v) else np.nan)
            lo.append(v.quantile(0.25) if len(v) else np.nan)
            hi.append(v.quantile(0.75) if len(v) else np.nan)
        ax.plot(x, med, marker="o", ms=3.5, linewidth=1.4, color=c,
                label=labels[m])
        ax.fill_between(x, lo, hi, color=c, alpha=0.13, linewidth=0)

    ax.axhline(1.0, color="#999999", linewidth=0.9, linestyle="--")
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([("0" if v == 0 else f"{v:.0f}") for v in sig],
                       fontsize=7)
    ax.set_xlabel("Additional per-point noise added (m, SD)")
    ax.set_ylabel("Feature value \u00F7 its zero-noise value")
    ax.set_title("How each feature drifts as noise is added", fontsize=9.5)
    ax.legend(frameon=False, fontsize=6.8, ncol=2)

    ax2 = axes[1]
    med, lo, hi = [], [], []
    for s_ in sig:
        v = df[df["sigma_m"] == s_]["home_displacement_m"].dropna()
        med.append(v.median() if len(v) else np.nan)
        lo.append(v.quantile(0.25) if len(v) else np.nan)
        hi.append(v.quantile(0.75) if len(v) else np.nan)
    ax2.plot(x, med, marker="s", ms=3.5, linewidth=1.4, color="#B5432F")
    ax2.fill_between(x, lo, hi, color="#B5432F", alpha=0.13, linewidth=0)
    ax2.axhline(HOME_RADIUS_M, color="#999999", linestyle="--", linewidth=0.9,
                label=f"home radius = {HOME_RADIUS_M:.0f} m")
    ax2.set_xscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels([("0" if v == 0 else f"{v:.0f}") for v in sig],
                        fontsize=7)
    ax2.set_xlabel("Additional per-point noise added (m, SD)")
    ax2.set_ylabel("Home estimate moved (m)")
    ax2.set_title("Where the home estimate lands", fontsize=9.5)
    ax2.legend(frameon=False, fontsize=7)

    fig.suptitle("Noise added on top of an already-protected trace: read the "
                 "shape, not absolute thresholds", fontsize=8.4)
    fig.tight_layout()
    fig.savefig(figdir / "fig8_noise_sensitivity.png")
    plt.close(fig)
    print(f"  figure -> {figdir / 'fig8_noise_sensitivity.png'}")


# ----------------------------------------------------------------------------
# Run mode
# ----------------------------------------------------------------------------

def cmd_run(args):
    data_dir = Path(args.data_dir)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    participants = find_participants(data_dir)
    if not participants:
        sys.exit(f"No participant directories found in {data_dir}")

    all_features, all_noise, summary_rows, all_traces = [], [], [], []

    for pdir in participants:
        pid = pdir.name
        print(f"\n[{pid}]")

        gps = load_stream(pdir, "gps")
        if gps.empty:
            print("  no GPS data; skipping mobility features")
            feats = pd.DataFrame()
            home = None
        else:
            lat_col = pick_column(gps, "lat")
            lon_col = pick_column(gps, "lon")
            if lat_col is None or lon_col is None:
                sys.exit(f"  Could not identify lat/lon columns: {list(gps.columns)}")

            home = locate_home(gps, lat_col, lon_col)
            print(f"  GPS fixes: {len(gps):,}   home estimated: "
                  f"{'yes' if home else 'no'}")

            noise = gps_noise_diagnostic(gps, lat_col, lon_col, home)
            if not noise.empty:
                all_noise.append(noise)
                d_ = noise["displacement_m"]
                a_ = noise["reported_accuracy_m"]
                msg = (f"  stationary-window displacement: median "
                       f"{d_.median():.0f} m, 95th pct {d_.quantile(0.95):.0f} m")
                if a_.notna().any():
                    msg += (f"   | device-reported accuracy median "
                            f"{a_.median():.0f} m")
                print(msg)

            # Home-relative offsets in metres, so no absolute location is
            # ever plotted or published.
            if home is not None:
                lat_a = gps[lat_col].to_numpy()
                lon_a = gps[lon_col].to_numpy()
                dy = np.sign(lat_a - home[0]) * haversine_m(
                    home[0], home[1], lat_a, np.full_like(lat_a, home[1]))
                dx = np.sign(lon_a - home[1]) * haversine_m(
                    home[0], home[1], np.full_like(lon_a, home[0]), lon_a)
                hr = gps["t_local"].dt.hour.to_numpy()
                is_night = in_window(hr, STATIONARY_WINDOW)
                all_traces.append((pid, dx, dy, is_night))

            feats = daily_gps_features(gps, lat_col, lon_col, home)
            ent, nplaces = location_entropy(gps, lat_col, lon_col)
            summary_rows.append({
                "participant": pid,
                "n_gps_fixes": len(gps),
                "location_entropy_nats": ent,
                "n_significant_places": nplaces,
            })

        screens = screen_features(pdir)
        if not screens.empty:
            print(f"  power events: {int(screens['n_power_events'].sum()):,}   "
                  f"screen-on sessions: {int(screens['n_screen_sessions'].sum()):,}")
            feats = (screens if feats.empty
                     else feats.merge(screens, on="date", how="outer"))

        survey = extract_home_survey(pdir)
        if not survey.empty:
            print(f"  self-reported home answers: {len(survey)}")
            daily_report = survey.groupby("date", as_index=False)[
                "reported_home_h"].mean()
            feats = (daily_report if feats.empty
                     else feats.merge(daily_report, on="date", how="outer"))

        if not feats.empty:
            feats.insert(0, "participant", pid)
            all_features.append(feats)

    if not all_features:
        sys.exit("No features could be extracted. Re-run `inspect`.")

    features = pd.concat(all_features, ignore_index=True).sort_values(
        ["participant", "date"])
    noise = (pd.concat(all_noise, ignore_index=True) if all_noise
             else pd.DataFrame(columns=["displacement_m",
                                        "reported_accuracy_m"]))

    features.to_csv(outdir / "daily_features.csv", index=False)
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            outdir / "participant_summary.csv", index=False)

    # Validation statistics
    validation = features.copy()
    if {"reported_home_h", "time_at_home_h"}.issubset(validation.columns):
        v = validation.dropna(subset=["reported_home_h", "time_at_home_h"])
        if len(v) >= 3:
            from scipy import stats
            r, p = stats.pearsonr(v["reported_home_h"], v["time_at_home_h"])
            diff = v["time_at_home_h"] - v["reported_home_h"]
            print("\n--- Validation: GPS-derived vs self-reported time at home ---")
            print(f"  paired observations : {len(v)}")
            print(f"  Pearson r           : {r:.3f} (p = {p:.4f})")
            print(f"  mean bias           : {diff.mean():+.2f} h")
            print(f"  mean absolute error : {diff.abs().mean():.2f} h")
            print(f"  95% limits of agree.: {diff.mean() - 1.96*diff.std(ddof=1):+.2f}"
                  f" to {diff.mean() + 1.96*diff.std(ddof=1):+.2f} h")
            with open(outdir / "validation_stats.txt", "w") as fh:
                fh.write(f"n_pairs\t{len(v)}\npearson_r\t{r:.4f}\n"
                         f"p_value\t{p:.4f}\nmean_bias_h\t{diff.mean():.4f}\n"
                         f"mae_h\t{diff.abs().mean():.4f}\n")

            # Coverage-normalized comparison. Raw time_at_home_h is bounded
            # above by coverage_h (median ~3.3 h of a 24 h day), so it can
            # never approach a self-report answer given on a full-day scale
            # even when the algorithm is working correctly -- that mismatch,
            # not measurement error, drives most of the raw bias above.
            # Comparing the fraction of *observed* time spent at home against
            # the fraction of the *day* reported at home puts both numbers on
            # the same 0-1 scale.
            vc = v[v["coverage_h"] > 0].copy()
            vc["prop_home_observed"] = vc["time_at_home_h"] / vc["coverage_h"]
            vc["prop_home_reported"] = vc["reported_home_h"] / 24.0
            if len(vc) >= 3:
                rc, pc = stats.pearsonr(vc["prop_home_observed"],
                                         vc["prop_home_reported"])
                diffc = vc["prop_home_observed"] - vc["prop_home_reported"]
                print("\n--- Validation (coverage-normalized): "
                      "proportion of time at home ---")
                print(f"  paired observations : {len(vc)}")
                print(f"  Pearson r           : {rc:.3f} (p = {pc:.4f})")
                print(f"  mean bias           : {diffc.mean():+.3f} "
                      "(proportion of day)")
                print(f"  mean absolute error : {diffc.abs().mean():.3f} "
                      "(proportion of day)")
                print(f"  95% limits of agree.: "
                      f"{diffc.mean() - 1.96*diffc.std(ddof=1):+.3f} to "
                      f"{diffc.mean() + 1.96*diffc.std(ddof=1):+.3f} "
                      "(proportion of day)")
                with open(outdir / "validation_stats.txt", "a") as fh:
                    fh.write(f"n_pairs_normalized\t{len(vc)}\n"
                             f"pearson_r_normalized\t{rc:.4f}\n"
                             f"p_value_normalized\t{pc:.4f}\n"
                             f"mean_bias_prop\t{diffc.mean():.4f}\n"
                             f"mae_prop\t{diffc.abs().mean():.4f}\n")
        else:
            print("\n  Too few paired observations for validation statistics.")

    make_figures(features, noise, validation, outdir)
    make_trace_figure(all_traces, outdir)
    print(f"\nTables  -> {outdir}/daily_features.csv")
    print("Report every number from these files. Do not round toward a "
          "hypothesis, and report the coverage column alongside any feature.\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("inspect", help="survey directory layout and columns")
    pi.add_argument("--data-dir", required=True)
    pi.set_defaults(func=cmd_inspect)

    pr = sub.add_parser("run", help="extract features and build figures")
    pr.add_argument("--data-dir", required=True)
    pr.add_argument("--out", default="./output")
    pr.set_defaults(func=cmd_run)

    ps = sub.add_parser("sensitivity",
                        help="measure how features degrade as noise is added")
    ps.add_argument("--data-dir", required=True)
    ps.add_argument("--out", default="./output")
    ps.set_defaults(func=cmd_sensitivity)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
