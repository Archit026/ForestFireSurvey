import argparse
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import scipy.ndimage as ndimage
from pyhdf.SD import SD, SDC


@dataclass
class SelectedEvent:
    acq_date: str
    acq_time: int
    latitude: float
    longitude: float
    confidence: int
    frp: float
    daynight: str


@dataclass
class GranuleSelection:
    file_img2_prefix: str
    file_img1_prefix: str
    fire_hhmm: str
    is_daytime: bool
    bbox: Tuple[float, float, float, float]
    fire_dt: datetime
    ref_dt: datetime


FLARE_ZONES = [
    dict(lat_min=28.5, lat_max=33.0, lon_min=44.5, lon_max=50.5),
    dict(lat_min=24.0, lat_max=29.5, lon_min=47.0, lon_max=53.5),
    dict(lat_min=21.0, lat_max=26.5, lon_min=53.0, lon_max=60.0),
    dict(lat_min=30.0, lat_max=33.5, lon_min=48.0, lon_max=52.0),
    dict(lat_min=26.0, lat_max=33.0, lon_min=5.0, lon_max=16.0),
    dict(lat_min=4.0, lat_max=8.0, lon_min=4.0, lon_max=10.0),
]

CANDIDATE_ZONES = [
    dict(lat_min=35.5, lat_max=38.5, lon_min=48.0, lon_max=57.0, name="N.Iran Caspian forests"),
    dict(lat_min=38.0, lat_max=43.5, lon_min=40.0, lon_max=53.0, name="Caucasus forests"),
    dict(lat_min=36.0, lat_max=42.0, lon_min=26.0, lon_max=44.5, name="Turkey forests"),
    dict(lat_min=37.0, lat_max=44.0, lon_min=55.0, lon_max=75.0, name="Central Asia"),
    dict(lat_min=31.5, lat_max=37.5, lon_min=34.0, lon_max=42.0, name="E.Mediterranean forests"),
]


def find_firms_csv(search_roots: list[str]) -> str:
    for root in search_roots:
        if not os.path.exists(root):
            continue
        for dirpath, _, files in os.walk(root):
            for filename in files:
                if "M-C61" in filename and filename.endswith(".csv"):
                    return os.path.join(dirpath, filename)
    raise FileNotFoundError("fire_nrt_M-C61_*.csv not found in configured search roots")


def select_best_event(csv_path: str, min_confidence: int = 60, min_frp: float = 20.0) -> SelectedEvent:
    df = pd.read_csv(csv_path)
    terra = df[(df["satellite"] == "Terra") & (df["confidence"] >= min_confidence)].copy()

    for zone in FLARE_ZONES:
        terra = terra[
            ~(
                (terra["latitude"] >= zone["lat_min"])
                & (terra["latitude"] <= zone["lat_max"])
                & (terra["longitude"] >= zone["lon_min"])
                & (terra["longitude"] <= zone["lon_max"])
            )
        ]

    terra["score"] = terra["confidence"] * np.log10(terra["frp"].clip(lower=0.1) + 1)

    top = None
    for zone in CANDIDATE_ZONES:
        subset = terra[
            (terra["latitude"] >= zone["lat_min"])
            & (terra["latitude"] <= zone["lat_max"])
            & (terra["longitude"] >= zone["lon_min"])
            & (terra["longitude"] <= zone["lon_max"])
        ]
        day_sub = subset[(subset["daynight"] == "D") & (subset["frp"] >= min_frp)]
        if len(day_sub) >= 1:
            top = day_sub.sort_values("score", ascending=False).head(20).reset_index(drop=True)
            break

    if top is None:
        day_all = terra[(terra["daynight"] == "D") & (terra["frp"] >= 5.0)]
        if len(day_all) >= 1:
            top = day_all.sort_values("score", ascending=False).head(20).reset_index(drop=True)
        else:
            top = terra.sort_values("score", ascending=False).head(20).reset_index(drop=True)

    daytime_top = top[top["daynight"] == "D"]
    best = daytime_top.iloc[0] if len(daytime_top) > 0 else top.iloc[0]

    return SelectedEvent(
        acq_date=best["acq_date"],
        acq_time=int(best["acq_time"]),
        latitude=float(best["latitude"]),
        longitude=float(best["longitude"]),
        confidence=int(best["confidence"]),
        frp=float(best["frp"]),
        daynight=str(best["daynight"]),
    )


def compute_granules(event: SelectedEvent) -> GranuleSelection:
    fire_dt = datetime.strptime(event.acq_date, "%Y-%m-%d")
    fire_doy = fire_dt.timetuple().tm_yday

    acq_hh, acq_mm = divmod(event.acq_time, 100)
    gran_mm = (acq_mm // 5) * 5
    fire_hhmm = f"{acq_hh:02d}{gran_mm:02d}"

    ref_dt = fire_dt - timedelta(days=1)
    ref_doy = ref_dt.timetuple().tm_yday

    file_img2_prefix = f"MOD021KM.A{fire_dt.year}{fire_doy:03d}.{fire_hhmm}.061"
    file_img1_prefix = f"MOD021KM.A{ref_dt.year}{ref_doy:03d}.{fire_hhmm}.061"

    return GranuleSelection(
        file_img2_prefix=file_img2_prefix,
        file_img1_prefix=file_img1_prefix,
        fire_hhmm=fire_hhmm,
        is_daytime=(event.daynight == "D"),
        bbox=(event.longitude - 3, event.latitude - 3, event.longitude + 3, event.latitude + 3),
        fire_dt=fire_dt,
        ref_dt=ref_dt,
    )


def find_hdf_by_prefix(prefix: str, search_roots: list[str]) -> Optional[str]:
    pattern = re.compile(rf"^{re.escape(prefix)}\.[0-9]{{13}}\.hdf$")
    for root in search_roots:
        if not os.path.exists(root):
            continue
        for dirpath, _, files in os.walk(root):
            for filename in files:
                if pattern.match(filename):
                    return os.path.join(dirpath, filename)
    return None


def ladsweb_download(year: int, doy: int, hhmm: str, download_dir: str, session: requests.Session) -> Optional[str]:
    dir_url = f"https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MOD021KM/{year}/{doy:03d}/"
    prefix = f"MOD021KM.A{year}{doy:03d}.{hhmm}.061"
    resp = session.get(dir_url, timeout=30)
    if resp.status_code != 200:
        return None

    matches = re.findall(rf"({re.escape(prefix)}\.[0-9]{{13}}\.hdf)", resp.text)
    if not matches:
        return None

    filename = matches[0]
    local_path = os.path.join(download_dir, filename)
    if os.path.exists(local_path):
        return local_path

    file_url = dir_url + filename
    dl = session.get(file_url, stream=True, timeout=600)
    if dl.status_code != 200:
        return None

    with open(local_path, "wb") as handle:
        for chunk in dl.iter_content(chunk_size=4 * 1024 * 1024):
            handle.write(chunk)
    return local_path


def maybe_download_granules(granules: GranuleSelection, download_dir: str) -> Tuple[Optional[str], Optional[str]]:
    import earthaccess

    earthaccess.login(strategy="interactive")
    session = earthaccess.get_requests_https_session()
    os.makedirs(download_dir, exist_ok=True)

    fire_path = ladsweb_download(
        granules.fire_dt.year,
        granules.fire_dt.timetuple().tm_yday,
        granules.fire_hhmm,
        download_dir,
        session,
    )
    ref_path = ladsweb_download(
        granules.ref_dt.year,
        granules.ref_dt.timetuple().tm_yday,
        granules.fire_hhmm,
        download_dir,
        session,
    )
    return fire_path, ref_path


def radiance_to_brightness_temp(radiance: np.ndarray, wavelength_um: float) -> np.ndarray:
    c1 = 1.19104e8
    c2 = 1.43878e4
    with np.errstate(divide="ignore", invalid="ignore"):
        bt = c2 / (wavelength_um * np.log(c1 / (wavelength_um**5 * radiance) + 1))
    return np.where(np.isfinite(bt), bt, np.nan)


def load_modis_bands(hdf_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hdf = SD(hdf_path, SDC.READ)

    band_ids = hdf.select("Band_1KM_Emissive")[:]
    idx22 = int(np.where(band_ids == 22)[0][0])
    idx31 = int(np.where(band_ids == 31)[0][0])

    emissive = hdf.select("EV_1KM_Emissive")
    attrs = emissive.attributes()
    raw = emissive[:].astype(np.float32)
    fill_val = attrs.get("_FillValue", 65535)

    r22 = np.where(raw[idx22] == fill_val, np.nan, raw[idx22])
    rad22 = (r22 - attrs["radiance_offsets"][idx22]) * attrs["radiance_scales"][idx22]
    t39 = radiance_to_brightness_temp(rad22, 3.959)

    r31 = np.where(raw[idx31] == fill_val, np.nan, raw[idx31])
    rad31 = (r31 - attrs["radiance_offsets"][idx31]) * attrs["radiance_scales"][idx31]
    t11 = radiance_to_brightness_temp(rad31, 11.03)

    ref = hdf.select("EV_250_Aggr1km_RefSB")
    attrs2 = ref.attributes()
    raw2 = ref[:].astype(np.float32)
    fill2 = attrs2.get("_FillValue", 65535)

    red_raw = np.where(raw2[0] == fill2, np.nan, raw2[0])
    red = (red_raw - attrs2["reflectance_offsets"][0]) * attrs2["reflectance_scales"][0]

    nir_raw = np.where(raw2[1] == fill2, np.nan, raw2[1])
    nir = (nir_raw - attrs2["reflectance_offsets"][1]) * attrs2["reflectance_scales"][1]

    hdf.end()
    return t39, t11, red, nir


def fill_nan(arr: np.ndarray, fill_value: float) -> np.ndarray:
    return np.where(np.isnan(arr), fill_value, arr)


def calculate_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    denominator = nir + red
    denominator = np.where(denominator == 0, 1e-6, denominator)
    return (nir - red) / denominator


def get_water_mask(red: np.ndarray, nir: np.ndarray, ndvi_threshold: float = 0.1) -> np.ndarray:
    ndvi = calculate_ndvi(red, nir)
    return ndvi <= ndvi_threshold


def remove_boundary_false_alarms(water_mask: np.ndarray) -> np.ndarray:
    dilated = ndimage.binary_dilation(water_mask)
    closed = ndimage.binary_closing(dilated)
    return closed


def get_change_mask(t39_img1: np.ndarray, t39_img2: np.ndarray) -> np.ndarray:
    mean_t1 = np.nanmean(t39_img1)
    mean_t2 = np.nanmean(t39_img2)
    threshold = (mean_t2 - mean_t1) / 3.0
    return (t39_img2 - t39_img1) > threshold


def potential_fire_pixels(t39: np.ndarray, t11: np.ndarray, ref_86: np.ndarray) -> np.ndarray:
    mean_t39_col = np.nanmean(t39, axis=0)
    mean_dt_col = np.nanmean(t39 - t11, axis=0)
    t_d1 = np.tile(mean_t39_col + 5.0, (t39.shape[0], 1))
    t_d2 = np.tile(mean_dt_col + 5.0, (t39.shape[0], 1))
    return (t39 > t_d1) & ((t39 - t11) > t_d2) & (ref_86 < 0.35)


def absolute_fire_test(t39: np.ndarray, is_daytime: bool) -> np.ndarray:
    return t39 > (360.0 if is_daytime else 320.0)


def relative_fire_test(
    t39: np.ndarray,
    t11: np.ndarray,
    potential_fires: np.ndarray,
    invalid_pixels_mask: np.ndarray,
    is_daytime: bool,
) -> np.ndarray:
    rows, cols = t39.shape
    active_fires = np.zeros_like(t39, dtype=bool)

    if is_daytime:
        too_hot_bg = (t39 > 315.0) | ((t39 - t11) > 19.5)
    else:
        too_hot_bg = (t39 > 305.0) | ((t39 - t11) > 9.5)

    full_invalid_bg = too_hot_bg | invalid_pixels_mask
    delta_t = t39 - t11

    pot_r, pot_c = np.where(potential_fires)
    for r, c in zip(pot_r, pot_c):
        window_size = 3
        valid_bg_values_t39 = []
        valid_bg_values_t11 = []
        valid_bg_values_dt = []

        while window_size <= 10:
            half_w = window_size // 2
            r_start = max(0, r - half_w)
            r_end = min(rows, r + half_w + 1)
            c_start = max(0, c - half_w)
            c_end = min(cols, c + half_w + 1)

            win_t39 = t39[r_start:r_end, c_start:c_end]
            win_t11 = t11[r_start:r_end, c_start:c_end]
            win_dt = delta_t[r_start:r_end, c_start:c_end]
            win_invalid = full_invalid_bg[r_start:r_end, c_start:c_end]
            valid_mask = ~win_invalid
            valid_mask[r - r_start, c - c_start] = False

            if np.sum(valid_mask) >= 4:
                valid_bg_values_t39 = win_t39[valid_mask]
                valid_bg_values_t11 = win_t11[valid_mask]
                valid_bg_values_dt = win_dt[valid_mask]
                break

            window_size += 2

        if len(valid_bg_values_t39) >= 4:
            mean_dt_bg = np.mean(valid_bg_values_dt)
            std_dt_bg = np.std(valid_bg_values_dt)
            mean_t39_bg = np.mean(valid_bg_values_t39)
            std_t39_bg = np.std(valid_bg_values_t39)
            mean_t11_bg = np.mean(valid_bg_values_t11)
            std_t11_bg = np.std(valid_bg_values_t11)

            dt_val = delta_t[r, c]
            t39_val = t39[r, c]
            t11_val = t11[r, c]

            test_8 = dt_val > (mean_dt_bg + 3.5 * std_dt_bg)
            test_9 = dt_val > (mean_dt_bg + 6.0)
            test_10 = t39_val > (mean_t39_bg + 3.0 * std_t39_bg)
            test_11 = t11_val > (mean_t11_bg + std_t11_bg)
            test_12 = std_t11_bg > 5.0

            if is_daytime and (test_8 and test_9 and test_10) and (test_11 or test_12):
                active_fires[r, c] = True
            if (not is_daytime) and (test_8 and test_9 and test_10):
                active_fires[r, c] = True

    return active_fires


def detect_fires(
    t39_img1: np.ndarray,
    t39_img2: np.ndarray,
    t11_img2: np.ndarray,
    red_img2: np.ndarray,
    nir_img2: np.ndarray,
    ref_86_img2: np.ndarray,
    is_daytime: bool,
) -> np.ndarray:
    water_mask = get_water_mask(red_img2, nir_img2, ndvi_threshold=0.1)
    boundary_mask = remove_boundary_false_alarms(water_mask)
    combined_mask = water_mask | boundary_mask
    temporal_mask = get_change_mask(t39_img1, t39_img2)

    potential_fires = potential_fire_pixels(t39_img2, t11_img2, ref_86_img2)
    refined_potentials = potential_fires & temporal_mask & (~combined_mask)
    absolute_fires = absolute_fire_test(t39_img2, is_daytime) & temporal_mask & (~combined_mask)
    remaining_potentials = refined_potentials & (~absolute_fires)

    relative_fires = relative_fire_test(
        t39=t39_img2,
        t11=t11_img2,
        potential_fires=remaining_potentials,
        invalid_pixels_mask=combined_mask,
        is_daytime=is_daytime,
    )
    return absolute_fires | relative_fires


def harmonize_and_prepare(
    t39_img1: np.ndarray,
    t39_img2: np.ndarray,
    t11_img2: np.ndarray,
    red_img2: np.ndarray,
    nir_img2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    min_rows = min(t39_img1.shape[0], t39_img2.shape[0])
    min_cols = min(t39_img1.shape[1], t39_img2.shape[1])

    t39_img1 = t39_img1[:min_rows, :min_cols]
    t39_img2 = t39_img2[:min_rows, :min_cols]
    t11_img2 = t11_img2[:min_rows, :min_cols]
    red_img2 = red_img2[:min_rows, :min_cols]
    nir_img2 = nir_img2[:min_rows, :min_cols]

    return (
        fill_nan(t39_img1, 200.0),
        fill_nan(t39_img2, 200.0),
        fill_nan(t11_img2, 200.0),
        fill_nan(red_img2, 0.0),
        fill_nan(nir_img2, 0.0),
    )


def plot_result(t11_img2: np.ndarray, fire_mask: np.ndarray, title: str, output_path: str, is_daytime: bool) -> None:
    vmin_t11 = 280 if is_daytime else 270
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    im = axes[0].imshow(t11_img2, cmap="inferno", vmin=vmin_t11, vmax=340)
    fig.colorbar(im, ax=axes[0], label="Brightness Temperature (K)")
    fire_r, fire_c = np.where(fire_mask)
    axes[0].scatter(fire_c, fire_r, c="cyan", s=6, label=f"{len(fire_r)} fire pixels")
    axes[0].set_title(f"T11 Background + Fire Pixels ({len(fire_r)} detected)")
    axes[0].legend(loc="upper right")

    axes[1].imshow(fire_mask, cmap="hot")
    axes[1].set_title("Fire Pixel Mask")

    plt.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def resolve_input_hdfs(
    granules: GranuleSelection,
    search_roots: list[str],
    download_if_missing: bool,
    download_dir: str,
) -> Tuple[str, str]:
    hdf_img2 = find_hdf_by_prefix(granules.file_img2_prefix, search_roots)
    hdf_img1 = find_hdf_by_prefix(granules.file_img1_prefix, search_roots)

    if (hdf_img1 is None or hdf_img2 is None) and download_if_missing:
        downloaded_fire, downloaded_ref = maybe_download_granules(granules, download_dir)
        if hdf_img2 is None:
            hdf_img2 = downloaded_fire
        if hdf_img1 is None:
            hdf_img1 = downloaded_ref if downloaded_ref is not None else downloaded_fire

    if hdf_img1 is None or hdf_img2 is None:
        raise FileNotFoundError(
            "Missing required MOD021KM HDF files. Provide local files matching granule prefixes "
            "or use --download-if-missing."
        )

    return hdf_img1, hdf_img2


def run_pipeline(args: argparse.Namespace) -> None:
    csv_path = args.csv_path or find_firms_csv(args.csv_search_roots)
    event = select_best_event(csv_path=csv_path, min_confidence=args.min_confidence, min_frp=args.min_frp)
    granules = compute_granules(event)

    hdf_img1, hdf_img2 = resolve_input_hdfs(
        granules=granules,
        search_roots=args.hdf_search_roots,
        download_if_missing=args.download_if_missing,
        download_dir=args.download_dir,
    )

    t39_img1, _, _, _ = load_modis_bands(hdf_img1)
    t39_img2, t11_img2, red_img2, nir_img2 = load_modis_bands(hdf_img2)

    t39_img1_in, t39_img2_in, t11_img2_in, red_img2_in, nir_img2_in = harmonize_and_prepare(
        t39_img1=t39_img1,
        t39_img2=t39_img2,
        t11_img2=t11_img2,
        red_img2=red_img2,
        nir_img2=nir_img2,
    )

    fire_mask = detect_fires(
        t39_img1=t39_img1_in,
        t39_img2=t39_img2_in,
        t11_img2=t11_img2_in,
        red_img2=red_img2_in,
        nir_img2=nir_img2_in,
        ref_86_img2=nir_img2_in,
        is_daytime=granules.is_daytime,
    )

    total_fires = int(np.sum(fire_mask))
    mean_t1 = float(np.nanmean(t39_img1_in))
    mean_t2 = float(np.nanmean(t39_img2_in))
    t_d = (mean_t2 - mean_t1) / 3.0
    temporal_hits = int(np.sum((t39_img2_in - t39_img1_in) > t_d))
    abs_threshold = 360.0 if granules.is_daytime else 320.0
    absolute_hits = int(np.sum(t39_img2_in > abs_threshold))

    print("Selected event:")
    print(
        f"  {event.acq_date} {event.acq_time:04d} UTC | "
        f"lat={event.latitude:.3f} lon={event.longitude:.3f} | "
        f"conf={event.confidence} FRP={event.frp:.1f} MW | "
        f"{'Daytime' if granules.is_daytime else 'Nighttime'}"
    )
    print("Resolved granules:")
    print(f"  IMG1: {hdf_img1}")
    print(f"  IMG2: {hdf_img2}")
    print("Detection summary:")
    print(f"  Total fire pixels detected: {total_fires}")
    print(f"  Temporal-change mask pixels: {temporal_hits}")
    print(f"  Absolute threshold hits (>{abs_threshold:.0f} K): {absolute_hits}")

    if args.save_plot:
        title = (
            "MODIS Fire Detection\n"
            f"{Path(hdf_img2).name}\n"
            f"{event.acq_date} {granules.fire_hhmm[:2]}:{granules.fire_hhmm[2:]} UTC | "
            f"{'Daytime' if granules.is_daytime else 'Nighttime'} | "
            f"conf={event.confidence}% FRP={event.frp:.1f} MW"
        )
        plot_result(
            t11_img2=t11_img2_in,
            fire_mask=fire_mask,
            title=title,
            output_path=args.plot_path,
            is_daytime=granules.is_daytime,
        )
        print(f"Saved plot: {args.plot_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MODIS fire detection pipeline converted from notebook")
    parser.add_argument("--csv-path", type=str, default=None, help="Path to FIRMS MODIS M-C61 CSV")
    parser.add_argument("--min-confidence", type=int, default=60, help="Minimum FIRMS confidence for Terra events")
    parser.add_argument("--min-frp", type=float, default=20.0, help="Minimum FRP (MW) for preferred daytime candidate")
    parser.add_argument(
        "--csv-search-roots",
        nargs="+",
        default=["/kaggle/input", "/kaggle/working", "/dataset", "dataset", "."],
        help="Directories to search for FIRMS CSV",
    )
    parser.add_argument(
        "--hdf-search-roots",
        nargs="+",
        default=["/kaggle/input", "/kaggle/working", "./", "."],
        help="Directories to search for MOD021KM HDF granules",
    )
    parser.add_argument("--download-if-missing", action="store_true", help="Try Earthdata/LAADS download when HDF files are missing")
    parser.add_argument("--download-dir", type=str, default="./modis", help="Download directory for HDF files")
    parser.add_argument("--save-plot", action="store_true", help="Save output visualization")
    parser.add_argument("--plot-path", type=str, default="./fire_detection_result.png", help="Output image path")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_pipeline(args)


def run(dataset_path):
    """Standard pipeline interface for MODIS satellite fire detection.
    dataset_path: directory that contains the FIRMS CSV and HDF granule files.
    This is a physics-based, pixel-level detection algorithm — no classification
    probabilities exist, so accuracy/AUC/AUPR are returned as None.
    """
    search_roots = [dataset_path] if (dataset_path and os.path.exists(dataset_path)) else ["."]

    try:
        csv_path = find_firms_csv(search_roots)
    except FileNotFoundError as exc:
        return {"model_name": "MODIS-Satellite", "error": str(exc), "metrics": None}

    try:
        event    = select_best_event(csv_path=csv_path)
        granules = compute_granules(event)
        hdf_img1 = find_hdf_by_prefix(granules.file_img1_prefix, search_roots)
        hdf_img2 = find_hdf_by_prefix(granules.file_img2_prefix, search_roots)

        if hdf_img1 is None or hdf_img2 is None:
            return {"model_name": "MODIS-Satellite",
                    "error": "Required HDF granule files not found in dataset_path",
                    "metrics": None}

        t39_1, _, _, _                     = load_modis_bands(hdf_img1)
        t39_2, t11_2, red_2, nir_2         = load_modis_bands(hdf_img2)
        t39_1, t39_2, t11_2, red_2, nir_2  = harmonize_and_prepare(t39_1, t39_2, t11_2, red_2, nir_2)
        fire_mask = detect_fires(t39_1, t39_2, t11_2, red_2, nir_2, nir_2, granules.is_daytime)

        return {
            "model_name": "MODIS-Satellite",
            "note": f"Fire pixels detected: {int(np.sum(fire_mask))} "
                    "(pixel-level algorithm — image-level GT unavailable)",
            "metrics": {
                "accuracy":  None,
                "precision": None,
                "recall":    None,
                "f1":        None,
                "auc":       None,
                "aupr":      None,
            },
        }
    except Exception as exc:
        return {"model_name": "MODIS-Satellite", "error": str(exc), "metrics": None}


if __name__ == "__main__":
    main()
