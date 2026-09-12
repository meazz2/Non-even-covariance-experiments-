from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TIMESTAMPS = {"wind": "2000-01-02 12:00:00", "edas": "2025-07-01 03:00:00"}
RUN_NAMES = {"wind": "run_20260912T133905_102865Z", "edas": "run_20260912T140023_586087Z"}
VALUE_COLUMNS = ["u_std", "v_std"]
COORDINATE_COLUMNS = ["latitude", "longitude"]
N_BINS = 12
MAX_DISTANCE_QUANTILE = 0.95
MIN_PAIRS = 8
DIRECTION_TOLERANCE_RADIANS = np.pi / 12
DIRECTIONS_RADIANS = np.arange(8) * np.pi / 4


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_slice(path, timestamp):
    """Mirror the relevant MLE loader semantics, returning km coordinates."""
    frame = pd.read_csv(path)
    required = ["datetime"] + COORDINATE_COLUMNS + VALUE_COLUMNS
    if any(column not in frame for column in required):
        raise ValueError("CSV does not contain all required columns")
    frame = frame.loc[:, required].copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True, errors="raise")
    for column in COORDINATE_COLUMNS + VALUE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    if not np.isfinite(frame[COORDINATE_COLUMNS].to_numpy()).all():
        raise ValueError("Invalid coordinates")
    if (frame.latitude.abs() > 90).any():
        raise ValueError("Invalid latitude")
    frame["longitude"] = (frame.longitude + 180) % 360 - 180
    if frame.duplicated(["datetime"] + COORDINATE_COLUMNS).any():
        raise ValueError("Duplicate timestamp/location keys")
    grid = frame[COORDINATE_COLUMNS].drop_duplicates().sort_values(COORDINATE_COLUMNS).reset_index(drop=True)
    index = pd.MultiIndex.from_frame(grid)
    requested_time = pd.Timestamp(timestamp, tz="UTC")
    chosen = frame.loc[frame.datetime == requested_time]
    if len(chosen) != len(grid) or len(grid) != 144:
        raise ValueError("Expected exactly 144 complete locations at the requested timestamp")
    values = chosen.set_index(COORDINATE_COLUMNS).reindex(index)[VALUE_COLUMNS].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Selected slice is incomplete or nonfinite")
    latitude, longitude = grid.to_numpy(float).T
    lat_center = float(latitude.mean())
    radians = np.deg2rad(longitude)
    lon_center = float(np.rad2deg(np.angle(complex(np.cos(radians).mean(), np.sin(radians).mean()))))
    lon_offset = (longitude - lon_center + 180) % 360 - 180
    coords_km = np.column_stack([111 * np.cos(np.deg2rad(lat_center)) * lon_offset, 111 * (latitude - lat_center)])
    selected = grid.copy()
    selected.insert(0, "datetime", timestamp)
    for j, column in enumerate(VALUE_COLUMNS):
        selected[column] = values[:, j]
    selected["x_km"], selected["y_km"] = coords_km.T
    selected["u_centered"], selected["v_centered"] = (values - values.mean(axis=0)).T
    metadata = {
        "source_path": str(Path(path).resolve()), "source_sha256": sha256(path),
        "source_rows": len(frame), "available_timestamps": int(frame.datetime.nunique()),
        "timestamp": requested_time.isoformat(), "timestamp_convention": "UTC; naive input interpreted as UTC",
        "n_locations": len(grid), "n_replicates": 1, "value_columns": VALUE_COLUMNS,
        "original_coordinate_points": grid.to_numpy(float).tolist(),
        "supplied_component_means": values.mean(axis=0).tolist(),
        "supplied_component_standard_deviations_ddof0": values.std(axis=0, ddof=0).tolist(),
        "coordinate_reference_length_km_in_mle": 100.0,
        "projection": {"method": "local equirectangular approximation, 111 km/degree",
            "latitude_center_degrees": lat_center, "longitude_center_degrees": lon_center,
            "longitude_span_degrees": float(np.ptp(lon_offset)), "latitude_span_degrees": float(np.ptp(latitude)),
            "broad_extent_approximation": bool(np.ptp(lon_offset) > 30 or np.ptp(latitude) > 30)},
        "mle_values_centered_or_standardized_by_loader": False,
        "empirical_covariance_centering": "Subtract each component's spatial mean within the selected timestamp",
        "empirical_additional_sd_scaling": False,
        "semivariogram": "Mean squared pair difference divided by two",
    }
    return selected, coords_km, values, metadata


def marginal_diagnostics(coords_km, values):
    centered = values - values.mean(axis=0)
    i, j = np.triu_indices(len(values), k=1)
    distance = np.linalg.norm(coords_km[j] - coords_km[i], axis=1)
    maximum = float(np.quantile(distance, MAX_DISTANCE_QUANTILE))
    edges = np.linspace(0, maximum, N_BINS + 1)
    # The right endpoint is retained in the final bin.
    bins = np.minimum(np.searchsorted(edges, distance, side="right") - 1, N_BINS - 1)
    rows = []
    for b in range(N_BINS):
        keep = (bins == b) & (distance <= maximum)
        if int(keep.sum()) < MIN_PAIRS:
            continue
        row = {"bin": b + 1, "lower_km": edges[b], "upper_km": edges[b + 1],
               "mean_distance_km": distance[keep].mean(), "pair_count": int(keep.sum())}
        for k, column in enumerate(VALUE_COLUMNS):
            row["covariance_" + column] = np.mean(centered[i[keep], k] * centered[j[keep], k])
            row["semivariogram_" + column] = 0.5 * np.mean((values[i[keep], k] - values[j[keep], k]) ** 2)
        rows.append(row)
    return pd.DataFrame(rows), edges


def directional_diagnostics(coords_km, values, edges):
    centered = values - values.mean(axis=0)
    i, j = np.where(~np.eye(len(values), dtype=bool))
    delta = coords_km[j] - coords_km[i]
    distance = np.linalg.norm(delta, axis=1)
    bearing = np.arctan2(delta[:, 1], delta[:, 0]) % (2 * np.pi)
    bins = np.minimum(np.searchsorted(edges, distance, side="right") - 1, N_BINS - 1)
    cross = centered[i, 0] * centered[j, 1]
    rows, scores = [], []
    for angle in DIRECTIONS_RADIANS:
        positive = abs((bearing - angle + np.pi) % (2 * np.pi) - np.pi) <= DIRECTION_TOLERANCE_RADIANS + 1e-12
        negative = abs((bearing - (angle + np.pi) + np.pi) % (2 * np.pi) - np.pi) <= DIRECTION_TOLERANCE_RADIANS + 1e-12
        angle_rows = []
        for b in range(N_BINS):
            within = (bins == b) & (distance <= edges[-1])
            pos, neg = within & positive, within & negative
            if min(int(pos.sum()), int(neg.sum())) < MIN_PAIRS:
                continue
            cplus, cminus = float(cross[pos].mean()), float(cross[neg].mean())
            row = {"direction_radians": float(angle), "bin": b + 1, "mean_distance_km": float(distance[pos | neg].mean()),
                   "positive_pair_count": int(pos.sum()), "negative_pair_count": int(neg.sum()),
                   "cross_covariance_plus": cplus, "cross_covariance_minus": cminus,
                   "even_component": 0.5 * (cplus + cminus), "odd_component": 0.5 * (cplus - cminus)}
            rows.append(row)
            angle_rows.append(row)
        odds = np.array([row["odd_component"] for row in angle_rows])
        if not len(odds):
            raise ValueError("No eligible directional lag bins")
        scores.append({"direction_radians": float(angle), "retained_bins": len(odds),
                       "mean_absolute_odd_component": float(np.mean(np.abs(odds))),
                       "maximum_absolute_odd_component": float(np.max(np.abs(odds))),
                       "mean_absolute_opposite_lag_discrepancy": float(2 * np.mean(np.abs(odds)))})
    return pd.DataFrame(rows), pd.DataFrame(scores)


def plot_marginals(summary, dataset, timestamp, output_dir):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
        "axes.titlesize": 12, "axes.labelsize": 12, "legend.fontsize": 12,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "pdf.fonttype": 42, "ps.fonttype": 42})
    title = "Wind: U and V" if dataset == "wind" else "EDAS: ozone and nitrogen dioxide"
    labels = ("U", "V") if dataset == "wind" else (r"O$_3$", r"NO$_2$")
    styles = [("#1876ad", "o", "-"), ("#cf5921", "s", "--")]
    for measure, filename, ylabel in [
            ("covariance", "marginal_empirical_covariance", "Empirical covariance"),
            ("semivariogram", "marginal_empirical_variograms", "Empirical semivariogram")]:
        fig, ax = plt.subplots(figsize=(5.6, 4.3), constrained_layout=True)
        for column, label, (color, marker, linestyle) in zip(VALUE_COLUMNS, labels, styles):
            ax.plot(summary.mean_distance_km, summary[measure + "_" + column], label=label,
                    color=color, marker=marker, linestyle=linestyle, markersize=4.5, linewidth=1.8)
        ax.axhline(0, color="#747474", linewidth=0.8, zorder=0)
        ax.set(xlabel="Distance (km)", ylabel=ylabel, title=title + "\n" + timestamp + " UTC")
        ax.set_xlim(left=0)
        ax.grid(axis="both", color="#dddddd", linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=True, facecolor="white", framealpha=0.95, edgecolor="none")
        if measure == "semivariogram":
            ax.set_ylim(bottom=0)
        for extension in ("png", "pdf"):
            fig.savefig(output_dir / (filename + "." + extension), dpi=300, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)



def plot_direction_scores(scores, dataset, timestamp, output_dir):
    """A diagnostic criterion, not an optimized likelihood comparison."""
    fig, ax = plt.subplots(figsize=(6.8, 7.0), subplot_kw={"projection": "polar"})
    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.12, top=0.78)
    angles = scores.direction_radians.to_numpy()
    magnitude = scores.mean_absolute_odd_component.to_numpy()
    ax.plot(np.r_[angles, angles[0]], np.r_[magnitude, magnitude[0]], color="#1876ad", marker="o", linewidth=1.7)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_xticks(DIRECTIONS_RADIANS)
    ax.set_xticklabels(["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$", r"$5\pi/4$", r"$3\pi/2$", r"$7\pi/4$"])
    ax.set_ylim(bottom=0)
    ax.set_rlabel_position(22.5)
    radial_max = float(np.ceil(magnitude.max() / 0.025) * 0.025)
    ax.set_ylim(0, radial_max * 1.05)
    ax.set_yticks(np.linspace(0, radial_max, 4)[1:])
    ax.tick_params(axis="y", labelsize=9)
    title = "Wind: U and V" if dataset == "wind" else "EDAS: ozone and nitrogen dioxide"
    ax.set_title(title + "\n" + timestamp + " UTC\nMean absolute empirical odd cross-covariance", fontsize=11, pad=20)
    fig.text(0.5, 0.02, "Direction in radians; 0 = east, " + r"$\pi/2$" + " = north", ha="center", fontsize=10)
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / ("directional_asymmetry_scores." + extension), dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def verify_run(metadata, path):
    """Verify against original run metadata or its bundled exact copy."""
    saved = json.loads(path.read_text(encoding="utf-8"))
    tests = {
        "source_bytes_identical_to_mle_run": metadata["source_sha256"] == saved["data_sha256"],
        "same_timestamp": saved["used_times"] == [metadata["timestamp"]],
        "same_location_count": saved["n_locations"] == metadata["n_locations"],
        "same_coordinate_order": saved["original_coordinate_points"] == metadata["original_coordinate_points"],
        "same_supplied_means": bool(np.allclose(saved["supplied_component_means"], metadata["supplied_component_means"], atol=1e-14, rtol=1e-14)),
        "same_supplied_sds": bool(np.allclose(saved["supplied_component_standard_deviations_ddof0"], metadata["supplied_component_standard_deviations_ddof0"], atol=1e-14, rtol=1e-14)),
        "same_projection": all(saved["projection"][k] == metadata["projection"][k] for k in metadata["projection"]),
        "same_value_columns": saved["value_columns"] == metadata["value_columns"],
    }
    if not all(tests.values()):
        raise AssertionError("Supplied data do not match the expected MLE run: " + str(tests))
    return {"run_metadata_path": str(path.resolve()), "run_metadata_sha256": sha256(path),
            "saved_run_source_path": saved["data_path"], "tests": tests,
            "fitted_direction_radians": float(np.deg2rad(saved["direction_degrees"])), "p_candidates": saved["p_candidates"]}, saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wind-data", type=Path, default=Path("Data_wind.csv"))
    parser.add_argument("--edas-data", type=Path, default=Path("Data_edas.csv"))
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--mle-results", type=Path, default=Path("mle_results"))
    args = parser.parse_args()
    sources = args.output_root / "numerical_sources"
    sources.mkdir(parents=True, exist_ok=True)
    combined = {}
    for dataset, source in [("wind", args.wind_data), ("edas", args.edas_data)]:
        selected, coords_km, values, metadata = load_slice(source, TIMESTAMPS[dataset])
        run_metadata_path = args.mle_results / RUN_NAMES[dataset] / "run_metadata.json"
        if not run_metadata_path.is_file():
            run_metadata_path = sources / ("diagnostics_" + dataset + "_run_metadata.json")
        verification, saved = verify_run(metadata, run_metadata_path)
        metadata["matching_mle_run"] = verification
        metadata["column_meanings"] = dict(zip(VALUE_COLUMNS, ["U wind component", "V wind component"] if dataset == "wind" else ["ozone (O3)", "nitrogen dioxide (NO2)"]))
        summary, edges = marginal_diagnostics(coords_km, values)
        directional, asymmetry = directional_diagnostics(coords_km, values, edges)
        metadata["diagnostic_binning"] = {"equal_width_bins": N_BINS, "distance_cutoff_quantile": MAX_DISTANCE_QUANTILE,
            "max_distance_km": float(edges[-1]), "min_pairs_per_bin": MIN_PAIRS,
            "lag_coordinate": "mean distance of retained pairs in each bin", "edges_km": edges.tolist(),
            "covariance_pairs": "unordered distinct pairs", "directional_pairs": "ordered distinct pairs, at least 8 per orientation",
            "direction_tolerance_radians": float(DIRECTION_TOLERANCE_RADIANS),
            "candidate_directions_radians": DIRECTIONS_RADIANS.tolist(),
            "directional_score": "unweighted mean across retained bins of absolute odd component"}
        output_dir = args.output_root / "figures" / "analysis" / dataset
        output_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_dir / "marginal_empirical_summary.csv", index=False)
        directional.to_csv(output_dir / "directional_crosscov_summary.csv", index=False)
        asymmetry.to_csv(output_dir / "directional_asymmetry_scores.csv", index=False)
        slice_path = sources / ("diagnostics_" + dataset + "_selected_slice.csv")
        selected.to_csv(slice_path, index=False)
        metadata["selected_slice_export_sha256"] = sha256(slice_path)
        metadata["script_sha256"] = sha256(__file__)
        plot_marginals(summary, dataset, TIMESTAMPS[dataset], output_dir)
        plot_direction_scores(asymmetry, dataset, TIMESTAMPS[dataset], output_dir)
        if not np.allclose(asymmetry.mean_absolute_odd_component.to_numpy()[:4],
                           asymmetry.mean_absolute_odd_component.to_numpy()[4:], atol=1e-13, rtol=1e-13):
            raise AssertionError("Opposite-direction absolute scores must agree")
        metadata["outputs_sha256"] = {p.name: sha256(p) for p in sorted(output_dir.iterdir()) if p.is_file()}
        (sources / ("diagnostics_" + dataset + "_provenance.json")).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (sources / ("diagnostics_" + dataset + "_run_metadata.json")).write_text(json.dumps(saved, indent=2), encoding="utf-8")
        combined[dataset] = {"mle_parity": verification, "first_marginal_bin": summary.iloc[0].to_dict(),
                             "last_marginal_bin": summary.iloc[-1].to_dict(), "directional_scores": asymmetry.to_dict(orient="records")}
    (sources / "diagnostics_summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
