#!/usr/bin/env python3
import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from results_metric_keys import metrics_rmse_theta_odom, metrics_rmse_xy_odom


METHODS = [
    ("baseline", "Odom vs GT", "rmse_xy_odom_topic", "rmse_theta_odom_topic", "iou_map_baseline"),
    ("ai", "AI", "rmse_xy_ai", "rmse_theta_ai", "iou_map_ai"),
    ("robak", "Robak", "rmse_xy_robak", "rmse_theta_robak", "iou_map_robak"),
    ("rywak", "Rywak", "rmse_xy_rywak", "rmse_theta_rywak", "iou_map_rywak"),
    ("scanmatch", "ScanMatch", "rmse_xy_scanmatch", "rmse_theta_scanmatch", None),
    ("bruteforce", "BruteForce", "rmse_xy_bruteforce", "rmse_theta_bruteforce", None),
]


def to_float(v):
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def collect_results_paths(root_out: Path, sweep_paths, experiment_paths, result_paths):
    collected = []
    seen = set()

    def add_path(p: Path):
        rp = p.resolve()
        if rp in seen:
            return
        if rp.exists():
            collected.append(rp)
            seen.add(rp)

    for r in result_paths:
        add_path(Path(r))

    for e in experiment_paths:
        exp = Path(e)
        if exp.is_dir():
            add_path(exp / "results.json")
        else:
            add_path(exp)

    for s in sweep_paths:
        sweep_path = Path(s)
        if not sweep_path.exists():
            continue
        with open(sweep_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                status = str(row.get("status", "OK")).strip().upper()
                if status and status != "OK":
                    continue
                exp_id = str(row.get("exp_id", "")).strip()
                if not exp_id:
                    continue
                add_path(root_out / exp_id / "results.json")

    if not collected:
        for p in sorted(root_out.glob("exp_*/results.json")):
            add_path(p)

    return collected


def load_records(results_paths):
    records = []
    for path in results_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        metrics = data.get("metrics", {}) or {}
        rec = {
            "exp_id": path.parent.name,
            "results_path": str(path),
            "duration_sec": to_float(data.get("duration_sec")),
            "seed": data.get("seed"),
            "mode": data.get("mode"),
        }
        for _key, _label, rmse_xy_key, rmse_th_key, iou_key in METHODS:
            if rmse_xy_key == "rmse_xy_odom_topic":
                rec[rmse_xy_key] = to_float(metrics_rmse_xy_odom(metrics))
            else:
                rec[rmse_xy_key] = to_float(metrics.get(rmse_xy_key))
            if rmse_th_key == "rmse_theta_odom_topic":
                rec[rmse_th_key] = to_float(metrics_rmse_theta_odom(metrics))
            else:
                rec[rmse_th_key] = to_float(metrics.get(rmse_th_key))
            if iou_key is not None:
                rec[iou_key] = to_float(metrics.get(iou_key))

        records.append(rec)

    records.sort(key=lambda r: r["exp_id"])
    return records


def write_experiment_table(records, out_path: Path):
    columns = [
        "exp_id",
        "mode",
        "seed",
        "duration_sec",
        "rmse_xy_odom_topic",
        "rmse_theta_odom_topic",
        "iou_map_baseline",
        "rmse_xy_ai",
        "rmse_theta_ai",
        "iou_map_ai",
        "rmse_xy_robak",
        "rmse_theta_robak",
        "iou_map_robak",
        "rmse_xy_rywak",
        "rmse_theta_rywak",
        "iou_map_rywak",
        "rmse_xy_scanmatch",
        "rmse_theta_scanmatch",
        "rmse_xy_bruteforce",
        "rmse_theta_bruteforce",
        "results_path",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)


def stats(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "median": None,
            "min": None,
            "max": None,
        }
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def method_statistics(records):
    out = []
    for method_key, label, rmse_xy_key, rmse_th_key, iou_key in METHODS:
        rmse_xy_vals = [r[rmse_xy_key] for r in records if r.get(rmse_xy_key) is not None]
        rmse_th_vals = [r[rmse_th_key] for r in records if r.get(rmse_th_key) is not None]
        iou_vals = []
        if iou_key is not None:
            iou_vals = [r[iou_key] for r in records if r.get(iou_key) is not None]

        rmse_xy_imp = []
        rmse_th_imp = []
        iou_imp = []
        for r in records:
            b_xy = r.get("rmse_xy_odom_topic")
            b_th = r.get("rmse_theta_odom_topic")
            b_iou = r.get("iou_map_baseline")
            m_xy = r.get(rmse_xy_key)
            m_th = r.get(rmse_th_key)
            m_iou = r.get(iou_key) if iou_key is not None else None

            if method_key == "baseline":
                if b_xy is not None:
                    rmse_xy_imp.append(0.0)
                if b_th is not None:
                    rmse_th_imp.append(0.0)
                if b_iou is not None:
                    iou_imp.append(0.0)
                continue

            if b_xy is not None and m_xy is not None and b_xy > 0:
                rmse_xy_imp.append((b_xy - m_xy) / b_xy * 100.0)
            if b_th is not None and m_th is not None and b_th > 0:
                rmse_th_imp.append((b_th - m_th) / b_th * 100.0)
            if b_iou is not None and m_iou is not None and b_iou > 0:
                iou_imp.append((m_iou - b_iou) / b_iou * 100.0)

        s_xy = stats(rmse_xy_vals)
        s_th = stats(rmse_th_vals)
        s_iou = stats(iou_vals)
        s_xy_imp = stats(rmse_xy_imp)
        s_th_imp = stats(rmse_th_imp)
        s_iou_imp = stats(iou_imp)

        out.append(
            {
                "method": method_key,
                "label": label,
                "n_rmse_xy": s_xy["n"],
                "rmse_xy_mean": s_xy["mean"],
                "rmse_xy_std": s_xy["std"],
                "rmse_xy_median": s_xy["median"],
                "n_rmse_theta": s_th["n"],
                "rmse_theta_mean": s_th["mean"],
                "rmse_theta_std": s_th["std"],
                "rmse_theta_median": s_th["median"],
                "n_iou": s_iou["n"],
                "iou_mean": s_iou["mean"],
                "iou_std": s_iou["std"],
                "iou_median": s_iou["median"],
                "rmse_xy_improvement_vs_baseline_pct_mean": s_xy_imp["mean"],
                "rmse_theta_improvement_vs_baseline_pct_mean": s_th_imp["mean"],
                "iou_improvement_vs_baseline_pct_mean": s_iou_imp["mean"],
            }
        )
    return out


def fmt(v, digits=4):
    if v is None:
        return "-"
    return f"{v:.{digits}f}"


def write_method_tables(rows, out_dir: Path):
    csv_path = out_dir / "table_method_stats.csv"
    md_path = out_dir / "table_method_stats.md"
    tex_path = out_dir / "table_method_stats.tex"

    columns = [
        "method",
        "label",
        "n_rmse_xy",
        "rmse_xy_mean",
        "rmse_xy_std",
        "rmse_xy_median",
        "n_rmse_theta",
        "rmse_theta_mean",
        "rmse_theta_std",
        "rmse_theta_median",
        "n_iou",
        "iou_mean",
        "iou_std",
        "iou_median",
        "rmse_xy_improvement_vs_baseline_pct_mean",
        "rmse_theta_improvement_vs_baseline_pct_mean",
        "iou_improvement_vs_baseline_pct_mean",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| Metoda | n | RMSE XY [m] | RMSE TH [rad] | IoU | Delta RMSE XY vs baseline [%] | Delta IoU vs baseline [%] |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['label']} | {r['n_rmse_xy']} | "
                f"{fmt(r['rmse_xy_mean'])} +- {fmt(r['rmse_xy_std'])} | "
                f"{fmt(r['rmse_theta_mean'])} +- {fmt(r['rmse_theta_std'])} | "
                f"{fmt(r['iou_mean'])} +- {fmt(r['iou_std'])} | "
                f"{fmt(r['rmse_xy_improvement_vs_baseline_pct_mean'], 2)} | "
                f"{fmt(r['iou_improvement_vs_baseline_pct_mean'], 2)} |\n"
            )

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lrrrrrr}\n")
        f.write("\\hline\n")
        f.write("Metoda & n & RMSE$_{xy}$ [m] & RMSE$_{\\theta}$ [rad] & IoU & $\\Delta$RMSE$_{xy}$ [\\%] & $\\Delta$IoU [\\%] \\\\\n")
        f.write("\\hline\n")
        for r in rows:
            f.write(
                f"{r['label']} & {r['n_rmse_xy']} & "
                f"{fmt(r['rmse_xy_mean'])} $\\pm$ {fmt(r['rmse_xy_std'])} & "
                f"{fmt(r['rmse_theta_mean'])} $\\pm$ {fmt(r['rmse_theta_std'])} & "
                f"{fmt(r['iou_mean'])} $\\pm$ {fmt(r['iou_std'])} & "
                f"{fmt(r['rmse_xy_improvement_vs_baseline_pct_mean'], 2)} & "
                f"{fmt(r['iou_improvement_vs_baseline_pct_mean'], 2)} \\\\\n"
            )
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")


def plot_box(records, metric_key, title, ylabel, out_path: Path):
    labels = []
    series = []
    for _mk, label, rmse_xy_key, rmse_th_key, _iou_key in METHODS:
        key = metric_key
        if key == "rmse_xy":
            values = [r[rmse_xy_key] for r in records if r.get(rmse_xy_key) is not None]
        else:
            values = [r[rmse_th_key] for r in records if r.get(rmse_th_key) is not None]
        if values:
            labels.append(label)
            series.append(values)

    if not series:
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(series, patch_artist=True, labels=labels, showmeans=True)
    for patch in bp["boxes"]:
        patch.set_alpha(0.45)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Metoda")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_iou_bar(method_rows, out_path: Path):
    labels = []
    means = []
    stds = []
    for r in method_rows:
        if r["iou_mean"] is None:
            continue
        labels.append(r["label"])
        means.append(r["iou_mean"])
        stds.append(0.0 if r["iou_std"] is None else r["iou_std"])

    if not labels:
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4, alpha=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("IoU")
    ax.set_title("IoU map - srednia i odchylenie standardowe")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_rmse_heatmap(records, out_path: Path):
    if not records:
        return

    method_labels = [m[1] for m in METHODS]
    method_rmse_keys = [m[2] for m in METHODS]

    mat = np.full((len(records), len(METHODS)), np.nan, dtype=np.float64)
    ylabels = []
    for i, rec in enumerate(records):
        ylabels.append(rec["exp_id"])
        for j, key in enumerate(method_rmse_keys):
            v = rec.get(key)
            if v is not None:
                mat[i, j] = v

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(records) + 2)))
    masked = np.ma.masked_invalid(mat)
    im = ax.imshow(masked, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(method_labels)))
    ax.set_xticklabels(method_labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_title("Heatmap RMSE XY (m) - eksperymenty vs metody")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("RMSE XY [m]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_mean_rank(records, out_path: Path):
    method_keys = [m[0] for m in METHODS]
    method_labels = [m[1] for m in METHODS]
    rmse_xy_keys = [m[2] for m in METHODS]
    ranks = {k: [] for k in method_keys}

    for rec in records:
        pairs = []
        for k, metric_key in zip(method_keys, rmse_xy_keys):
            v = rec.get(metric_key)
            if v is not None:
                pairs.append((k, v))
        if len(pairs) < 2:
            continue
        pairs.sort(key=lambda x: x[1])
        for rank_idx, (k, _v) in enumerate(pairs, start=1):
            ranks[k].append(float(rank_idx))

    labels = []
    means = []
    for k, label in zip(method_keys, method_labels):
        if not ranks[k]:
            continue
        labels.append(label)
        means.append(float(np.mean(ranks[k])))

    if not labels:
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    ax.bar(x, means, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Srednia pozycja w rankingu (RMSE XY)")
    ax.set_title("Ranking metod (im mniej, tym lepiej)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generuje wykresy i tabele pod raport/magisterke z wynikow eksperymentow.")
    parser.add_argument("--root-out", default="out", help="Katalog z exp_*/results.json (domyslnie: out)")
    parser.add_argument("--sweep", action="append", default=[], help="Sciezka do pliku sweep_*.csv (mozna podac wiele razy)")
    parser.add_argument("--experiment", action="append", default=[], help="Sciezka do katalogu exp_* lub do results.json")
    parser.add_argument("--result", action="append", default=[], help="Sciezka do konkretnego results.json")
    parser.add_argument("--output-dir", default="", help="Katalog wyjsciowy raportu (domyslnie: out/thesis_YYYYMMDD_HHMMSS)")
    args = parser.parse_args()

    root_out = Path(args.root_out).resolve()
    results_paths = collect_results_paths(root_out, args.sweep, args.experiment, args.result)
    records = load_records(results_paths)
    if not records:
        raise SystemExit("Brak poprawnych results.json do analizy.")

    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = root_out / f"thesis_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    write_experiment_table(records, out_dir / "table_experiments.csv")
    method_rows = method_statistics(records)
    write_method_tables(method_rows, out_dir)

    plot_box(records, "rmse_xy", "RMSE XY - porownanie metod", "RMSE XY [m]", out_dir / "fig_rmse_xy_boxplot.png")
    plot_box(records, "rmse_theta", "RMSE Theta - porownanie metod", "RMSE Theta [rad]", out_dir / "fig_rmse_theta_boxplot.png")
    plot_iou_bar(method_rows, out_dir / "fig_iou_bar.png")
    plot_rmse_heatmap(records, out_dir / "fig_rmse_xy_heatmap.png")
    plot_mean_rank(records, out_dir / "fig_rmse_xy_rank.png")

    print(f"Raport gotowy: {out_dir}")
    print(f"Liczba eksperymentow: {len(records)}")
    print("Pliki:")
    for name in [
        "table_experiments.csv",
        "table_method_stats.csv",
        "table_method_stats.md",
        "table_method_stats.tex",
        "fig_rmse_xy_boxplot.png",
        "fig_rmse_theta_boxplot.png",
        "fig_iou_bar.png",
        "fig_rmse_xy_heatmap.png",
        "fig_rmse_xy_rank.png",
    ]:
        print(f"- {out_dir / name}")


if __name__ == "__main__":
    main()
