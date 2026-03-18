# SLAM_AI

Projekt porównuje klasyczne i AI-wspomagane metody lokalizacji/mapowania robota 2D (ROS 2 Jazzy + Gazebo + LiDAR 360).

## Metody w eksperymencie

1. `Baseline SLAM`  
`slam_toolbox` na zaszumionej odometrii (`/map`).

2. `AI SLAM`  
Model MLP (`363 -> 256 -> 128 -> 64 -> 3`) koryguje estymację pozycji, a wynik trafia do `slam_toolbox` (`/map_ai`).

3. `ScanMatcher localmap`  
Klasyczne dopasowanie kolejnych skanów (szybkie, lekkie; tor referencyjny).

4. `ScanMatcher bruteforce` (opcjonalnie)  
Pełne przeszukiwanie siatki transformacji (wolniejsze, referencyjne).

5. `Robak`  
Model Conv1D na parze skanów `(scan_{t-1}, scan_t)` przewiduje `Δx, Δy, Δθ`.

6. `Rywak`  
Model MLP przewiduje `v, ω` z cech: `d_theta1, d_theta2, delta_scan`.

## Fazy pipeline

1. Zbieranie datasetu (train world)
2. Trening modeli
3. Test i ewaluacja (test world)
4. Zapis wyników do `out/exp_YYYYMMDD_HHMMSS`

## One-click uruchomienie

Nowy skrypt wykonuje pełną sekwencję:
- czyszczenie `build/install/log`
- aktywacja `.venv`
- `rosdep install`
- `colcon build --symlink-install`
- source ROS + workspace
- `cleanup.sh`
- `run_full_cycle.sh`

```bash
cd ~/projects/SLAM_AI
./scripts/run_all.sh
```

## Dashboard

Lokalny dashboard HTTP pozwala:
- wybierać istniejące eksperymenty i datasety z `out/`
- odpalać `run_all.sh`, `run_full_cycle.sh`, `run_experiment.sh`
- trenować `AI`, `robak`, `rywak` na wybranym eksperymencie
- generować wykres trajektorii i błędu z własnym zakresem czasu oraz osi
- przeglądać artefakty i logi zadań
- otwierać automatycznie generowany indeks funkcji

Uruchomienie:

```bash
cd ~/SLAM_AI
source .venv/bin/activate
./scripts/run_dashboard.sh
```

Domyślny adres: `http://127.0.0.1:8765`

Zatrzymanie:

```bash
cd ~/projects/SLAM_AI
bash ./scripts/kill_dashboard.sh
```

Jeśli dashboard działa na innym porcie, podaj go jako argument, np. `bash ./scripts/kill_dashboard.sh 8766`.

## Najważniejsze pliki

- Konfiguracja eksperymentu: `ai_slam_ws/src/ai_slam_bringup/config/experiment_config.yaml`
- Launch główny: `ai_slam_ws/src/ai_slam_bringup/launch/demo.launch.py`
- Skrypt one-click: `scripts/run_all.sh`

## Wyniki

Po zakończeniu eksperymentu katalog `out/exp_*` zawiera m.in.:
- `results.json` (RMSE/IoU)
- `trajectory.png`, `errors.png`, `maps.png`
- `dataset*.npz`, `model*.pt`, `train_history*.json`

## Przydatne skrypty

- `scripts/run_all.sh` – pełny pipeline (build + train + test)
- `scripts/run_full_cycle.sh` – train + test na istniejącym środowisku
- `scripts/run_experiment.sh` – pojedyncze launchowanie (`fast`, `full`, `train`, `test`)
- `scripts/run_dashboard.sh` – dashboard WWW nad eksperymentami, datasetami i wykresami
- `scripts/generate_function_index.py` – generator `docs/function_index.{md,json}`
- `scripts/cleanup.sh` – ubijanie zaległych procesów ROS/Gazebo
- `scripts/generate_thesis_report.py` – wykresy i tabele do pracy (CSV/MD/LaTeX + PNG) z `results.json`

### Raport do pracy

Przykład na bazie sweepa:

```bash
python3 scripts/generate_thesis_report.py \
  --sweep out/sweep_20260308_170242.csv \
  --output-dir out/thesis_raport
```

Wygenerowane pliki: `table_experiments.csv`, `table_method_stats.{csv,md,tex}`, `fig_*.png`.
