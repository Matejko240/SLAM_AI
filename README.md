# AI SLAM - Simultaneous Localization and Mapping z Sztuczną Inteligencją

Kompletna implementacja systemu SLAM wspomaganego AI. System mapuje otoczenie robota za pomocą czujnika LIDAR 2D (360 próbek/obrót) i jednocześnie estymuje jego położenie (x, y, θ). Moduł AI uczy się korygować błędy odometrii, poprawiając dokładność lokalizacji.

## Szybki Start (5 minut)

### 1. Instalacja
```bash
cd ~/SLAM_AI
chmod +x ./scripts/install_deps.sh ./scripts/cleanup.sh
./scripts/install_deps.sh
source ./.venv/bin/activate
```

### 2. Build
```bash
cd ~/SLAM_AI/ai_slam_ws
rosdep install --from-paths src --ignore-src -r -y --skip-keys ament_python
colcon build --symlink-install
source /opt/ros/jazzy/setup.bash
source install/setup.bash
```

### 3. Uruchomienie

**Domyślny eksperyment (pełna konfiguracja):**
```bash
cd ~/SLAM_AI
./scripts/cleanup.sh
./scripts/run_experiment.sh
```

**Szybki test (~40 sekund):**
```bash
./scripts/run_experiment.sh fast
```

**Baseline SLAM (bez AI):**
```bash
./scripts/run_experiment.sh mode:=baseline
```

**Z GUI Gazebo:**
```bash
./scripts/run_experiment.sh gui:=true
```

### 4. Wyniki

Wyniki zapisywane są w `out/exp_YYYYMMDD_HHMMSS/`:
```bash
# Najnowszy eksperyment
ls -lt out/ | head -2
cat out/exp_*/results.json | python -m json.tool
```

Kluczowe metryki w `results.json`:
- `iou_map_baseline` - IoU mapy baseline (0-1, wyżej = lepiej)
- `iou_map_ai` - IoU mapy AI
- `rmse_xy_baseline/ai` - błąd pozycji (m)

## Architektura Systemu

### Pipeline Eksperymentu (tryb AI)

```
FAZA 1: Zbieranie danych (dataset_duration)
   Robot jeździ → LiDAR + Odometria + GT → dataset.npz
   
FAZA 2: Trening modelu
   dataset.npz → MLP (363→256→128→64→3) → model.pt
   
FAZA 3: Inferencja AI
   LiDAR + Odom → model.pt → /pose_ai (korekcja)
   
FAZA 4: Ewaluacja
   Porównanie: baseline vs AI → results.json + wykresy
```

### 4 Moduły

**1. Moduł SLAM** (`ai_slam_bringup`)
- Algorytm: slam_toolbox (synchroniczny)
- Input: LaserScan (360 próbek) z `/scan_slam`
- Output: Mapa (`/map`) + Pozycja robota (x, y, θ)

**2. Moduł AI** (`ai_slam_ai`)
- Zbieranie datasetu: `dataset_recorder.py`
- Trening modelu: `train_model.py` (MLP: 363→256→128→64→3)
- Inferencja: `infer_node.py` (publikuje na `/pose_ai`)

**3. Moduł Odometrii** (`ai_slam_bringup/odom_corruptor.py`)
- Ground truth: `/odom_raw` z Gazebo
- Z dryfem: `/odom` (symuluje błędy rzeczywiste)

**4. Moduł Ewaluacji** (`ai_slam_eval`)
- Metryki: RMSE (x, y, θ), IoU mapy
- Output: `results.json`, wykresy PNG

## Centralna Konfiguracja (YAML)

Wszystkie parametry eksperymentu znajdują się w jednym pliku YAML:

### Pliki Konfiguracyjne

| Plik | Opis | Czas |
|------|------|------|
| `experiment_config.yaml` | Pełny eksperyment | ~2-3 min |
| `fast_test.yaml` | Szybki test | ~40 sek |

### Użycie

```bash
# Domyślna konfiguracja (experiment_config.yaml)
ros2 launch ai_slam_bringup demo.launch.py

# Szybki test
ros2 launch ai_slam_bringup demo.launch.py config:=fast_test.yaml

# Własna konfiguracja
ros2 launch ai_slam_bringup demo.launch.py config:=/path/to/my_config.yaml

# Override pojedynczych parametrów
ros2 launch ai_slam_bringup demo.launch.py config:=fast_test.yaml seed:=999 duration_sec:=60
```

### Struktura experiment_config.yaml

```yaml
experiment:
  mode: "ai"              # "baseline" lub "ai"
  seed: 123               # Dla powtarzalności
  gui: false              # GUI Gazebo

timing:
  dataset_duration: 45.0  # Czas zbierania danych (s)
  experiment_duration: 120.0  # Całkowity czas (s)
  dataset_wait_timeout: 120.0 # Max czekanie na dataset

dataset:
  max_samples: 5000       # Max próbek w datasecie
  scan_topic: "/scan"
  odom_topic: "/odom"
  gt_topic: "/ground_truth_pose"

training:
  max_epochs: 200
  patience: 20            # Early stopping
  learning_rate: 0.001
  batch_size: 128
  validation_ratio: 0.2

inference:
  model_wait_timeout: 300.0

odometry:
  rw_sigma_xy: 0.005      # Szum pozycyjny
  rw_sigma_theta: 0.003   # Szum kątowy

driver:
  linear_velocity: 0.3    # Prędkość robota (m/s)
  angular_velocity: 0.5   # Prędkość kątowa (rad/s)
  turn_probability: 0.02

output:
  base_dir: "out"         # Folder wyjściowy
```

## Instrukcja Użytkownika

### Tryb 1: Szybki Test
```bash
ros2 launch ai_slam_bringup demo.launch.py config:=fast_test.yaml
```
- ~40 sekund, szybka weryfikacja że wszystko działa

### Tryb 2: Pełny Eksperyment AI
```bash
ros2 launch ai_slam_bringup demo.launch.py
```
- Używa `experiment_config.yaml`
- ~2-3 minuty pełnego pipeline'u

### Tryb 3: Baseline (bez AI)
```bash
ros2 launch ai_slam_bringup demo.launch.py mode:=baseline duration_sec:=120
```
- Tylko SLAM bez korekcji AI
- Do porównań i testowania

### Wyniki Eksperymentu

Każdy eksperyment tworzy folder `out/exp_YYYYMMDD_HHMMSS/` zawierający:

| Plik | Opis |
|------|------|
| `results.json` | Metryki (RMSE, IoU) |
| `dataset.npz` | Zebrane dane |
| `model.pt` | Wytrenowany model |
| `train_history.json` | Historia treningu |
| `experiment_metadata.json` | Pełne metadane |
| `trajectory.png` | Wykres trajektorii |
| `errors.png` | Wykres błędów |
| `maps.png` | Porównanie map |

### Interpretacja results.json

```json
{
  "mode": "ai",
  "metrics": {
    "iou_map_baseline": 0.037,    // IoU mapy baseline (0-1)
    "iou_map_ai": 0.181,          // IoU mapy AI
    "rmse_xy_baseline": 0.0054,   // Błąd pozycji baseline (m)
    "rmse_xy_ai": 0.0038          // Błąd pozycji AI (m)
  }
}
```

**Interpretacja IoU:**
- 0.0-0.1: Słaba jakość mapy
- 0.1-0.3: Średnia jakość
- 0.3+: Dobra jakość

### Parametry Launch (override)

| Parametr | Opis | Przykład |
|----------|------|----------|
| `config` | Plik konfiguracyjny | `config:=fast_test.yaml` |
| `mode` | baseline/ai | `mode:=baseline` |
| `seed` | Random seed | `seed:=42` |
| `duration_sec` | Czas eksperymentu | `duration_sec:=180` |
| `dataset_duration_sec` | Czas zbierania | `dataset_duration_sec:=60` |
| `gui` | GUI Gazebo | `gui:=true` |
| `out_dir` | Folder wyjściowy | `out_dir:=my_output` |

## Troubleshooting


### AI nie poprawia wyników
- Zwiększ `dataset_duration` w config (więcej danych)
- Zwiększ `max_epochs` (dłuższy trening)
- Zmniejsz `learning_rate` (wolniejsze uczenie)
- Sprawdź czy model się wytrenował: `cat out/exp_*/train_history.json`

### Błędy typu "parameter type mismatch"
```bash
# Przebuduj pakiety
cd ~/SLAM_AI/ai_slam_ws
colcon build --packages-select ai_slam_ai ai_slam_eval ai_slam_bringup
source install/setup.bash
```

## Wymagania Systemu
- Ubuntu 24.04
- ROS 2 Jazzy
- Gazebo Harmonic
- GPU opcjonalny

## Struktura Projektu
```
SLAM_AI/
├── README.md                    # Ta dokumentacja
├── scripts/
│   ├── install_deps.sh         # Instalacja zależności
│   ├── cleanup.sh              # Czyszczenie procesów
│   ├── inspect_dataset.py      # Analiza zebranego datasetu
│   └── generate_reference_map.py # Generowanie mapy referencyjnej
├── ai_slam_ws/                 # ROS 2 Workspace
│   ├── out/                    # Wyniki eksperymentów
│   │   └── exp_YYYYMMDD_HHMMSS/ # Folder każdego eksperymentu
│   └── src/
│       ├── ai_slam_ai/         # Zbieranie, trening, inferencja
│       ├── ai_slam_bringup/    # Launch files + config YAML
│       │   ├── config/
│       │   │   ├── experiment_config.yaml  # Główna konfiguracja
│       │   │   └── fast_test.yaml          # Szybki test
│       │   └── launch/
│       │       └── demo.launch.py
│       ├── ai_slam_description/# Model robota (URDF/SDF)
│       ├── ai_slam_gazebo/     # Świat Gazebo
│       └── ai_slam_eval/       # Ewaluacja + mapa referencyjna
└── .venv/                      # Python virtual environment
```

## Narzędzia Analizy

### Inspekcja Datasetu
Po zebraniu danych (FAZA 1) można przeanalizować dataset za pomocą skryptu:

```bash
cd ~/SLAM_AI
source .venv/bin/activate
python3 scripts/inspect_dataset.py out/exp_20260103_151922
```

Skrypt generuje:
- **Statystyki** - rozmiar datasetu, rozkład korekt (dx, dy, dθ)
- **Mapa LiDAR** - wizualizacja wszystkich skanów złożonych w mapę
- **Trajektoria** - ścieżka robota z odometrii
- **Histogram** - rozkład odległości z LiDAR
- **Wykres korekt** - rozrzut błędów pozycji

Wynik zapisywany do `out/dataset_analysis.png`.

### Generowanie Mapy Referencyjnej
Mapa referencyjna (ground truth) jest generowana na podstawie pliku świata Gazebo:

```bash
cd ~/SLAM_AI
python scripts/generate_reference_map.py
```

Generuje `reference_map.pgm` i `reference_map.yaml` w `ai_slam_ws/src/ai_slam_eval/maps/`.

## Ważne Topiki ROS
- `/scan` - LaserScan z Gazebo
- `/scan_slam` - LaserScan 360 próbek
- `/odom_raw` - Ground truth odometria
- `/odom` - Odometria z dryfem
- `/map` - Mapa SLAM baseline
- `/pose_ai` - Korekcja AI
- `/cmd_vel` - Komendy prędkości

## FAQ

**Q: Ile czasu trwa eksperyment?**  
A: `fast_test.yaml` ~40s, `experiment_config.yaml` ~2-3 min

**Q: Czy potrzebny GPU?**  
A: Nie, CPU wystarczy. GPU przyspiesza trening ale nie jest wymagany.

**Q: Gdzie są wyniki?**  
A: W `ai_slam_ws/out/exp_YYYYMMDD_HHMMSS/`

**Q: Jak zmienić parametry bez edycji plików?**  
A: Użyj override: `ros2 launch ... seed:=42 duration_sec:=60`

**Q: Jak stworzyć własną konfigurację?**  
A: Skopiuj `experiment_config.yaml`, zmodyfikuj, użyj `config:=/path/to/file.yaml`

**Q: Dlaczego IoU jest niskie?**  
A: To normalne dla krótkiego eksperymentu. IoU ~0.05-0.20 jest typowe.

## Ręczne Uruchomienie Modułów

```bash
# Zbieranie datasetu
ros2 run ai_slam_ai dataset_recorder --ros-args \
  -p out_dir:=out -p duration_sec:=60 -p max_samples:=1000

# Trening
ros2 run ai_slam_ai train_model --ros-args \
  -p out_dir:=out -p max_epochs:=100 -p patience:=15

# Inferencja
ros2 run ai_slam_ai infer_node --ros-args \
  -p out_dir:=out
```

