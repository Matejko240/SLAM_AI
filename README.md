# AI SLAM – Simultaneous Localization and Mapping z Sztuczną Inteligencją

Kompletna implementacja systemu SLAM wspomaganego AI. System mapuje otoczenie robota za pomocą czujnika LIDAR 2D (360 próbek/obrót) i jednocześnie estymuje jego położenie (x, y, θ). Moduł AI uczy się korygować błędy odometrii, poprawiając stabilność klasycznego SLAM.

## Co jest porównywane (mini-benchmarker)

Projekt porównuje kilka torów estymacji (x, y, θ) uruchamianych na tych samych danych wejściowych:

1. **Baseline SLAM** – `slam_toolbox` na odometrii z dryfem → `/map` + trajektoria.
2. **AI SLAM** – `slam_toolbox` + korekcja (AI) → `/map_ai` + trajektoria AI.
3. **Scan-to-scan (local)** – lekki estymator ruchu z porównania dwóch kolejnych skanów → `/pose_scanmatch`.
4. **Scan-to-scan (bruteforce)** – referencyjny wariant przeszukiwania siatki → `/pose_bruteforce`.

> Uwaga: skan-matching (3–4) jest na ten moment „eksperymentalny” i służy głównie jako punkt odniesienia / sanity-check w badaniach. W obecnej wersji potrafi wypadać słabo w porównaniu do `slam_toolbox`.

## Metody SLAM / Estymatory (porównanie)

W projekcie działają równolegle cztery tory estymacji pozycji robota (trajektorii) oraz moduł ewaluacji.
Celem jest porównanie podejścia AI z metodami klasycznymi na tych samych danych wejściowych i w tym samym przejeździe.

### Wspólne wejścia (dla wszystkich torów)
- LiDAR 2D: `sensor_msgs/LaserScan` (używane 360 próbek na skan)
- Odometria: `nav_msgs/Odometry` (w trybie AI dodatkowo „zaszumiona” i/lub „korygowana”)

---

## Tor 1: SLAM Toolbox – baseline (klasyczny SLAM)
**Node:** `slam_toolbox_baseline` (`slam_toolbox/sync_slam_toolbox_node`, lifecycle)

**Wejścia:**
- `/scan_slam` (LaserScan)
- odometria/TF zgodnie z konfiguracją `slam_toolbox_baseline.yaml`

**Wyjścia:**
- `/map` (OccupancyGrid)
- TF `map -> odom` (typowe dla slam_toolbox)
- pozycja robota (pośrednio przez TF / internal pose)

**Rola w porównaniu:**
- główny klasyczny punkt odniesienia (pełny SLAM, mapa + lokalizacja)
<details> <summary><b>📄 Kliknij, aby zobaczyć Pseudokod (Baseline Logic)</b></summary>

```python
def run_baseline_slam():
    """
    Konfiguracja: slam_toolbox_baseline.yaml
    Node: slam_toolbox_baseline (lifecycle)
    """
    # 1. Wejście: Skaner i TF (z dryfem)
    scan = subscribe("/scan_slam")          # Znormalizowany LiDAR
    base_pose = lookup_tf("odom", "base_link") # Pozycja zaszumiona (odometria)

    # 2. Parametry (slam_toolbox_baseline.yaml)
    resolution = 0.05        # [m/pixel]
    max_laser_range = 10.0   # [m]
    
    # 3. Karto Scan Matcher (Graph SLAM)
    # SLAM próbuje dopasować skan do mapy, startując z pozycji 'odom'
    corrected_pose = karto_match(
        scan, 
        initial_guess=base_pose, 
        search_window=0.5 # domyślny
    )

    # 4. Wyjście
    # Publikacja mapy i korekty TF (niwelującej dryf 'odom')
    publish_topic("/map", resolution=0.05)
    publish_tf(parent="map", child="odom")
```
</details>


## Tor 2: SLAM Toolbox + AI (SLAM z korygowaną odometrią)
**Node:** `slam_toolbox_ai` (`slam_toolbox/sync_slam_toolbox_node`, lifecycle)

**Wejścia:**
- `/scan_slam`
- odometria „AI”: TF / `odom_ai` publikowane przez `infer_node`

**Wyjścia:**
- `/map_ai` (OccupancyGrid)  ← (remap z `/map`)
- TF `map_ai -> odom_ai` (zgodnie z konfiguracją slam_toolbox_ai)
- `/pose_ai` (PoseStamped) z modułu AI (patrz poniżej)

**Rola w porównaniu:**
- główny tor „AI-SLAM” oceniany względem baseline
<details> <summary><b>📄 Kliknij, aby zobaczyć Pseudokod (AI Pipeline Logic)</b></summary>

```python
def run_ai_slam_pipeline():
    """
    Pipeline: infer_node -> slam_toolbox_ai
    Config: experiment_config.yaml (sekcja 'inference' i 'slam')
    """
    
    # --- KROK 1: Inferencja (infer_node.py) ---
    # Wejścia zdefiniowane w experiment_config.yaml
    scan_in = subscribe("/scan_slam")
    odom_in = subscribe("/odom")      # Zaszumiona odometria

    # Model MLP przewiduje błąd (dx, dy, dtheta)
    correction = model.predict(scan_in, odom_in)
    
    # Obliczanie "czystej" odometrii
    odom_ai_pose = odom_in.pose + correction
    
    # Wyjście inferencji: Nowa ramka TF
    # Służy jako "lepszy start" dla SLAM-a
    publish_tf(parent="odom_ai", child="base_link")


    # --- KROK 2: SLAM (slam_toolbox_ai) ---
    # Konfiguracja: slam_toolbox_ai.yaml
    
    # SLAM używa poprawionej ramki 'odom_ai' zamiast zwykłego 'odom'
    # odom_frame: "odom_ai"  
    
    current_scan = subscribe("/scan_slam")
    guess_pose = lookup_tf("odom_ai", "base_link") 
    
    # Budowanie mapy z mniejszym błędem wejściowym
    update_map(current_scan, guess_pose)
    
    # Wyjście: Mapa na osobnym topicu (z remapowania)
    publish_topic("/map_ai")
  ```
</details>


## Tor 3: Scan Matcher – local (klasyczne dopasowanie skanów, szybkie)
**Node:** `scan_matcher_local` (`ai_slam_bringup/scan_matcher.py`)

**Wejście:**
- `scan_topic=/scan_slam`

**Wyjścia:**
- `pose_topic=/pose_scanmatch` (PoseStamped, frame_id="odom" – kompatybilne z ewaluacją)
- `twist_topic=/twist_scanmatch` (TwistStamped: v, omega)
- TF (opcjonalnie): `odom_scanmatch -> base_link_scanmatch`

**Opis działania:**
- estymacja ruchu między kolejnymi skanami (dx, dy, dθ)
- metoda „local”: wielopoziomowe przeszukiwanie małego okna (szybka)

**Najważniejsze parametry:**
- `grid_res`, `grid_extent`, `max_use_range`, `max_points`
- okna i kroki local search: `local_lvl1_*`, `local_lvl2_*`, `local_lvl3_*`

<details> <summary><b>📄 Kliknij, aby zobaczyć Pseudokod (Local Search)</b></summary>

```python
def run_scan_matcher_local():
    """
    Node: scan_matcher_local
    Param: method="local"
    """
    prev_scan = None
    global_pose = (0, 0, 0) # Start w (0,0,0)

    while True:
        curr_scan = subscribe("/scan_slam")
        
        # Algorytm Local Search (Gradient / Hill Climbing)
        # Szuka tylko w bardzo małym otoczeniu poprzedniej pozycji
        dx, dy, dth = find_match_local(
            target=prev_scan, 
            source=curr_scan,
            search_step=0.01  # Precyzyjny, mały krok
        )
        
        # Integracja wyniku (Dead Reckoning)
        global_pose += (dx, dy, dth)
        
        # Wyjścia (zgodne z demo.launch.py)
        publish("/pose_scanmatch", global_pose)  # frame_id="odom"
        publish("/twist_scanmatch", (dx/dt, dth/dt))
        
        prev_scan = curr_scan
```
</details>


## Tor 4: Scan Matcher – bruteforce (klasyczne dopasowanie skanów, referencja)
**Node:** `scan_matcher_bruteforce` (`ai_slam_bringup/scan_matcher.py`)

**Wejście:**
- `scan_topic=/scan_slam`

**Wyjścia:**
- `pose_topic=/pose_bruteforce`
- `twist_topic=/twist_bruteforce`
- TF (opcjonalnie): `odom_bruteforce -> base_link_bruteforce`

**Opis działania:**
- pełne przeszukiwanie zakresów (dx,dy,dθ); wolniejsze, ale stabilne jako referencja
- zwykle publikowane rzadziej parametrem `publish_every_n`

**Najważniejsze parametry:**
- `bf_range_xy`, `bf_range_th`, `bf_step_xy`, `bf_step_th`
- `publish_every_n`

<details> <summary><b>📄 Kliknij, aby zobaczyć Pseudokod (Bruteforce Grid Search)</b></summary>

```python
def run_scan_matcher_bruteforce():
    """
    Node: scan_matcher_bruteforce
    Param: method="bruteforce"
    Parametry z demo.launch.py (konfigurowalne)
    """
    # Zakresy przeszukiwania siatki (Grid Search)
    LIMIT_XY = 0.15   # parametr: bf_range_xy
    STEP_XY  = 0.01   # parametr: bf_step_xy
    LIMIT_TH = 0.25   # parametr: bf_range_th (rad)
    
    # Pętle po wszystkich możliwych kombinacjach przesunięcia
    # (To obciąża CPU, dlatego publish_every_n=5)
    best_score = -1
    best_transform = (0,0,0)

    for x in range(-LIMIT_XY, LIMIT_XY, STEP_XY):
        for y in range(-LIMIT_XY, LIMIT_XY, STEP_XY):
            for th in range(-LIMIT_TH, LIMIT_TH, 0.01):
                
                score = check_overlap(curr_scan, prev_scan, offset=(x,y,th))
                
                if score > best_score:
                    best_score = score
                    best_transform = (x, y, th)
    
    # Wyjście
    publish("/pose_bruteforce", integrate(best_transform))
```
</details>
### Tabela porównawcza 4 torów estymacji (wejścia/wyjścia + znaczenie)

> Legenda: (x, y, θ) = pozycja i orientacja robota; (Δx, Δy, Δθ) = przyrost ruchu między kolejnymi skanami LiDAR.

| Tor | Metoda (idea) | Node / implementacja | Wejścia (ROS) | Wyjścia (ROS) | Co oznaczają wyjścia (semantyka) |
|---|---|---|---|---|---|
| 1. Baseline SLAM | Klasyczny SLAM (mapowanie + lokalizacja) | `slam_toolbox_baseline` (`slam_toolbox/sync_slam_toolbox_node`) | `/scan_slam` + odometria/TF (dryf) | `/map` + TF `map -> odom` | **Mapa** otoczenia + **globalna trajektoria** robota w układzie mapy (pośrednio przez TF / wewn. estymator). |
| 2. AI SLAM | SLAM Toolbox z odometrią korygowaną przez AI | `slam_toolbox_ai` + `infer_node` | `/scan_slam` + odometria „AI” (`odom_ai`/TF z inferencji) | `/map_ai` + `/pose_ai` + TF (dla toru AI) | `/pose_ai` to **korygowana pozycja (x,y,θ)**. W tle AI estymuje korekcję ruchu (w praktyce odpowiadającą (Δx,Δy,Δθ)), integruje ją do pozycji i publikuje „lepszą” odometrię dla SLAM. | To jest Twój **główny tor** do obrony. |
| 3. Scan-to-scan (local) |Estymacja ruchu między skanami (szybka, lokalna)|`scan_matcher_local` (`scan_matcher.py`) | `/scan_slam` | `pose_topic=/pose_scanmatch`, `twist_topic=/twist_scanmatch` | Algorytm liczy **ruch między skanami**: (Δx,Δy,Δθ), a następnie **integruje** to do pozycji (x,y,θ) publikowanej w `/pose_scanmatch`. Dodatkowo `/twist_scanmatch` to prędkości: **v i ω** wyliczone z tych przyrostów. | 
| 4. Scan-to-scan (bruteforce) | Klasyczne dopasowanie skanów przez przeszukiwanie siatki (wolniejsze, referencyjne) | `scan_matcher_bruteforce` (`scan_matcher.py`) | `/scan_slam` | `pose_topic=/pose_bruteforce`, `twist_topic=/twist_bruteforce` | Analogicznie: w środku powstaje (Δx,Δy,Δθ) z przeszukiwania, a `/pose_bruteforce` to **zintegrowana pozycja (x,y,θ)**. `/twist_bruteforce` = **v i ω** z przyrostów. | 

## Moduł AI (dataset → trening → inferencja)
### FAZA 1: Dataset
**Node:** `dataset_recorder` (`ai_slam_ai/dataset_recorder.py`)
Zapisuje `dataset.npz` zawierający:
- `X_scan` – skan LiDAR (360 wartości)
- `X_odom` – odometria (np. x, y, θ)
- `Y` – wektor korekcji (3 wartości)

### FAZA 2: Trening
**Node:** `train_model` (`ai_slam_ai/train_model.py`)
- MLP: `363 → 256 → 128 → 64 → 3`
- zapis: `model.pt` + logi treningu

### FAZA 3: Inferencja
**Node:** `infer_node` (`ai_slam_ai/infer_node.py`)
- subskrybuje LiDAR + odometrię
- publikuje:
  - `/pose_ai` (PoseStamped – korekcja/tor AI)
  - odometrię AI (`odom_ai`) + TF `odom_ai -> base_link` (dla SLAM Toolbox AI)

---

## Ewaluacja i wyniki
**Node:** `eval_node` (`ai_slam_eval/eval_node.py`)

**Metryki:**
- RMSE trajektorii (x, y, θ) dla wszystkich torów
- IoU mapy dla `/map` i `/map_ai`

**Artefakty:**
- `results.json`
- wykresy: `trajectory.png`, `errors.png`, `maps.png`


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
rm -rf build/ install/ log/
source ~/SLAM_AI/.venv/bin/activate
rosdep install --from-paths src --ignore-src -r -y --skip-keys ament_python
colcon build --symlink-install
source /opt/ros/jazzy/setup.bash
source install/setup.bash
```

### 3. Uruchomienie

**Domyślny eksperyment (pełna konfiguracja):**
```bash

~/SLAM_AI/scripts/cleanup.sh
~/SLAM_AI/scripts/run_full_cycle.sh
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
- `rmse_xy_*` – RMSE pozycji (m) względem GT
- `rmse_theta_*` – RMSE orientacji (rad) względem GT
- `iou_map_baseline`, `iou_map_ai` – IoU mapy (0–1, wyżej = lepiej)

Dostępne sufiksy (w zależności od trybu):
- `baseline`, `ai`, `scanmatch`, `bruteforce`

---

## Architektura Systemu

### Pipeline Eksperymentu (tryb AI)

```
FAZA 1: Zbieranie danych (dataset_duration)
   Robot jeździ → LiDAR + Odometria + GT → dataset.npz

FAZA 2: Trening modelu
   dataset.npz → MLP (363→256→128→64→3) → model.pt

FAZA 3: Inferencja AI
   LiDAR + Odom → model.pt → /pose_ai (korekcja)

FAZA 4: Estymatory porównawcze (równolegle)
   slam_toolbox baseline → /map, /pose_baseline
   slam_toolbox + AI      → /map_ai, /pose_ai
   scan matcher (local)   → /pose_scanmatch
   scan matcher (BF)      → /pose_bruteforce

FAZA 5: Ewaluacja
   Porównanie wszystkich torów do GT → results.json + wykresy
```

### Moduły

**1. Moduł SLAM** (`ai_slam_bringup`)
- Algorytm: `slam_toolbox` (synchroniczny)
- Input: LaserScan (360 próbek) z `/scan_slam`
- Output: Mapa (`/map`) + pozycja robota (x, y, θ)

**2. Moduł AI** (`ai_slam_ai`)
- Zbieranie datasetu: `dataset_recorder.py`
- Trening modelu: `train_model.py` (MLP: 363→256→128→64→3)
- Inferencja: `infer_node.py` (publikuje na `/pose_ai`)

**3. Moduł Odometrii** (`ai_slam_bringup/odom_corruptor.py`)
- Ground truth: `/odom_raw` z Gazebo
- Z dryfem: `/odom` (symuluje błędy rzeczywiste)

**4. Moduł Scan Matching (porównawczy)** (`ai_slam_bringup/scan_matcher.py`)
- Input: `/scan_slam`
- Output: 
  - `local` → `/pose_scanmatch`, `/twist_scanmatch`
  - `bruteforce` → `/pose_bruteforce`, `/twist_bruteforce`

**5. Moduł Ewaluacji** (`ai_slam_eval`)
- Metryki: RMSE (x, y, θ) dla wszystkich torów + IoU map (`/map`, `/map_ai`)
- Output: `results.json`, wykresy PNG (`trajectory.png`, `errors.png`, `maps.png`)

---

## Model robota (SDF jako źródło prawdy)

Opis robota jest utrzymywany w SDF, a URDF jest generowany automatycznie, aby oba pliki były zawsze spójne.

- Źródło prawdy: `ai_slam_ws/src/ai_slam_description/models/diffbot.sdf`
- Plik wynikowy: `ai_slam_ws/src/ai_slam_description/urdf/diffbot.urdf`
- Generator: `scripts/generate_urdf_from_sdf.py`

Regeneracja URDF po zmianach w SDF:
```bash
cd ~/SLAM_AI
python3 scripts/generate_urdf_from_sdf.py
```

---

## Centralna Konfiguracja (YAML)

Wszystkie parametry eksperymentu znajdują się w jednym pliku YAML.

### Pliki konfiguracyjne

| Plik | Opis | Czas |
|------|------|------|
| `experiment_config.yaml` | Pełny eksperyment | ~2–3 min |
| `fast_test.yaml` | Szybki test | ~40 s |

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

---

## Wyniki eksperymentu

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

Przykład (format skrócony):

```json
{
  "mode": "ai",
  "metrics": {
    "rmse_xy_baseline": 0.1108,
    "rmse_theta_baseline": 0.0249,
    "rmse_xy_ai": 0.1214,
    "rmse_theta_ai": 0.0519,
    "rmse_xy_scanmatch": 3.5454,
    "rmse_theta_scanmatch": 1.2896,
    "rmse_xy_bruteforce": 3.1027,
    "rmse_theta_bruteforce": 1.9744,
    "iou_map_baseline": 0.0310,
    "iou_map_ai": 0.2192
  }
}
```

**Interpretacja IoU:**
- 0.0–0.1: słaba jakość mapy
- 0.1–0.3: średnia jakość
- 0.3+: dobra jakość

---

## Ważne topiki ROS

- `/scan` – LaserScan z Gazebo
- `/scan_slam` – LaserScan po normalizacji do 360 próbek
- `/odom_raw` – ground truth odometria
- `/odom` – odometria z dryfem

**SLAM:**
- `/map` – mapa baseline
- `/map_ai` – mapa AI

**Pozycje (porównywane w ewaluacji):**
- `/pose_baseline` – tor baseline
- `/pose_ai` – tor AI
- `/pose_scanmatch` – scan-to-scan (local)
- `/pose_bruteforce` – scan-to-scan (bruteforce)

**Dodatkowo:**
- `/twist_scanmatch`, `/twist_bruteforce` – estymowane (v, ω)
- `/cmd_vel` – komendy prędkości

---

## Troubleshooting

### „AI nie poprawia wyników”
- Zwiększ `dataset_duration` (więcej danych)
- Zwiększ `max_epochs` albo `patience` (dłuższy trening)
- Zmniejsz `learning_rate` (wolniejsze uczenie)
- Sprawdź czy model się wytrenował: `cat out/exp_*/train_history.json`

### „Scan matching daje duże błędy”
Warianty `/pose_scanmatch` i `/pose_bruteforce` są w tej chwili algorytmem porównawczym (prosta funkcja dopasowania na siatce). Jeśli wyniki są niestabilne:
- zmniejsz prędkości w `driver` (żeby kolejne skany bardziej się pokrywały),
- dla `bruteforce` zwiększ `publish_every_n` (np. 5–10),
- zawęź zakresy: `bf_range_xy`, `bf_range_th` i zwiększ rozdzielczości kroków,
- zwiększ `max_points` (albo zmniejsz, jeśli CPU nie wyrabia) oraz zmniejsz `max_use_range` (żeby odsiać dalekie, mniej informacyjne pomiary).


---

## Wymagania systemu
- Ubuntu 24.04
- ROS 2 Jazzy
- Gazebo Harmonic
- GPU opcjonalny

---

## Struktura projektu

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
│       │   └── launch/
│       │       └── demo.launch.py
│       ├── ai_slam_description/# Model robota (URDF/SDF)
│       ├── ai_slam_gazebo/     # Świat Gazebo
│       └── ai_slam_eval/       # Ewaluacja + mapa referencyjna
└── .venv/                      # Python virtual environment
```

---

## Narzędzia analizy

### Inspekcja datasetu
Po zebraniu danych (FAZA 1) można przeanalizować dataset za pomocą skryptu:

```bash
cd ~/SLAM_AI
source .venv/bin/activate
python3 scripts/inspect_dataset.py out/exp_YYYYMMDD_HHMMSS
```

Skrypt generuje:
- **Statystyki** - rozmiar datasetu, rozkład korekt (dx, dy, dθ)
- **Mapa LiDAR** - wizualizacja wszystkich skanów złożonych w mapę
- **Trajektoria** - ścieżka robota z odometrii
- **Histogram** - rozkład odległości z LiDAR
- **Wykres korekt** - rozrzut błędów pozycji

### Generowanie mapy referencyjnej
Mapa referencyjna (ground truth) jest generowana na podstawie pliku świata Gazebo:

```bash
cd ~/SLAM_AI
python scripts/generate_reference_map.py
```

Generuje `reference_map.pgm` i `reference_map.yaml` w `ai_slam_ws/src/ai_slam_eval/maps/`.

