#!/usr/bin/env python3
"""Skrypt do odczytu i inspekcji datasetu z porównaniem do mapy referencyjnej."""
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
import argparse
# Ścieżki
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.join(SCRIPT_DIR, '..', 'ai_slam_ws')
REF_MAP_YAML = os.path.join(WORKSPACE_DIR, 'src', 'ai_slam_eval', 'maps', 'reference_map.yaml')
REF_MAP_PGM = os.path.join(WORKSPACE_DIR, 'src', 'ai_slam_eval', 'maps', 'reference_map.pgm')
DATASET_PATH = os.path.join(WORKSPACE_DIR, 'out', 'dataset.npz')
OUTPUT_PATH = os.path.join(WORKSPACE_DIR, 'out', 'dataset_analysis.png')

def load_reference_map(yaml_path, pgm_path):
    """Wczytuje mapę referencyjną z plików YAML i PGM."""
    # Wczytaj metadane
    with open(yaml_path, 'r') as f:
        meta = yaml.safe_load(f)
    
    resolution = meta['resolution']
    origin = meta['origin']  # [x, y, theta]
    
    # Wczytaj PGM (ASCII P2)
    with open(pgm_path, 'r') as f:
        lines = f.readlines()
    
    # Parsuj nagłówek PGM
    idx = 0
    while lines[idx].startswith('#') or lines[idx].strip() == '':
        idx += 1
    
    assert lines[idx].strip() == 'P2', "Oczekiwano formatu P2 (ASCII PGM)"
    idx += 1
    
    # Pomiń komentarze
    while lines[idx].startswith('#'):
        idx += 1
    
    width, height = map(int, lines[idx].split())
    idx += 1
    max_val = int(lines[idx].strip())
    idx += 1
    
    # Wczytaj dane pikseli
    pixels = []
    for line in lines[idx:]:
        pixels.extend(map(int, line.split()))
    
    grid = np.array(pixels, dtype=np.uint8).reshape((height, width))
    
    return grid, resolution, origin

def scan_to_points(scan, pose, max_range=5.0):
    """Konwertuje skan LiDAR na punkty (x, y) w układzie globalnym."""
    angles = np.linspace(-np.pi, np.pi, len(scan), endpoint=False)
    
    # Filtruj punkty poza zakresem
    valid = (scan > 0.1) & (scan < max_range)
    
    # Punkty w układzie robota
    local_x = scan[valid] * np.cos(angles[valid])
    local_y = scan[valid] * np.sin(angles[valid])
    
    # Transformacja do układu globalnego
    x, y, theta = pose
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    
    global_x = x + cos_t * local_x - sin_t * local_y
    global_y = y + sin_t * local_x + cos_t * local_y
    
    return global_x, global_y

def main():
    parser = argparse.ArgumentParser(
        description="Inspekcja datasetu i porównanie do mapy referencyjnej."
    )
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        default=os.path.join(WORKSPACE_DIR, "out"),
        help="Ścieżka do folderu eksperymentu (np. out/exp_20260103_151922). "
             "Domyślnie: ai_slam_ws/out"
    )
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    dataset_path = os.path.join(dataset_dir, "dataset.npz")
    output_path = os.path.join(dataset_dir, "dataset_analysis.png")

    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Nie istnieje folder datasetu: {dataset_dir}")
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Nie znaleziono pliku datasetu: {dataset_path}")

    # Wczytaj dataset
    print(f"Wczytywanie datasetu: {dataset_path}")
    data = np.load(dataset_path, allow_pickle=True)

    # Pokaż dostępne klucze
    print("Klucze w datasecie:", list(data.keys()))
    print()

    # Wczytaj dane
    X_scan = data["X_scan"]
    X_odom = data["X_odom"]
    Y = data["Y"]
    meta = data["meta"].item()

    print("X_scan shape:", X_scan.shape, "- skany LiDAR (n_samples x 360 promieni)")
    print("X_odom shape:", X_odom.shape, "- odometria (n_samples x 3: x,y,theta)")
    print("Y shape:", Y.shape, "- korekty GT (n_samples x 3: dx,dy,dtheta)")
    print()
    print("Metadane:", meta)
    print()
    # Statystyki korekt
    print('=== Statystyki korekt Y ===')
    print(f'dx:     mean={Y[:,0].mean():.6f}, std={Y[:,0].std():.6f}, min={Y[:,0].min():.6f}, max={Y[:,0].max():.6f}')
    print(f'dy:     mean={Y[:,1].mean():.6f}, std={Y[:,1].std():.6f}, min={Y[:,1].min():.6f}, max={Y[:,1].max():.6f}')
    print(f'dtheta: mean={Y[:,2].mean():.6f}, std={Y[:,2].std():.6f}, min={Y[:,2].min():.6f}, max={Y[:,2].max():.6f}')

    # Budowanie mapy ze wszystkich skanów
    print('\nBudowanie mapy ze wszystkich skanów...')

    all_points_x = []
    all_points_y = []

    for i in range(len(X_scan)):
        px, py = scan_to_points(X_scan[i], X_odom[i])
        all_points_x.extend(px)
        all_points_y.extend(py)

    all_points_x = np.array(all_points_x)
    all_points_y = np.array(all_points_y)
    print(f'Liczba punktów mapy: {len(all_points_x)}')

    # Wczytaj mapę referencyjną
    print(f'\nWczytywanie mapy referencyjnej: {REF_MAP_YAML}')
    ref_grid, ref_resolution, ref_origin = load_reference_map(REF_MAP_YAML, REF_MAP_PGM)
    print(f'Mapa referencyjna: {ref_grid.shape[1]}x{ref_grid.shape[0]} px, rozdzielczość: {ref_resolution} m/px')
    
    # Oblicz extent mapy referencyjnej
    ref_x_min = ref_origin[0]
    ref_y_min = ref_origin[1]
    ref_x_max = ref_x_min + ref_grid.shape[1] * ref_resolution
    ref_y_max = ref_y_min + ref_grid.shape[0] * ref_resolution

    # Tworzenie occupancy grid z LiDAR
    resolution = 0.05  # 5cm na piksel
    x_min, x_max = all_points_x.min() - 0.5, all_points_x.max() + 0.5
    y_min, y_max = all_points_y.min() - 0.5, all_points_y.max() + 0.5

    grid, x_edges, y_edges = np.histogram2d(
        all_points_x, all_points_y,
        bins=[int((x_max - x_min) / resolution), int((y_max - y_min) / resolution)],
        range=[[x_min, x_max], [y_min, y_max]]
    )

    # Wizualizacja - 3x2 layout
    fig = plt.figure(figsize=(16, 14))

    # 1. Mapa referencyjna (Ground Truth)
    ax1 = fig.add_subplot(2, 3, 1)
    # Konwertuj: 0=occupied(czarny), 254=free(biały), 205=unknown(szary)
    ref_display = np.where(ref_grid == 0, 0, np.where(ref_grid == 254, 1, 0.5))
    ax1.imshow(ref_display, extent=[ref_x_min, ref_x_max, ref_y_min, ref_y_max], 
               origin='lower', cmap='gray', vmin=0, vmax=1, aspect='equal')
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_title('Mapa Referencyjna (Ground Truth)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)

    # 2. Mapa LiDAR z trajektorią
    ax2 = fig.add_subplot(2, 3, 2)
    grid_normalized = np.clip(grid.T, 0, np.percentile(grid, 95))
    ax2.imshow(grid_normalized, extent=[x_min, x_max, y_min, y_max], origin='lower', 
               cmap='Greys', aspect='equal', interpolation='nearest')
    ax2.plot(X_odom[:, 0], X_odom[:, 1], 'b-', linewidth=1.5, label='Trajektoria', alpha=0.8)
    ax2.scatter(X_odom[0, 0], X_odom[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
    ax2.scatter(X_odom[-1, 0], X_odom[-1, 1], c='red', s=100, marker='x', label='Koniec', zorder=5)
    ax2.set_xlabel('X [m]')
    ax2.set_ylabel('Y [m]')
    ax2.set_title(f'Mapa LiDAR (z {len(X_scan)} skanów)')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-4, 4)

    # 3. Porównanie - mapa ref + punkty LiDAR
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(ref_display, extent=[ref_x_min, ref_x_max, ref_y_min, ref_y_max], 
               origin='lower', cmap='gray', vmin=0, vmax=1, aspect='equal', alpha=0.7)
    # Nałóż punkty LiDAR jako scatter (próbkowanie dla wydajności)
    sample_step = max(1, len(all_points_x) // 5000)
    ax3.scatter(all_points_x[::sample_step], all_points_y[::sample_step], 
                c='red', s=0.5, alpha=0.3, label='Punkty LiDAR')
    ax3.plot(X_odom[:, 0], X_odom[:, 1], 'b-', linewidth=1, label='Trajektoria', alpha=0.8)
    ax3.set_xlabel('X [m]')
    ax3.set_ylabel('Y [m]')
    ax3.set_title('Porównanie: Ref (szary) + LiDAR (czerwony)')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-4, 4)
    ax3.set_ylim(-4, 4)

    # 4. Przykładowy skan LiDAR (widok polarny)
    ax4 = fig.add_subplot(2, 3, 4, projection='polar')
    angles = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    scan_idx = 0
    ax4.plot(angles, X_scan[scan_idx], 'b.', markersize=1)
    ax4.set_title(f'Skan LiDAR (próbka {scan_idx})')
    ax4.set_rlim(0, 5)

    # 5. Histogram odległości skanów
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.hist(X_scan.flatten(), bins=50, edgecolor='black', alpha=0.7)
    ax5.set_xlabel('Odległość [m]')
    ax5.set_ylabel('Liczba pomiarów')
    ax5.set_title('Histogram odległości LiDAR')

    # 6. Rozkład korekt
    ax6 = fig.add_subplot(2, 3, 6)
    sc = ax6.scatter(Y[:, 0]*1000, Y[:, 1]*1000, c=np.arange(len(Y)), cmap='viridis', alpha=0.6, s=10)
    ax6.set_xlabel('dx [mm]')
    ax6.set_ylabel('dy [mm]')
    ax6.set_title('Korekty pozycji (dx, dy)')
    ax6.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax6.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax6.grid(True)
    cbar = plt.colorbar(sc, ax=ax6)
    cbar.set_label('Numer próbki')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nZapisano wizualizację: {output_path}")
    plt.show()

    print("\nPrzykładowy skan (pierwsze 10 wartości):", X_scan[0][:10])
    print("Przykładowa odometria (x,y,theta):", X_odom[0])
    print("Przykładowa korekta (dx,dy,dtheta):", Y[0])

if __name__ == '__main__':
    main()
