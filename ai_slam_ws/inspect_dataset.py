#!/usr/bin/env python3
"""Skrypt do odczytu i inspekcji datasetu."""
import numpy as np
import matplotlib.pyplot as plt

# Wczytaj dataset
data = np.load('out/dataset.npz', allow_pickle=True)

# Pokaż dostępne klucze
print('Klucze w datasecie:', list(data.keys()))
print()

# Wczytaj dane
X_scan = data['X_scan']  # Skany LiDAR
X_odom = data['X_odom']  # Odometria (x, y, theta)
Y = data['Y']            # Ground truth korekty (dx, dy, dtheta)
meta = data['meta'].item()  # Metadane

print('X_scan shape:', X_scan.shape, '- skany LiDAR (n_samples x 360 promieni)')
print('X_odom shape:', X_odom.shape, '- odometria (n_samples x 3: x,y,theta)')
print('Y shape:', Y.shape, '- korekty GT (n_samples x 3: dx,dy,dtheta)')
print()
print('Metadane:', meta)
print()

# Statystyki korekt
print('=== Statystyki korekt Y ===')
print(f'dx:     mean={Y[:,0].mean():.6f}, std={Y[:,0].std():.6f}, min={Y[:,0].min():.6f}, max={Y[:,0].max():.6f}')
print(f'dy:     mean={Y[:,1].mean():.6f}, std={Y[:,1].std():.6f}, min={Y[:,1].min():.6f}, max={Y[:,1].max():.6f}')
print(f'dtheta: mean={Y[:,2].mean():.6f}, std={Y[:,2].std():.6f}, min={Y[:,2].min():.6f}, max={Y[:,2].max():.6f}')

# Funkcja do konwersji skanu na punkty w układzie globalnym
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

# Budowanie mapy ze wszystkich skanów - używając samej odometrii
print('\nBudowanie mapy ze wszystkich skanów...')

all_points_x = []
all_points_y = []

# Użyj wszystkich skanów z pozycjami z odometrii
for i in range(len(X_scan)):
    px, py = scan_to_points(X_scan[i], X_odom[i])
    all_points_x.extend(px)
    all_points_y.extend(py)

all_points_x = np.array(all_points_x)
all_points_y = np.array(all_points_y)
print(f'Liczba punktów mapy: {len(all_points_x)}')

# Tworzenie occupancy grid (heatmapa)
resolution = 0.05  # 5cm na piksel
x_min, x_max = all_points_x.min() - 0.5, all_points_x.max() + 0.5
y_min, y_max = all_points_y.min() - 0.5, all_points_y.max() + 0.5

grid, x_edges, y_edges = np.histogram2d(
    all_points_x, all_points_y,
    bins=[int((x_max - x_min) / resolution), int((y_max - y_min) / resolution)],
    range=[[x_min, x_max], [y_min, y_max]]
)

# Wizualizacja
fig = plt.figure(figsize=(16, 12))

# 1. Occupancy Grid Map
ax1 = fig.add_subplot(2, 2, 1)
# Normalizuj i pokaż jako obraz
grid_normalized = np.clip(grid.T, 0, np.percentile(grid, 95))  # Clip outliers
ax1.imshow(grid_normalized, extent=[x_min, x_max, y_min, y_max], origin='lower', 
           cmap='Greys', aspect='equal', interpolation='nearest')
ax1.plot(X_odom[:, 0], X_odom[:, 1], 'b-', linewidth=1.5, label='Trajektoria (odom)', alpha=0.8)
ax1.scatter(X_odom[0, 0], X_odom[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
ax1.scatter(X_odom[-1, 0], X_odom[-1, 1], c='red', s=100, marker='x', label='Koniec', zorder=5)
ax1.set_xlabel('X [m]')
ax1.set_ylabel('Y [m]')
ax1.set_title(f'Mapa LiDAR (z {len(X_scan)} skanów)')
ax1.legend(loc='upper right')
ax1.axis('equal')
ax1.grid(True, alpha=0.3)

# 2. Przykładowy skan LiDAR (widok polarny)
ax2 = fig.add_subplot(2, 2, 2, projection='polar')
angles = np.linspace(-np.pi, np.pi, 360, endpoint=False)
scan_idx = 0
ax2.plot(angles, X_scan[scan_idx], 'b.', markersize=1)
ax2.set_title(f'Skan LiDAR (próbka {scan_idx})')
ax2.set_rlim(0, 5)

# 3. Histogram odległości skanów
ax3 = fig.add_subplot(2, 2, 3)
ax3.hist(X_scan.flatten(), bins=50, edgecolor='black', alpha=0.7)
ax3.set_xlabel('Odległość [m]')
ax3.set_ylabel('Liczba pomiarów')
ax3.set_title('Histogram odległości LiDAR')

# 4. Rozkład korekt
ax4 = fig.add_subplot(2, 2, 4)
sc = ax4.scatter(Y[:, 0]*1000, Y[:, 1]*1000, c=np.arange(len(Y)), cmap='viridis', alpha=0.6, s=10)
ax4.set_xlabel('dx [mm]')
ax4.set_ylabel('dy [mm]')
ax4.set_title('Korekty pozycji (dx, dy)')
ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax4.axvline(x=0, color='k', linestyle='--', alpha=0.3)
ax4.grid(True)
cbar = plt.colorbar(sc, ax=ax4)
cbar.set_label('Numer próbki')

plt.tight_layout()
plt.savefig('out/dataset_analysis.png', dpi=150)
print('\nZapisano wizualizację: out/dataset_analysis.png')
plt.show()
print('Przykladowy skan (pierwsze 10 wartosci):', X_scan[0][:450])
print('Przykladowa odometria (x,y,theta):', X_odom[0])
print('Przykladowa korekta (dx,dy,dtheta):', Y[0])
