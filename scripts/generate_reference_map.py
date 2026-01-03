#!/usr/bin/env python3
"""
Generuje mapę referencyjną na podstawie świata Gazebo (ai_slam_world.sdf).
Mapa jest używana do obliczania IoU w ewaluacji.
"""

import numpy as np
import os

# Parametry mapy
RESOLUTION = 0.05  # 5cm na piksel (wyższa rozdzielczość)
ORIGIN_X = -3.5    # Początek mapy w metrach
ORIGIN_Y = -3.5
MAP_SIZE_M = 7.0   # 7m x 7m (trochę większe niż arena)

# Oblicz rozmiar w pikselach
MAP_SIZE_PX = int(MAP_SIZE_M / RESOLUTION)

# Elementy świata Gazebo (z ai_slam_world.sdf)
# Format: (x, y, width, height) w metrach
WALLS = [
    # Zewnętrzne ściany
    (0, 3.0, 6.0, 0.1),    # wall_north
    (0, -3.0, 6.0, 0.1),   # wall_south
    (3.0, 0, 0.1, 6.0),    # wall_east
    (-3.0, 0, 0.1, 6.0),   # wall_west
]

OBSTACLES = [
    # Przeszkody wewnętrzne
    (1.0, 1.0, 0.6, 0.6),    # obs_1
    (-1.0, 1.2, 0.5, 0.5),   # obs_2
    (1.5, -1.2, 0.7, 0.7),   # obs_3
    (-1.8, -1.5, 0.5, 0.5),  # obs_4
]

def world_to_pixel(x, y):
    """Konwertuje współrzędne świata na piksele mapy."""
    px = int((x - ORIGIN_X) / RESOLUTION)
    py = int((y - ORIGIN_Y) / RESOLUTION)
    return px, py

def draw_box(grid, cx, cy, w, h, value=0):
    """Rysuje prostokąt na mapie (value=0 to occupied/czarny)."""
    # Oblicz rogi
    x1, y1 = world_to_pixel(cx - w/2, cy - h/2)
    x2, y2 = world_to_pixel(cx + w/2, cy + h/2)
    
    # Clamp do granic mapy
    x1 = max(0, min(x1, MAP_SIZE_PX - 1))
    x2 = max(0, min(x2, MAP_SIZE_PX - 1))
    y1 = max(0, min(y1, MAP_SIZE_PX - 1))
    y2 = max(0, min(y2, MAP_SIZE_PX - 1))
    
    # Rysuj (y to wiersze, x to kolumny)
    grid[y1:y2+1, x1:x2+1] = value

def main():
    # Utwórz mapę - domyślnie nieznane (205 = szary)
    grid = np.full((MAP_SIZE_PX, MAP_SIZE_PX), 205, dtype=np.uint8)
    
    # Wypełnij wnętrze areny jako wolne (254 = biały)
    # Arena: -3 do 3 w obu osiach
    x1_arena, y1_arena = world_to_pixel(-3.0, -3.0)
    x2_arena, y2_arena = world_to_pixel(3.0, 3.0)
    grid[y1_arena:y2_arena, x1_arena:x2_arena] = 254
    
    # Rysuj ściany (occupied = 0 = czarny)
    for cx, cy, w, h in WALLS:
        draw_box(grid, cx, cy, w, h, value=0)
    
    # Rysuj przeszkody
    for cx, cy, w, h in OBSTACLES:
        draw_box(grid, cx, cy, w, h, value=0)
    
    # Odwróć oś Y (PGM ma Y rosnące w dół, świat ma Y rosnące w górę)
    grid = np.flipud(grid)
    
    # Ścieżka wyjściowa
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "ai_slam_ws", "src", "ai_slam_eval", "maps")
    os.makedirs(output_dir, exist_ok=True)
    
    pgm_path = os.path.join(output_dir, "reference_map.pgm")
    yaml_path = os.path.join(output_dir, "reference_map.yaml")
    
    # Zapisz PGM (ASCII format P2)
    with open(pgm_path, 'w') as f:
        f.write("P2\n")
        f.write(f"{MAP_SIZE_PX} {MAP_SIZE_PX}\n")
        f.write("255\n")
        for row in grid:
            f.write(" ".join(str(v) for v in row) + "\n")
    
    # Zapisz YAML
    with open(yaml_path, 'w') as f:
        f.write(f"image: reference_map.pgm\n")
        f.write(f"resolution: {RESOLUTION}\n")
        f.write(f"origin: [{ORIGIN_X}, {ORIGIN_Y}, 0.0]\n")
        f.write("negate: 0\n")
        f.write("occupied_thresh: 0.65\n")
        f.write("free_thresh: 0.196\n")
    
    print(f"Wygenerowano mapę referencyjną:")
    print(f"  PGM: {pgm_path}")
    print(f"  YAML: {yaml_path}")
    print(f"  Rozmiar: {MAP_SIZE_PX}x{MAP_SIZE_PX} px ({MAP_SIZE_M}x{MAP_SIZE_M} m)")
    print(f"  Rozdzielczość: {RESOLUTION} m/px")
    print(f"  Ściany: {len(WALLS)}, Przeszkody: {len(OBSTACLES)}")

if __name__ == "__main__":
    main()
