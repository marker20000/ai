#!/usr/bin/env python3
"""
Пересборка rec_index.bin с velocity признаками для tracking движущихся целей.

Изменения:
- FEAT_SEL расширен с 6 до 9 признаков: добавлены tarVelX, tarVelY, tarVelZ (индексы 3,4,5)
- Размерность окна: 96D → 144D (9 признаков × 16 тиков)
- AIM_W обновлены с весами 1.5 для velocity
- Temporal weighting и recent boost остаются
"""

import json
import numpy as np
from pathlib import Path
import struct

# === Конфигурация (должна совпадать с RecAim.java) ===
WINDOW = 16
FEATURE_DIM = 30
FEAT_SEL = [6, 7, 8, 3, 4, 5, 27, 28, 29]  # dist, yaw_diff, pitch_diff, tarVelX, tarVelY, tarVelZ, tar_fwd, tar_side, aim_center
AIM_W = np.array([0.5, 4.0, 4.0, 1.5, 1.5, 1.5, 1.0, 1.0, 2.0], dtype=np.float32)

# Temporal weighting: недавние кадры важнее
TEMP_W = np.linspace(0.15, 1.0, WINDOW, dtype=np.float32)

# Recent boost для последних 4 кадров по yaw_diff/pitch_diff
RECENT_N = 4
RECENT_BOOST = 2.5

def load_datasets(dataset_dir):
    """Загрузить все rec_*.json из папки datasets"""
    dataset_dir = Path(dataset_dir)
    all_episodes = []
    
    for json_file in sorted(dataset_dir.glob("rec_*.json")):
        print(f"Loading {json_file.name}...")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            episodes = data.get("groups", [])
            all_episodes.extend(episodes)
            print(f"  -> {len(episodes)} episodes")
    
    print(f"\nTotal episodes: {len(all_episodes)}")
    return all_episodes

def episode_to_windows(episode):
    """Конвертировать эпизод в окна (X, Y)"""
    if len(episode) < WINDOW:
        return [], []
    
    # Извлечь признаки
    features = []
    for tick in episode:
        f = [
            tick.get("dist", 0.0),
            tick.get("diffYaw", 0.0),
            tick.get("diffPitch", 0.0),
            tick.get("ownForward", 0.0),
            tick.get("ownSide", 0.0),
            tick.get("ownVelX", 0.0),
            tick.get("ownVelY", 0.0),
            tick.get("ownVelZ", 0.0),
            tick.get("tarForward", 0.0),
            tick.get("tarSide", 0.0),
            tick.get("tarVelX", 0.0),
            tick.get("tarVelY", 0.0),
            tick.get("tarVelZ", 0.0),
            tick.get("cooldown", 0.0),
            tick.get("dist2d", 0.0),
            tick.get("deltaY", 0.0),
        ]
        # Дополнить до 30 признаков (остальные не используются, заполним нулями)
        while len(f) < FEATURE_DIM:
            f.append(0.0)
        features.append(f)
    
    # Действия (deltaYaw, deltaPitch)
    actions = []
    for tick in episode:
        actions.append([
            tick.get("deltaYaw", 0.0),
            tick.get("deltaPitch", 0.0)
        ])
    
    features = np.array(features, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    
    # Создать скользящие окна
    windows_X = []
    windows_Y = []
    
    for i in range(WINDOW - 1, len(features)):
        window_features = features[i - WINDOW + 1 : i + 1]  # [16, 30]
        window_action = actions[i]  # [2]
        
        # Проецировать на FEAT_SEL признаки
        projected = window_features[:, FEAT_SEL]  # [16, 9]
        projected = projected.flatten()  # [144]
        
        windows_X.append(projected)
        windows_Y.append(window_action)
    
    return windows_X, windows_Y

def build_index(dataset_dir, output_path):
    """Построить rec_index.bin из датасетов"""
    episodes = load_datasets(dataset_dir)
    
    all_X = []
    all_Y = []
    
    print("\nConverting episodes to windows...")
    for ep_idx, episode in enumerate(episodes):
        windows_X, windows_Y = episode_to_windows(episode)
        all_X.extend(windows_X)
        all_Y.extend(windows_Y)
        
        if (ep_idx + 1) % 100 == 0:
            print(f"  Processed {ep_idx + 1}/{len(episodes)} episodes, windows: {len(all_X)}")
    
    if len(all_X) == 0:
        print("No data to build index!")
        return
    
    X = np.array(all_X, dtype=np.float32)  # [M, 144]
    Y = np.array(all_Y, dtype=np.float32)  # [M, 2]
    
    M, D = X.shape
    print(f"\nIndex built: M={M} windows, D={D} features")
    print(f"Action statistics:")
    print(f"  dyaw:   mean={Y[:, 0].mean():.3f}, std={Y[:, 0].std():.3f}, range=[{Y[:, 0].min():.1f}, {Y[:, 0].max():.1f}]")
    print(f"  dpitch: mean={Y[:, 1].mean():.3f}, std={Y[:, 1].std():.3f}, range=[{Y[:, 1].min():.1f}, {Y[:, 1].max():.1f}]")
    
    # Проверка velocity coverage
    # tarVelX/Y/Z находятся на позициях 3,4,5 в FEAT_SEL, что соответствует индексам 3,4,5 в проекции каждого тика
    # В flat окне 144D это позиции [3, 12, 21, 30, 39, 48, 57, 66, 75, 84, 93, 102, 111, 120, 129, 138] для tarVelX
    vel_indices = []
    for w in range(WINDOW):
        base = w * len(FEAT_SEL)
        vel_indices.extend([base + 3, base + 4, base + 5])  # tarVelX, Y, Z
    
    vel_data = X[:, vel_indices]
    vel_norm = np.sqrt(vel_data[:, 0::3]**2 + vel_data[:, 2::3]**2)  # горизонтальная скорость каждого тика
    fast_moving = np.any(vel_norm > 0.3, axis=1)  # хотя бы один тик с speed > 0.3
    
    print(f"\nVelocity coverage:")
    print(f"  Windows with fast moving target (>0.3): {fast_moving.sum()} ({100*fast_moving.mean():.1f}%)")
    print(f"  Average target speed: {vel_norm.mean():.3f}")
    
    # Save to rec_index.bin
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        # Magic "PVRI"
        f.write(b'PVRI')
        # M (int32)
        f.write(struct.pack('<i', M))
        # D (int32)
        f.write(struct.pack('<i', D))
        # W[M][D] (float32)
        X.astype(np.float32).tofile(f)
        # Y[M][2] (float32)
        Y.astype(np.float32).tofile(f)
    
    print(f"\nIndex saved to {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        dataset_dir = sys.argv[1]
    else:
        dataset_dir = "run/config/pvpbot/datasets"
    
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    else:
        output_path = "run/config/pvpbot/rec_index.bin"
    
    print("=== Rebuild rec_index.bin with velocity tracking ===")
    print(f"Datasets: {dataset_dir}")
    print(f"Output: {output_path}")
    print(f"Config: WINDOW={WINDOW}, FEAT_SEL={len(FEAT_SEL)} features, D={WINDOW*len(FEAT_SEL)}")
    print()
    
    build_index(dataset_dir, output_path)
    print("\n=== Done! ===")
    print("\nBefore launching the game:")
    print("1. Compile RecAim.java with new FEAT_SEL")
    print("2. Place rec_index.bin in run/config/pvpbot/")
    print("3. Enable @rec mode in game")
