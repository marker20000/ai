import json
import os
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import statistics

# Установка кодировки для Windows консоли
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

def analyze_dataset(filepath: str) -> Dict:
    """Анализ одного датасета"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not data:
        return None
    
    # Извлекаем все записи из groups
    all_records = []
    if 'groups' in data:
        for group in data['groups']:
            for subgroup in group:
                if isinstance(subgroup, list):
                    all_records.extend(subgroup)
                elif isinstance(subgroup, dict):
                    all_records.append(subgroup)
    else:
        if isinstance(data, list):
            all_records = data
        else:
            return None
    
    total_records = len(all_records)
    
    # Счетчики
    bot_moving_count = 0
    target_moving_count = 0
    
    bot_velocities = []
    target_velocities = []
    
    # Для корреляции
    target_moving_deltas = []
    target_static_deltas = []
    
    # Активное движение + большие коррекции
    active_tracking_count = 0
    
    for record in all_records:
        # Извлекаем данные
        own_vel_x = abs(record.get('ownVelX', 0))
        own_vel_y = abs(record.get('ownVelY', 0))
        own_vel_z = abs(record.get('ownVelZ', 0))
        
        tar_vel_x = record.get('tarVelX', 0)
        tar_vel_y = record.get('tarVelY', 0)
        tar_vel_z = record.get('tarVelZ', 0)
        
        delta_yaw = record.get('deltaYaw', 0)
        delta_pitch = record.get('deltaPitch', 0)
        
        # 1. Бот движется
        if own_vel_x > 0.1 or own_vel_y > 0.1 or own_vel_z > 0.1:
            bot_moving_count += 1
            bot_vel_magnitude = math.sqrt(own_vel_x**2 + own_vel_y**2 + own_vel_z**2)
            bot_velocities.append(bot_vel_magnitude)
        
        # 2. Цель движется
        tar_vel_abs_x = abs(tar_vel_x)
        tar_vel_abs_y = abs(tar_vel_y)
        tar_vel_abs_z = abs(tar_vel_z)
        
        target_is_moving = tar_vel_abs_x > 0.1 or tar_vel_abs_y > 0.1 or tar_vel_abs_z > 0.1
        
        if target_is_moving:
            target_moving_count += 1
            tar_vel_magnitude = math.sqrt(tar_vel_abs_x**2 + tar_vel_abs_y**2 + tar_vel_abs_z**2)
            target_velocities.append(tar_vel_magnitude)
        
        # 4. Корреляция между движением цели и величиной коррекции
        delta_magnitude = math.sqrt(delta_yaw**2 + delta_pitch**2)
        
        if target_is_moving:
            target_moving_deltas.append(delta_magnitude)
        else:
            target_static_deltas.append(delta_magnitude)
        
        # 5. Активное движение цели + большие коррекции
        horizontal_speed = math.sqrt(tar_vel_x**2 + tar_vel_z**2)
        large_correction = abs(delta_yaw) > 2.0 or abs(delta_pitch) > 2.0
        
        if horizontal_speed > 0.3 and large_correction:
            active_tracking_count += 1
    
    # Вычисляем статистику
    bot_moving_pct = (bot_moving_count / total_records * 100) if total_records > 0 else 0
    target_moving_pct = (target_moving_count / total_records * 100) if total_records > 0 else 0
    
    avg_bot_vel = statistics.mean(bot_velocities) if bot_velocities else 0
    avg_target_vel = statistics.mean(target_velocities) if target_velocities else 0
    
    # Корреляция (упрощенная - сравниваем средние)
    avg_moving_delta = statistics.mean(target_moving_deltas) if target_moving_deltas else 0
    avg_static_delta = statistics.mean(target_static_deltas) if target_static_deltas else 0
    
    # Вычисляем корреляцию Пирсона если есть данные
    correlation = None
    if len(target_moving_deltas) > 1 and len(target_static_deltas) > 1:
        # Простое сравнение: насколько больше коррекции при движущейся цели
        if avg_static_delta > 0:
            correlation_ratio = avg_moving_delta / avg_static_delta
        else:
            correlation_ratio = float('inf') if avg_moving_delta > 0 else 1.0
        correlation = correlation_ratio
    
    active_tracking_pct = (active_tracking_count / total_records * 100) if total_records > 0 else 0
    
    return {
        'filename': os.path.basename(filepath),
        'total_records': total_records,
        'bot_moving_pct': bot_moving_pct,
        'target_moving_pct': target_moving_pct,
        'avg_bot_vel': avg_bot_vel,
        'avg_target_vel': avg_target_vel,
        'avg_moving_delta': avg_moving_delta,
        'avg_static_delta': avg_static_delta,
        'correlation_ratio': correlation,
        'active_tracking_count': active_tracking_count,
        'active_tracking_pct': active_tracking_pct,
        'bot_moving_count': bot_moving_count,
        'target_moving_count': target_moving_count
    }

def main():
    datasets_dir = r"C:\Users\user2\Desktop\ai bot\fabric-example-mod-26.2\run\config\pvpbot\datasets"
    
    # Получаем все JSON файлы
    json_files = list(Path(datasets_dir).glob("*.json"))
    
    print(f"Найдено файлов: {len(json_files)}\n")
    print("="*100)
    
    all_stats = []
    
    for filepath in json_files:
        stats = analyze_dataset(str(filepath))
        if stats:
            all_stats.append(stats)
    
    # Сортируем по интересности (больше активного tracking)
    all_stats.sort(key=lambda x: x['active_tracking_count'], reverse=True)
    
    # Суммарная статистика
    total_records = sum(s['total_records'] for s in all_stats)
    total_bot_moving = sum(s['bot_moving_count'] for s in all_stats)
    total_target_moving = sum(s['target_moving_count'] for s in all_stats)
    total_active_tracking = sum(s['active_tracking_count'] for s in all_stats)
    
    print("\n📊 СУММАРНАЯ СТАТИСТИКА ПО ВСЕМ ФАЙЛАМ")
    print("="*100)
    print(f"Всего записей: {total_records:,}")
    print(f"Бот движется: {total_bot_moving:,} ({total_bot_moving/total_records*100:.1f}%)")
    print(f"Цель движется: {total_target_moving:,} ({total_target_moving/total_records*100:.1f}%)")
    print(f"Активный tracking (цель быстро движется + большие коррекции): {total_active_tracking:,} ({total_active_tracking/total_records*100:.1f}%)")
    
    avg_all_bot_vel = statistics.mean([s['avg_bot_vel'] for s in all_stats if s['avg_bot_vel'] > 0])
    avg_all_target_vel = statistics.mean([s['avg_target_vel'] for s in all_stats if s['avg_target_vel'] > 0])
    
    print(f"\nСредняя скорость бота (когда движется): {avg_all_bot_vel:.3f} м/тик")
    print(f"Средняя скорость цели (когда движется): {avg_all_target_vel:.3f} м/тик")
    
    # Корреляция по всем данным
    avg_correlation = statistics.mean([s['correlation_ratio'] for s in all_stats if s['correlation_ratio'] is not None and math.isfinite(s['correlation_ratio'])])
    print(f"\nСредняя корреляция (отношение коррекций для движущейся/статичной цели): {avg_correlation:.2f}x")
    
    print("\n" + "="*100)
    print("\n📁 ТОП-3 САМЫХ ИНТЕРЕСНЫХ ФАЙЛА (по количеству активного tracking)")
    print("="*100)
    
    for i, stats in enumerate(all_stats[:3], 1):
        print(f"\n{i}. {stats['filename']}")
        print(f"   Записей: {stats['total_records']:,}")
        print(f"   Бот движется: {stats['bot_moving_pct']:.1f}%")
        print(f"   Цель движется: {stats['target_moving_pct']:.1f}%")
        print(f"   Средняя скорость бота: {stats['avg_bot_vel']:.3f} м/тик")
        print(f"   Средняя скорость цели: {stats['avg_target_vel']:.3f} м/тик")
        print(f"   Средняя коррекция при движущейся цели: {stats['avg_moving_delta']:.2f}°")
        print(f"   Средняя коррекция при статичной цели: {stats['avg_static_delta']:.2f}°")
        if stats['correlation_ratio'] is not None and math.isfinite(stats['correlation_ratio']):
            print(f"   Корреляция (отношение): {stats['correlation_ratio']:.2f}x")
        print(f"   🎯 Активный tracking: {stats['active_tracking_count']:,} ({stats['active_tracking_pct']:.1f}%)")
    
    print("\n" + "="*100)
    print("\n📋 ВСЕ ФАЙЛЫ (краткая сводка)")
    print("="*100)
    print(f"{'Файл':<50} {'Записей':>10} {'Бот дв.%':>10} {'Цель дв.%':>10} {'Tracking':>10}")
    print("-"*100)
    
    for stats in all_stats:
        print(f"{stats['filename']:<50} {stats['total_records']:>10,} {stats['bot_moving_pct']:>9.1f}% {stats['target_moving_pct']:>9.1f}% {stats['active_tracking_pct']:>9.1f}%")
    
    print("\n" + "="*100)
    print("\n🎯 ВЫВОД:")
    print("="*100)
    
    if total_target_moving / total_records < 0.2:
        print("❌ Датасет содержит МАЛО примеров движущихся целей (<20%)")
        print("   Рекомендуется собрать больше данных с активно движущимися противниками.")
    elif total_target_moving / total_records < 0.5:
        print("⚠️  Датасет содержит УМЕРЕННОЕ количество примеров движущихся целей (20-50%)")
        print("   Для хорошего tracking желательно увеличить долю динамических сценариев.")
    else:
        print("✅ Датасет содержит ДОСТАТОЧНО примеров движущихся целей (>50%)")
    
    if total_active_tracking / total_records < 0.05:
        print("\n❌ КРИТИЧНО: Очень мало примеров активного tracking (<5%)")
        print("   Модель может плохо обучиться tracking быстрых целей.")
    elif total_active_tracking / total_records < 0.15:
        print("\n⚠️  Мало примеров активного tracking (5-15%)")
        print("   Рекомендуется добавить сценариев с быстрым strafing.")
    else:
        print("\n✅ Хорошее количество примеров активного tracking (>15%)")
    
    # Сохраняем результаты в файл
    with open('dataset_analysis_results.txt', 'w', encoding='utf-8') as f:
        f.write(f"АНАЛИЗ ДАТАСЕТОВ PVPBOT\n")
        f.write(f"="*100 + "\n\n")
        f.write(f"Всего записей: {total_records:,}\n")
        f.write(f"Бот движется: {total_bot_moving:,} ({total_bot_moving/total_records*100:.1f}%)\n")
        f.write(f"Цель движется: {total_target_moving:,} ({total_target_moving/total_records*100:.1f}%)\n")
        f.write(f"Активный tracking: {total_active_tracking:,} ({total_active_tracking/total_records*100:.1f}%)\n\n")
        
        f.write("ТОП-3 файла:\n")
        for i, stats in enumerate(all_stats[:3], 1):
            f.write(f"{i}. {stats['filename']}\n")
            f.write(f"   Активный tracking: {stats['active_tracking_count']} ({stats['active_tracking_pct']:.1f}%)\n")
    
    print("\n✅ Результаты сохранены в dataset_analysis_results.txt")

if __name__ == "__main__":
    main()
