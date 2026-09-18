# Улучшения tracking для @rec

## Проблема
Бот не успевал наводиться на движущиеся цели на малых углах (<30°). Анализ датасетов показал:
- ✅ 57.8% записей с движущейся целью (достаточно)
- ❌ Только 1.7% активного tracking (быстрая цель + большие коррекции) — критически мало
- Velocity данные есть в датасете, но не использовались в retrieval

## Реализованные улучшения

### 1. Velocity в FEAT_SEL (RecAim.java)
**Изменение:** Расширили проекцию с 96D до 144D, добавив `tarVelX/Y/Z` (индексы 3,4,5)

```java
// Было: 6 признаков × 16 тиков = 96D
private static final int[] FEAT_SEL = {6, 7, 8, 27, 28, 29};

// Стало: 9 признаков × 16 тиков = 144D
private static final int[] FEAT_SEL = {6, 7, 8, 3, 4, 5, 27, 28, 29};
// dist, yaw_diff, pitch_diff, tarVelX, tarVelY, tarVelZ, tar_fwd, tar_side, aim_center
```

**Эффект:** KNN теперь ищет соседей с **похожей скоростью цели**, не только с похожей ошибкой прицела. При velocity=0.4 он найдёт примеры tracking быстрых целей, а не статичных.

**Веса AIM_W:**
```java
private static final float[] AIM_W = {0.5f, 4.0f, 4.0f, 1.5f, 1.5f, 1.5f, 1.0f, 1.0f, 2.0f};
// dist, yaw_diff(доминирует), pitch_diff(доминирует), tarVelX, tarVelY, tarVelZ, tar_fwd, tar_side, aim_center
```

### 2. Velocity-aware weighting (RecAim.java)
**Изменение:** При быстром движении цели (>0.3 блока/тик) увеличиваем вес соседей с агрессивными действиями

```java
// В aim():
float tarSpeed = (float) Math.sqrt(tarVelX * tarVelX + tarVelZ * tarVelZ);
boolean fastTarget = tarSpeed > 0.3f;

for (int k = 0; k < K; k++) {
    float wgt = 1.0 / (Math.sqrt(bestD[k]) + 1e-3);
    
    if (fastTarget) {
        float actionMag = Math.abs(Y[bestI[k]][0]) + Math.abs(Y[bestI[k]][1]);
        wgt *= (1.0f + actionMag * 0.4f); // до ~2x для больших коррекций
    }
    
    dyaw += wgt * Y[bestI[k]][0];
    dpitch += wgt * Y[bestI[k]][1];
}
```

**Эффект:** Вместо усреднения всех top-5 (что могло давать ~0° при противоречивых соседях), мы предпочитаем **быстрые реакции** при движении цели.

### 3. Adaptive OOD fallback (RecAim.java)
**Изменение:** При OOD (нет похожих соседей) агрессивность fallback зависит от скорости цели

```java
if (bestD[0] > oodThreshold) {
    float k = 0.25f + Math.min(0.2f, tarSpeed * 0.5f); // 0.25-0.45
    return new float[]{
        Mth.clamp(yawDiffDeg * k, -Config.MAX_YAW_RT, Config.MAX_YAW_RT),
        Mth.clamp(pitchDiffDeg * k, -Config.MAX_YAW_RT, Config.MAX_YAW_RT)
    };
}
```

**Эффект:** Даже в OOD режиме бот реагирует быстрее на движущуюся цель.

### 4. Adaptive gain после retrieval (BotController.java)
**Изменение:** Дополнительно усиливаем выход rec при высокой скорости цели

```java
if (useRecAim && recAim != null) {
    float[] ra = recAim.aim(window);
    out[0] = ra[0];
    out[1] = ra[1];
    
    // Adaptive gain для tracking
    float tarSpeed = (float) Math.sqrt(tarVelX * tarVelX + tarVelZ * tarVelZ);
    if (tarSpeed > 0.25f) {
        float boost = 1.0f + Math.min(0.5f, tarSpeed * 0.8f); // 1.0x-1.5x
        out[0] *= boost;
        out[1] *= boost;
    }
}
```

**Эффект:** Двойное усиление — изнутри KNN (velocity-aware weighting) и снаружи (adaptive gain). При скорости 0.4 блока/тик получаем ~1.3x boost.

### 5. Lead compensation (BotController.java)
**Изменение:** Предсказываем позицию цели на 2 тика вперёд для компенсации lag

```java
// Предсказать позицию через 2 тика
Vec3 oppVel = opp.getDeltaMovement();
Vec3 predictedPos = oppAtF.add(oppVel.scale(2.0));

// Пересчитать yawDiff и pitchDiff к ПРЕДСКАЗАННОЙ позиции
double dx = predictedPos.x - self.getX();
double dy = (predictedPos.y + opp.getBbHeight() * 0.5) - selfEye.y;
double dz = predictedPos.z - self.getZ();
// ... расчёт desiredYaw/Pitch ...

// Заменить в state vector
s[7] = (float) (yawDiffPredicted / 180.0);
s[8] = (float) (pitchDiffPredicted / 90.0);
```

**Эффект:** Бот целится **куда цель будет**, а не где она сейчас. При скорости 0.4 блока/тик и 2 тика lag — это компенсация ~0.8 блока (~15° на дистанции 3).

## Пересборка индекса

Создан скрипт `rebuild_rec_index.py`:
```bash
python rebuild_rec_index.py
```

**Результат:**
- M=105,625 окон (было ~12,000 в старом анализе — значит новый индекс из всех датасетов)
- D=144 признака (было 96)
- **23.2% окон с быстро движущейся целью** (>0.3 блока/тик)
- Средняя скорость цели: 0.150 блока/тик
- Размер файла: 58.83 МБ

## Суммарный эффект улучшений

### Статичная цель (velocity ≈ 0):
- KNN ищет соседей среди статичных примеров
- Velocity-aware weighting не срабатывает
- Adaptive gain не срабатывает
- Lead compensation минимальна
- **Поведение близко к оригинальному**

### Медленная цель (0.1-0.3 блока/тик):
- KNN находит похожие примеры по velocity
- Lead compensation ~0.2-0.6 блока
- Адаптивный OOD: 0.25-0.35 коэффициент
- **~1.1-1.2x агрессивнее оригинала**

### Быстрая цель (>0.3 блока/тик):
- KNN находит примеры активного tracking
- Velocity-aware weighting: boost агрессивных соседей до 2x
- Adaptive gain: 1.3-1.5x усиление выхода
- Lead compensation: ~0.8-1.0 блока (2 тика × 0.4-0.5)
- Адаптивный OOD: 0.35-0.45 коэффициент
- **Суммарно: ~1.5-2x агрессивнее оригинала**

## Компиляция и тестирование

### 1. Скомпилировать мод
```bash
cd fabric-example-mod-26.2
./gradlew build
```

### 2. Проверить rec_index.bin
```bash
# Должен быть файл 58.83 МБ
ls -lh run/config/pvpbot/rec_index.bin
```

### 3. Запустить игру
```bash
./gradlew runClient
```

### 4. Включить @rec режим
В игре: `/pvpbot rec on`

### 5. Тестовые сценарии
- ✅ Статичная цель на малых углах (<20°) — должен работать как раньше
- ✅ Медленно strafing цель — должен немного улучшиться
- ✅✅✅ Быстро strafing цель (w-tap, diagonal sprint) — **главный тест**

## Диагностика

При проблемах проверить логи:
```
[pvpbot] rec_index.bin загружен: M=105625 D=144 oodThreshold=...
[pvpbot] цель=... d=... dyaw=... dpitch=... ood=0 knnD=...
[pvpbot] KNN top-5: #1 d=... yawDiff=... dyaw=... dpitch=...
```

Если `D=96` вместо `D=144` — индекс не пересобран или старый файл не заменён.

Если `ood=1` часто — датасет не покрывает текущие условия (особенно velocity).

## Следующие шаги (если tracking всё ещё недостаточен)

1. **Записать больше данных против быстрых целей** — сейчас только 23.2% окон с speed >0.3
2. **Separate index** для fastTarget — два индекса (slow/fast), выбор по tarSpeed
3. **Directional consistency filter** — не усреднять противоречивых соседей ([-10°, +8°] → взять группу с одним знаком)
4. **Увеличить RECENT_BOOST** с 2.5 до 3.5-4.0 для ещё более быстрой реакции
5. **Critic** — только после того как убедимся что KNN работает хорошо

## Отличия от оригинального плана

Оригинальный план предлагал:
1. ❌ Recovery-дуэли от ±90°/±120° — **невозможно записать**
2. ❌ weapon_ready в FEAT_SEL — **отложено** (сначала velocity)
3. ❌ Critic — **отложено** (сначала базовый tracking)

Вместо этого сделали:
1. ✅ Velocity tracking — **работает с существующими данными**
2. ✅ Lead compensation — **простая и эффективная**
3. ✅ Adaptive behaviors — **масштабируются по скорости цели**

Это более практичный подход: улучшаем то, что есть, вместо ожидания идеальных данных.
