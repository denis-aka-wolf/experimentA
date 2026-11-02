
# -*- coding: utf-8 -*-
"""
ЭКСПЕРИМЕНТ C5

Локальная геометрическая сигнатура траектории
и аппроксимация скаляра.

Исследовательская задача:
Проверить, содержит ли последовательность целочисленных поправок
локальную геометрическую информацию о положении точки в цикле.

Скаляр не используется при построении признаков.
Прямой перебор скаляра для предсказания не применяется.

Зависимость:
    numpy

Установка:
    python -m pip install numpy

Запуск:
    python experimentC5.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


# ============================================================
# ПАРАМЕТРЫ КРИВОЙ
# ============================================================

P = 499
A = 0
B = 7

ORDER = 73
G = (3, 212)

WINDOW_LENGTHS = (3, 5, 7)

RIDGE_ALPHA = 10.0
PERMUTATIONS = 100

RANDOM_SEED = 20261023

ORIGIN_SHIFTS = (
    0, 1, 3, 7, 11, 18, 27, 36, 49, 61, 72
)

OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "experimentC5_results"
)


# ============================================================
# АРИФМЕТИКА ЭЛЛИПТИЧЕСКОЙ КРИВОЙ
# ============================================================

Point = tuple[int, int] | None


def inv_mod(value, modulus=P):
    value %= modulus

    if value == 0:
        raise ZeroDivisionError(
            "Нулевой знаменатель."
        )

    return pow(value, -1, modulus)


def point_add(p1: Point, p2: Point) -> Point:

    if p1 is None:
        return p2

    if p2 is None:
        return p1

    x1, y1 = p1
    x2, y2 = p2

    if x1 == x2 and (y1 + y2) % P == 0:
        return None

    if p1 == p2:

        numerator = (
            3 * x1 * x1 + A
        ) % P

        denominator = (
            2 * y1
        ) % P

    else:

        numerator = (
            y2 - y1
        ) % P

        denominator = (
            x2 - x1
        ) % P

    slope = (
        numerator * inv_mod(denominator)
    ) % P

    x3 = (
        slope * slope - x1 - x2
    ) % P

    y3 = (
        slope * (x1 - x3) - y1
    ) % P

    return x3, y3


def build_cycle():

    cycle = [None]

    current = None

    for _ in range(1, ORDER):

        current = point_add(current, G)

        cycle.append(current)

    if point_add(cycle[-1], G) is not None:

        raise RuntimeError(
            "Порядок генератора проверен некорректно."
        )

    if len(set(cycle[1:])) != ORDER - 1:

        raise RuntimeError(
            "В цикле обнаружены повторяющиеся точки."
        )

    return cycle


# ============================================================
# ЦЕЛОЧИСЛЕННЫЕ ПОПРАВКИ ПЕРЕХОДА
# ============================================================

def transition_corrections(point):

    """
    Вычисляет поправки перехода:

        P -> P + G

    q_slope:

        lambda * denominator
        = numerator + q_slope * P

    q_x:

        x_raw = x_next + q_x * P

    q_y:

        y_raw = y_next + q_y * P

    Переходы через бесконечность пропускаются.
    """

    if point is None:
        return None

    next_point = point_add(point, G)

    if next_point is None:
        return None

    x1, y1 = point

    x2, y2 = G

    x_next, y_next = next_point

    if point == G:

        numerator = (
            3 * x1 * x1 + A
        )

        denominator = (
            2 * y1
        )

    else:

        numerator = y2 - y1

        denominator = x2 - x1

    denominator_mod = denominator % P

    if denominator_mod == 0:
        return None

    lam = (
        numerator
        * inv_mod(denominator_mod)
    ) % P

    # Поправка наклона.

    q_slope_num = (
        lam * denominator - numerator
    )

    if q_slope_num % P != 0:
        raise ArithmeticError("Ошибка q_slope.")

    q_slope = q_slope_num // P

    # Поправка координаты X.

    x_raw = (
        lam * lam - x1 - x2
    )

    if (x_raw - x_next) % P != 0:
        raise ArithmeticError("Ошибка q_x.")

    q_x = (
        x_raw - x_next
    ) // P

    # Поправка координаты Y.

    y_raw = (
        lam * (x1 - x_next) - y1
    )

    if (y_raw - y_next) % P != 0:
        raise ArithmeticError("Ошибка q_y.")

    q_y = (
        y_raw - y_next
    ) // P

    return {
        "q_slope": int(q_slope),
        "q_x": int(q_x),
        "q_y": int(q_y),
        "lambda_mod_p": int(lam),
        "x": int(x1),
        "y": int(y1),
        "x_next": int(x_next),
        "y_next": int(y_next),
    }


def build_transition_map(cycle):

    result = {}

    for t in range(ORDER):

        rec = transition_corrections(
            cycle[t]
        )

        if rec is not None:
            result[t] = rec

    return result


# ============================================================
# ПОСТРОЕНИЕ ЛОКАЛЬНЫХ СИГНАТУР
# ============================================================

def build_dataset(
    cycle,
    transition_map,
    origin_shift,
    window_length,
    feature_kind
):

    radius = window_length // 2

    features = []
    targets = []

    for j in range(ORDER):

        absolute_t = (
            origin_shift + j
        ) % ORDER

        offsets = range(
            -radius,
            radius + 1
        )

        absolute_indices = [
            (absolute_t + off) % ORDER
            for off in offsets
        ]

        # Если окно пересекает переход через бесконечность,
        # образец не используется.

        if any(
            t not in transition_map
            for t in absolute_indices
        ):
            continue

        center_point = cycle[absolute_t]

        if center_point is None:
            continue

        center = transition_map[absolute_t]

        qy_window = [
            transition_map[t]["q_y"]
            for t in absolute_indices
        ]

        all_window = []

        for t in absolute_indices:

            rec = transition_map[t]

            all_window.extend([
                rec["q_slope"],
                rec["q_x"],
                rec["q_y"]
            ])

        if feature_kind == "coordinates_xy":

            row = [
                center["x"],
                center["y"]
            ]

        elif feature_kind == "center_qy":

            row = [
                center["q_y"]
            ]

        elif feature_kind == "window_qy":

            row = qy_window

        elif feature_kind == "window_all":

            row = all_window

        elif feature_kind == "xy_plus_window_all":

            row = [
                center["x"],
                center["y"]
            ] + all_window

        else:

            raise ValueError(
                f"Неизвестная модель: {feature_kind}"
            )

        features.append(row)

        # Это целевой скаляр относительно выбранного начала.
        # Он не входит в признаки.

        targets.append(j)

    return (
        np.asarray(features, dtype=float),
        np.asarray(targets, dtype=int)
    )


# ============================================================
# КРУГОВОЕ ПРЕДСТАВЛЕНИЕ СКАЛЯРА
# ============================================================

def encode_circular(targets):

    angle = (
        2.0 * np.pi * targets / ORDER
    )

    return np.column_stack([
        np.cos(angle),
        np.sin(angle)
    ])


def decode_circular(predictions):

    angles = np.arctan2(
        predictions[:, 1],
        predictions[:, 0]
    )

    values = (
        np.mod(angles, 2.0 * np.pi)
        * ORDER
        / (2.0 * np.pi)
    )

    return np.mod(
        np.rint(values).astype(int),
        ORDER
    )


def circular_distance(a, b):

    d = abs(int(a) - int(b)) % ORDER

    return min(d, ORDER - d)


# ============================================================
# LOO RIDGE
# ============================================================

def fit_predict_loo(
    X,
    y,
    alpha=RIDGE_ALPHA
):

    n = len(y)

    predictions = np.zeros(
        (n, 2),
        dtype=float
    )

    y_encoded = encode_circular(y)

    for test_i in range(n):

        train_mask = np.ones(
            n,
            dtype=bool
        )

        train_mask[test_i] = False

        X_train = X[train_mask]

        X_test = X[
            test_i:test_i + 1
        ]

        Y_train = y_encoded[train_mask]

        # Нормализация только по обучающей выборке.

        mean = X_train.mean(axis=0)

        std = X_train.std(axis=0)

        std[std < 1e-12] = 1.0

        X_train_s = (
            X_train - mean
        ) / std

        X_test_s = (
            X_test - mean
        ) / std

        # Центрирование целевых координат.

        train_mean_y = Y_train.mean(
            axis=0
        )

        Y_centered = (
            Y_train - train_mean_y
        )

        gram = (
            X_train_s.T @ X_train_s
        )

        weights = np.linalg.solve(
            gram + alpha * np.eye(
                gram.shape[0]
            ),
            X_train_s.T @ Y_centered
        )

        predictions[test_i] = (
            X_test_s @ weights
        )[0] + train_mean_y

    decoded = decode_circular(
        predictions
    )

    accuracy = float(
        np.mean(decoded == y)
    )

    mae = float(
        np.mean([
            circular_distance(a, b)
            for a, b in zip(decoded, y)
        ])
    )

    return decoded, accuracy, mae


# ============================================================
# ПЕРЕСТАНОВОЧНЫЙ КОНТРОЛЬ
# ============================================================

def permutation_test(
    X,
    y,
    observed_accuracy,
    permutations,
    rng
):

    scores = []

    for _ in range(permutations):

        shuffled = rng.permutation(y)

        _, acc, _ = fit_predict_loo(
            X,
            shuffled
        )

        scores.append(acc)

    scores = np.asarray(scores)

    p_value = (
        1.0 + np.sum(
            scores >= observed_accuracy
        )
    ) / (
        permutations + 1.0
    )

    return (
        float(scores.mean()),
        float(p_value)
    )


# ============================================================
# ЗАПУСК C5
# ============================================================

def main():

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Запуск эксперимента C5...")
    print("Локальные геометрические сигнатуры.")
    print("=" * 78)

    cycle = build_cycle()

    transition_map = build_transition_map(
        cycle
    )

    if len(transition_map) != 71:

        raise RuntimeError(
            f"Ожидалось 71 переход, "
            f"получено {len(transition_map)}."
        )

    # Проверка принадлежности точек кривой.

    for t in range(1, ORDER):

        x, y = cycle[t]

        if (
            y * y - (x * x * x + B)
        ) % P != 0:

            raise RuntimeError(
                f"Точка {t}G не лежит на кривой."
            )

    feature_kinds = (
        "coordinates_xy",
        "center_qy",
        "window_qy",
        "window_all",
        "xy_plus_window_all"
    )

    rows = []
    datasets = {}

    print(f"Кривая: y^2 = x^3 + {B} (mod {P})")
    print(f"G = {G}")
    print(f"Порядок = {ORDER}")
    print(f"Конечных переходов = {len(transition_map)}")
    print()

    print("ОСНОВНАЯ ПРОВЕРКА LOO")

    print(
        f"{'Окно':>5} "
        f"{'Модель':<25} "
        f"{'N':>4} "
        f"{'Acc':>9} "
        f"{'Circular MAE':>14}"
    )

    print("-" * 78)

    for window_length in WINDOW_LENGTHS:

        for feature_kind in feature_kinds:

            X, y = build_dataset(
                cycle,
                transition_map,
                0,
                window_length,
                feature_kind
            )

            _, acc, mae = fit_predict_loo(
                X,
                y
            )

            row = {
                "origin_shift": 0,
                "window_length": window_length,
                "feature_kind": feature_kind,
                "samples": len(y),
                "accuracy": acc,
                "circular_mae": mae
            }

            rows.append(row)

            datasets[
                (window_length, feature_kind)
            ] = (X, y)

            print(
                f"{window_length:>5} "
                f"{feature_kind:<25} "
                f"{len(y):>4} "
                f"{acc:>9.4f} "
                f"{mae:>14.3f}"
            )

    # Выбираем конфигурацию с минимальной круговой ошибкой.

    best = sorted(
        rows,
        key=lambda r: (
            r["circular_mae"],
            -r["accuracy"]
        )
    )[0]

    best_key = (
        best["window_length"],
        best["feature_kind"]
    )

    best_X, best_y = datasets[best_key]

    print()
    print("ПЕРЕСТАНОВОЧНЫЙ КОНТРОЛЬ")

    _, best_acc, best_mae = fit_predict_loo(
        best_X,
        best_y
    )

    perm_mean, perm_p = permutation_test(
        best_X,
        best_y,
        best_acc,
        PERMUTATIONS,
        rng
    )

    print(
        f"Лучшая конфигурация: {best_key}"
    )

    print(
        f"Наблюдаемая accuracy: {best_acc:.4f}"
    )

    print(
        f"Средняя accuracy случайных меток: "
        f"{perm_mean:.4f}"
    )

    print(
        f"Permutation p-value: {perm_p:.4f}"
    )

    print(
        f"Circular MAE: {best_mae:.3f}"
    )

    # Проверка смены начала отсчёта.

    print()
    print("ПРОВЕРКА СМЕНЫ НАЧАЛА ОТСЧЁТА")

    origin_rows = []

    print(
        f"{'Сдвиг':>10} "
        f"{'N':>5} "
        f"{'Accuracy':>12} "
        f"{'Circular MAE':>14}"
    )

    print("-" * 48)

    for shift in ORIGIN_SHIFTS:

        X, y = build_dataset(
            cycle,
            transition_map,
            shift,
            best_key[0],
            best_key[1]
        )

        _, acc, mae = fit_predict_loo(
            X,
            y
        )

        origin_rows.append({
            "origin_shift": shift,
            "samples": len(y),
            "accuracy": acc,
            "circular_mae": mae
        })

        print(
            f"{shift:>10} "
            f"{len(y):>5} "
            f"{acc:>12.4f} "
            f"{mae:>14.3f}"
        )

    # Центральный q_y против последовательности q_y.

    print()
    print("ЦЕНТРАЛЬНЫЙ q_y ПРОТИВ ОКНА q_y")

    comparison_rows = []

    for window_length in WINDOW_LENGTHS:

        for kind in (
            "center_qy",
            "window_qy"
        ):

            X, y = build_dataset(
                cycle,
                transition_map,
                0,
                window_length,
                kind
            )

            _, acc, mae = fit_predict_loo(
                X,
                y
            )

            comparison_rows.append({
                "window_length": window_length,
                "feature_kind": kind,
                "samples": len(y),
                "accuracy": acc,
                "circular_mae": mae
            })

            print(
                f"Окно={window_length}, "
                f"{kind}: "
                f"N={len(y)}, "
                f"Acc={acc:.4f}, "
                f"MAE={mae:.3f}"
            )

    # Сохранение CSV.

    def save_csv(filename, data):

        with (
            OUTPUT_DIR / filename
        ).open(
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=list(data[0].keys())
            )

            writer.writeheader()

            writer.writerows(data)

    save_csv(
        "c5_model_results.csv",
        rows
    )

    save_csv(
        "c5_origin_stability.csv",
        origin_rows
    )

    save_csv(
        "c5_center_vs_window.csv",
        comparison_rows
    )

    # Таблица поправок для ручной проверки.

    transition_rows = []

    for t in sorted(transition_map):

        rec = transition_map[t]

        transition_rows.append({
            "absolute_scalar": t,
            "x": rec["x"],
            "y": rec["y"],
            "x_next": rec["x_next"],
            "y_next": rec["y_next"],
            "lambda_mod_p": rec["lambda_mod_p"],
            "q_slope": rec["q_slope"],
            "q_x": rec["q_x"],
            "q_y": rec["q_y"]
        })

    save_csv(
        "c5_transition_corrections.csv",
        transition_rows
    )

    summary = {
        "experiment": "C5",
        "curve": "y^2 = x^3 + 7 mod 499",
        "generator": list(G),
        "order": ORDER,
        "finite_transitions": len(transition_map),
        "window_lengths": list(WINDOW_LENGTHS),
        "ridge_alpha": RIDGE_ALPHA,
        "permutations": PERMUTATIONS,
        "random_seed": RANDOM_SEED,
        "best_configuration": best,
        "permutation_control": {
            "observed_accuracy": best_acc,
            "permutation_mean_accuracy": perm_mean,
            "p_value": perm_p,
            "circular_mae": best_mae
        },
        "origin_shifts": origin_rows
    }

    (
        OUTPUT_DIR / "c5_summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print()
    print("Результаты сохранены:")
    print(OUTPUT_DIR)
    print("C5 завершён.")


if __name__ == "__main__":
    main()
