
# -*- coding: utf-8 -*-

"""
ЭКСПЕРИМЕНТ C13
Проективная геометрия конфигурации кратных точек.

Кривая:
    y^2 = x^3 + 7 (mod 499)

Генератор:
    G = (3, 212)

Порядок:
    73

Гипотеза:
    Скаляр n может проявляться в проективной конфигурации
    точек Q, 2Q, ..., 8Q.

Признаки:
    1. Исходные координаты Q.
    2. Координаты x кратных точек.
    3. Координаты x,y кратных точек.
    4. Наклоны всех хорд.
    5. Наклоны и свободные члены всех хорд.
    6. Проективные cross-ratio.
    7. Объединённые признаки.

Нет:
    - перебора кандидатов n;
    - таблицы дискретных логарифмов;
    - поиска Q среди заранее известных точек.

Модель:
    RBF Kernel Ridge Regression.

Контроль:
    6 фиксированных blocked CV-блоков;
    purge radius = 2;
    99 перестановок;
    поправка на выбор лучшей конфигурации.

Выход:
    C13_model_summary.csv
    C13_predictions.csv
    C13_folds.csv
    C13_permutations.csv
    C13_features.csv
    C13_report.txt
    experimentC13.zip
"""

import csv
import time
import random
import zipfile
import traceback

from pathlib import Path
from itertools import combinations

import numpy as np


# ============================================================
# 1. НАСТРОЙКИ
# ============================================================

P = 499
A = 0
B = 7

G = (3, 212)
ORDER = 73

MULTIPLIERS = list(range(1, 9))

GAMMAS = [0.1, 0.3, 1.0]
ALPHAS = [0.01, 0.1]

PERMUTATIONS = 99
PURGE_RADIUS = 2

RANDOM_SEED = 131313

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "experimentC13_results"

OUT_DIR.mkdir(exist_ok=True)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# 2. АРИФМЕТИКА ЭЛЛИПТИЧЕСКОЙ КРИВОЙ
# ============================================================

def inverse_mod(x, p=P):

    x %= p

    if x == 0:
        raise ZeroDivisionError(
            "Обратного элемента не существует."
        )

    return pow(x, -1, p)


def point_add(P1, P2):

    if P1 is None:
        return P2

    if P2 is None:
        return P1

    x1, y1 = P1
    x2, y2 = P2

    if x1 == x2 and (y1 + y2) % P == 0:
        return None

    if P1 == P2:

        if y1 % P == 0:
            return None

        numerator = (3 * x1 * x1 + A) % P
        denominator = (2 * y1) % P

    else:

        numerator = (y2 - y1) % P
        denominator = (x2 - x1) % P

    lam = (
        numerator * inverse_mod(denominator)
    ) % P

    x3 = (lam * lam - x1 - x2) % P

    y3 = (
        lam * (x1 - x3) - y1
    ) % P

    return x3, y3


def scalar_multiply(k, point):

    result = None
    addend = point

    while k > 0:

        if k & 1:
            result = point_add(result, addend)

        addend = point_add(addend, addend)

        k >>= 1

    return result


def is_on_curve(point):

    if point is None:
        return True

    x, y = point

    left = y * y % P

    right = (
        x * x * x + A * x + B
    ) % P

    return left == right


# ============================================================
# 3. ПОСТРОЕНИЕ ГРУППЫ
# ============================================================

def build_group():

    points = {}

    current = None

    for n in range(ORDER):

        points[n] = current

        current = point_add(current, G)

    if current is not None:
        raise RuntimeError(
            "Порядок генератора не равен 73."
        )

    for n in range(1, ORDER):

        if points[n] is None:
            raise RuntimeError(
                "Неожиданная точка на бесконечности."
            )

        if not is_on_curve(points[n]):
            raise RuntimeError(
                "Точка не принадлежит кривой."
            )

    unique_points = set(
        points[n]
        for n in range(1, ORDER)
    )

    if len(unique_points) != ORDER - 1:
        raise RuntimeError(
            "Обнаружены повторяющиеся точки."
        )

    return points


# ============================================================
# 4. ПРЕДСТАВЛЕНИЕ ЭЛЕМЕНТОВ ПОЛЯ
# ============================================================

def centered_residue(value):

    value %= P

    if value > P // 2:
        value -= P

    return value


def normalize_field(value):

    return centered_residue(value) / (P // 2)


# ============================================================
# 5. ГЕОМЕТРИЯ ХОРД
# ============================================================

def chord_parameters(P1, P2):

    """
    Возвращает параметры прямой:

        y = lambda*x + b

    Все вычисления выполняются по модулю P.
    """

    if P1 is None or P2 is None:
        raise ValueError(
            "Нельзя построить хорду через бесконечность."
        )

    x1, y1 = P1
    x2, y2 = P2

    if x1 == x2:

        raise ValueError(
            "Вертикальная хорда. "
            "Для выбранных множителей она не ожидается."
        )

    numerator = (y2 - y1) % P

    denominator = (x2 - x1) % P

    lam = (
        numerator * inverse_mod(denominator)
    ) % P

    intercept = (
        y1 - lam * x1
    ) % P

    return lam, intercept


# ============================================================
# 6. ПРОЕКТИВНОЕ ДВОЙНОЕ ОТНОШЕНИЕ
# ============================================================

def cross_ratio(x1, x2, x3, x4):

    """
    CR = ((x1-x3)*(x2-x4)) /
         ((x1-x4)*(x2-x3))

    Вычисление в проективной прямой над F_P.
    """

    numerator = (
        (x1 - x3) * (x2 - x4)
    ) % P

    denominator = (
        (x1 - x4) * (x2 - x3)
    ) % P

    if denominator == 0:

        raise ZeroDivisionError(
            "Нулевой знаменатель cross-ratio."
        )

    return (
        numerator * inverse_mod(denominator)
    ) % P


# ============================================================
# 7. ПОСТРОЕНИЕ ГЕОМЕТРИЧЕСКИХ ПРИЗНАКОВ
# ============================================================

def build_features(points):

    feature_names = [
        "xy_control",
        "orbit_x",
        "orbit_xy",
        "chord_slopes",
        "chord_lines",
        "cross_ratios",
        "projective_full",
        "complete_geometry"
    ]

    features = {
        name: []
        for name in feature_names
    }

    feature_rows = []

    # Все пары из восьми точек.
    pair_indices = list(
        combinations(range(8), 2)
    )

    # Все четверки из восьми точек.
    quadruple_indices = list(
        combinations(range(8), 4)
    )

    print(
        f"Количество хорд: {len(pair_indices)}"
    )

    print(
        f"Количество cross-ratio: "
        f"{len(quadruple_indices)}"
    )

    for scalar in range(1, ORDER):

        Q = points[scalar]

        x0, y0 = Q

        # Вычисляем только фиксированные кратные Q:
        # Q, 2Q, ..., 8Q.
        orbit = []

        for multiplier in MULTIPLIERS:

            point = scalar_multiply(
                multiplier,
                Q
            )

            if point is None:
                raise RuntimeError(
                    "Неожиданная точка на бесконечности."
                )

            if not is_on_curve(point):
                raise RuntimeError(
                    "Кратная точка не принадлежит кривой."
                )

            orbit.append(point)

        xs = [point[0] for point in orbit]
        ys = [point[1] for point in orbit]

        # ----------------------------------------------------
        # 1. Контрольные исходные координаты
        # ----------------------------------------------------

        xy_control = [
            normalize_field(x0),
            normalize_field(y0)
        ]

        # ----------------------------------------------------
        # 2. Координаты x орбиты
        # ----------------------------------------------------

        orbit_x = [
            normalize_field(x)
            for x in xs
        ]

        # ----------------------------------------------------
        # 3. Координаты x,y орбиты
        # ----------------------------------------------------

        orbit_xy = []

        for x, y in orbit:

            orbit_xy.append(
                normalize_field(x)
            )

            orbit_xy.append(
                normalize_field(y)
            )

        # ----------------------------------------------------
        # 4. Все наклоны хорд
        # ----------------------------------------------------

        chord_slopes = []

        # ----------------------------------------------------
        # 5. Наклоны и свободные члены
        # ----------------------------------------------------

        chord_lines = []

        for i, j in pair_indices:

            lam, intercept = chord_parameters(
                orbit[i],
                orbit[j]
            )

            chord_slopes.append(
                normalize_field(lam)
            )

            chord_lines.append(
                normalize_field(lam)
            )

            chord_lines.append(
                normalize_field(intercept)
            )

        # ----------------------------------------------------
        # 6. Cross-ratio
        # ----------------------------------------------------

        cross_ratios = []

        for i, j, k, l in quadruple_indices:

            value = cross_ratio(
                xs[i],
                xs[j],
                xs[k],
                xs[l]
            )

            cross_ratios.append(
                normalize_field(value)
            )

        # ----------------------------------------------------
        # 7. Проективное объединение
        # ----------------------------------------------------

        projective_full = (
            orbit_x
            + chord_slopes
            + cross_ratios
        )

        # ----------------------------------------------------
        # 8. Полное объединение
        # ----------------------------------------------------

        complete_geometry = (
            orbit_xy
            + chord_lines
            + cross_ratios
        )

        features["xy_control"].append(
            xy_control
        )

        features["orbit_x"].append(
            orbit_x
        )

        features["orbit_xy"].append(
            orbit_xy
        )

        features["chord_slopes"].append(
            chord_slopes
        )

        features["chord_lines"].append(
            chord_lines
        )

        features["cross_ratios"].append(
            cross_ratios
        )

        features["projective_full"].append(
            projective_full
        )

        features["complete_geometry"].append(
            complete_geometry
        )

        row = {
            "scalar": scalar,
            "x": x0,
            "y": y0
        }

        for i, point in enumerate(orbit, start=1):

            row[f"x_{i}"] = point[0]
            row[f"y_{i}"] = point[1]

        for index, value in enumerate(
            chord_slopes,
            start=1
        ):

            row[f"slope_{index}"] = value

        for index, value in enumerate(
            cross_ratios,
            start=1
        ):

            row[f"cross_ratio_{index}"] = value

        feature_rows.append(row)

    for name in feature_names:

        features[name] = np.asarray(
            features[name],
            dtype=float
        )

    return features, feature_rows


# ============================================================
# 8. КРУГОВОЕ ПРЕДСТАВЛЕНИЕ ЦЕЛИ
# ============================================================

def encode_targets(labels):

    angles = (
        2.0 * np.pi
        * np.asarray(labels, dtype=float)
        / ORDER
    )

    return np.column_stack([
        np.cos(angles),
        np.sin(angles)
    ])


def decode_targets(predictions):

    """
    Разрешены только ненулевые скаляры 1..72.
    """

    candidates = np.arange(1, ORDER)

    angles = (
        2.0 * np.pi
        * candidates
        / ORDER
    )

    vectors = np.column_stack([
        np.cos(angles),
        np.sin(angles)
    ])

    scores = predictions @ vectors.T

    return candidates[
        np.argmax(scores, axis=1)
    ]


def circular_distance(a, b):

    d = abs(int(a) - int(b))

    return min(d, ORDER - d)


# ============================================================
# 9. FIXED BLOCKED CROSS-VALIDATION
# ============================================================

def build_folds():

    all_indices = np.arange(ORDER - 1)

    blocks = np.array_split(
        all_indices,
        6
    )

    folds = []

    for fold_id, test_indices in enumerate(
        blocks,
        start=1
    ):

        excluded = set(
            int(i)
            for i in test_indices
        )

        for i in test_indices:

            for offset in range(
                -PURGE_RADIUS,
                PURGE_RADIUS + 1
            ):

                j = int(i) + offset

                if 0 <= j < ORDER - 1:
                    excluded.add(j)

        train_indices = np.array([
            i for i in all_indices
            if int(i) not in excluded
        ])

        test_indices = np.asarray(
            test_indices,
            dtype=int
        )

        folds.append((
            train_indices,
            test_indices
        ))

        print(
            f"Fold {fold_id}: "
            f"позиции [{test_indices[0]}, "
            f"{test_indices[-1] + 1}), "
            f"train={len(train_indices)}, "
            f"test={len(test_indices)}"
        )

    return folds


# ============================================================
# 10. RBF KERNEL RIDGE REGRESSION
# ============================================================

def standardize_train_test(X_train, X_test):

    mean = np.mean(
        X_train,
        axis=0
    )

    std = np.std(
        X_train,
        axis=0
    )

    std[std < 1e-12] = 1.0

    return (
        (X_train - mean) / std,
        (X_test - mean) / std
    )


def rbf_kernel(X1, X2, gamma):

    d2 = (
        np.sum(
            X1 * X1,
            axis=1
        )[:, None]
        +
        np.sum(
            X2 * X2,
            axis=1
        )[None, :]
        -
        2.0 * X1 @ X2.T
    )

    d2 = np.maximum(
        d2,
        0.0
    )

    d2 /= max(
        1,
        X1.shape[1]
    )

    return np.exp(
        -gamma * d2
    )


def evaluate_configuration(
    X,
    labels,
    folds,
    gamma,
    alpha
):

    target_vectors = encode_targets(
        labels
    )

    predictions_all = np.full(
        (len(labels), 2),
        np.nan,
        dtype=float
    )

    fold_records = []

    for fold_id, (
        train_idx,
        test_idx
    ) in enumerate(
        folds,
        start=1
    ):

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = target_vectors[train_idx]

        X_train, X_test = standardize_train_test(
            X_train,
            X_test
        )

        K_train = rbf_kernel(
            X_train,
            X_train,
            gamma
        )

        K_test = rbf_kernel(
            X_test,
            X_train,
            gamma
        )

        system = (
            K_train
            + alpha * np.eye(len(train_idx))
        )

        coefficients = np.linalg.solve(
            system,
            y_train
        )

        prediction = (
            K_test @ coefficients
        )

        predictions_all[test_idx] = prediction

        predicted_labels = decode_targets(
            prediction
        )

        true_labels = labels[test_idx]

        errors = [
            circular_distance(a, b)
            for a, b in zip(
                true_labels,
                predicted_labels
            )
        ]

        fold_records.append({
            "fold": fold_id,
            "test_count": len(test_idx),
            "accuracy": float(
                np.mean(
                    predicted_labels == true_labels
                )
            ),
            "mae": float(
                np.mean(errors)
            )
        })

    if np.isnan(predictions_all).any():

        raise RuntimeError(
            "Не все точки получили прогноз."
        )

    predicted_labels = decode_targets(
        predictions_all
    )

    errors = np.array([
        circular_distance(a, b)
        for a, b in zip(
            labels,
            predicted_labels
        )
    ])

    return {
        "accuracy": float(
            np.mean(
                predicted_labels == labels
            )
        ),
        "mae": float(
            np.mean(errors)
        ),
        "predictions": predicted_labels,
        "errors": errors,
        "fold_records": fold_records
    }


# ============================================================
# 11. ОЦЕНКА ВСЕХ КОНФИГУРАЦИЙ
# ============================================================

def evaluate_all(features, labels, folds):

    results = []

    for feature_name, X in features.items():

        for gamma in GAMMAS:

            for alpha in ALPHAS:

                result = evaluate_configuration(
                    X,
                    labels,
                    folds,
                    gamma,
                    alpha
                )

                results.append({
                    "model": feature_name,
                    "gamma": gamma,
                    "alpha": alpha,
                    "accuracy": result["accuracy"],
                    "mae": result["mae"],
                    "predictions": result["predictions"],
                    "errors": result["errors"],
                    "fold_records": result["fold_records"]
                })

    results.sort(
        key=lambda item: (
            item["mae"],
            -item["accuracy"]
        )
    )

    return results


# ============================================================
# 12. СОХРАНЕНИЕ CSV
# ============================================================

def write_csv(path, rows, fieldnames):

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(rows)


# ============================================================
# 13. ГЛАВНЫЙ ЭКСПЕРИМЕНТ
# ============================================================

def main():

    start_time = time.time()

    print("=" * 78)
    print("ЗАПУСК ЭКСПЕРИМЕНТА C13")
    print("Проективная геометрия конфигурации кратных точек")
    print("=" * 78)

    # --------------------------------------------------------
    # ШАГ 1
    # --------------------------------------------------------

    print("\n[1/9] Построение группы...")

    points = build_group()

    print(
        f"Кривая: y^2 = x^3 + 7 (mod {P})"
    )

    print(f"Генератор: {G}")
    print(f"Порядок: {ORDER}")
    print(f"Ненулевых точек: {ORDER - 1}")

    # --------------------------------------------------------
    # ШАГ 2
    # --------------------------------------------------------

    print("\n[2/9] Построение геометрических признаков...")

    features, feature_rows = build_features(
        points
    )

    labels = np.arange(
        1,
        ORDER
    )

    print(
        f"Общая выборка: {len(labels)}"
    )

    for name, X in features.items():

        print(
            f"{name:25s} "
            f"признаков={X.shape[1]:3d} "
            f"образцов={X.shape[0]:3d}"
        )

    # --------------------------------------------------------
    # ШАГ 3
    # --------------------------------------------------------

    print("\n[3/9] Построение CV-блоков...")

    folds = build_folds()

    fold_rows = []

    for fold_id, (
        train_idx,
        test_idx
    ) in enumerate(
        folds,
        start=1
    ):

        fold_rows.append({
            "fold": fold_id,
            "train_count": len(train_idx),
            "test_count": len(test_idx),
            "train_indices": ";".join(
                str(int(i))
                for i in train_idx
            ),
            "test_indices": ";".join(
                str(int(i))
                for i in test_idx
            )
        })

    # --------------------------------------------------------
    # ШАГ 4
    # --------------------------------------------------------

    print("\n[4/9] Подготовка и обучение моделей...")

    observed = evaluate_all(
        features,
        labels,
        folds
    )

    # --------------------------------------------------------
    # ШАГ 5
    # --------------------------------------------------------

    print("\n[5/9] РЕЗУЛЬТАТЫ BLOCKED CV")

    print("-" * 78)

    print(
        f"{'Модель':25s} "
        f"{'Gamma':>8s} "
        f"{'Alpha':>8s} "
        f"{'Accuracy':>11s} "
        f"{'Circular MAE':>14s}"
    )

    print("-" * 78)

    for item in observed:

        print(
            f"{item['model']:25s} "
            f"{item['gamma']:8.3f} "
            f"{item['alpha']:8.3f} "
            f"{item['accuracy']:11.4f} "
            f"{item['mae']:14.4f}"
        )

    best = observed[0]

    observed_best_mae = min(
        item["mae"]
        for item in observed
    )

    observed_max_accuracy = max(
        item["accuracy"]
        for item in observed
    )

    print("\nЛУЧШАЯ КОНФИГУРАЦИЯ ПО MAE:")

    print(f"Модель: {best['model']}")
    print(f"Gamma: {best['gamma']}")
    print(f"Alpha: {best['alpha']}")
    print(f"Circular MAE: {best['mae']:.4f}")
    print(f"Accuracy: {best['accuracy']:.4f}")

    # --------------------------------------------------------
    # ШАГ 6
    # --------------------------------------------------------

    print("\n[6/9] ПЕРЕСТАНОВОЧНЫЙ КОНТРОЛЬ")

    print(
        f"Количество перестановок: {PERMUTATIONS}"
    )

    print(
        "Одна и та же переставленная "
        "последовательность меток используется всеми моделями."
    )

    null_best_mae = []
    null_max_accuracy = []

    permutation_rows = []

    for permutation_id in range(
        1,
        PERMUTATIONS + 1
    ):

        permuted_labels = np.random.permutation(
            labels
        )

        permutation_results = evaluate_all(
            features,
            permuted_labels,
            folds
        )

        best_mae = min(
            item["mae"]
            for item in permutation_results
        )

        max_accuracy = max(
            item["accuracy"]
            for item in permutation_results
        )

        null_best_mae.append(
            best_mae
        )

        null_max_accuracy.append(
            max_accuracy
        )

        permutation_rows.append({
            "permutation": permutation_id,
            "best_mae": best_mae,
            "max_accuracy": max_accuracy
        })

        if (
            permutation_id % 10 == 0
            or permutation_id == PERMUTATIONS
        ):

            print(
                f"Перестановка "
                f"{permutation_id}/{PERMUTATIONS}"
            )

    null_best_mae = np.asarray(
        null_best_mae,
        dtype=float
    )

    null_max_accuracy = np.asarray(
        null_max_accuracy,
        dtype=float
    )

    # --------------------------------------------------------
    # ШАГ 7
    # --------------------------------------------------------

    print("\n[7/9] СТАТИСТИЧЕСКИЙ АНАЛИЗ")

    p_mae = (
        1 + np.sum(
            null_best_mae <= observed_best_mae
        )
    ) / (PERMUTATIONS + 1)

    p_accuracy = (
        1 + np.sum(
            null_max_accuracy >= observed_max_accuracy
        )
    ) / (PERMUTATIONS + 1)

    exact_baseline = 1.0 / (ORDER - 1)

    print("=" * 78)
    print("СТАТИСТИЧЕСКИЕ РЕЗУЛЬТАТЫ C13")
    print("=" * 78)

    print(
        f"Наблюдаемая лучшая MAE: "
        f"{observed_best_mae:.4f}"
    )

    print(
        f"Средняя случайная лучшая MAE: "
        f"{np.mean(null_best_mae):.4f}"
    )

    print(
        f"Минимальная случайная лучшая MAE: "
        f"{np.min(null_best_mae):.4f}"
    )

    print(
        f"Permutation p-value MAE: "
        f"{p_mae:.4f}"
    )

    print()

    print(
        f"Наблюдаемая максимальная Accuracy: "
        f"{observed_max_accuracy:.4f}"
    )

    print(
        f"Средняя случайная максимальная Accuracy: "
        f"{np.mean(null_max_accuracy):.4f}"
    )

    print(
        f"Permutation p-value Accuracy: "
        f"{p_accuracy:.4f}"
    )

    print()

    print(
        f"Случайный ориентир Accuracy: "
        f"1/72 = {exact_baseline:.4%}"
    )

    # --------------------------------------------------------
    # ШАГ 8
    # --------------------------------------------------------

    print("\n[8/9] СОХРАНЕНИЕ РЕЗУЛЬТАТОВ...")

    summary_rows = []

    for item in observed:

        summary_rows.append({
            "model": item["model"],
            "gamma": item["gamma"],
            "alpha": item["alpha"],
            "accuracy": item["accuracy"],
            "circular_mae": item["mae"]
        })

    write_csv(
        OUT_DIR / "C13_model_summary.csv",
        summary_rows,
        [
            "model",
            "gamma",
            "alpha",
            "accuracy",
            "circular_mae"
        ]
    )

    prediction_rows = []

    for i, true_label in enumerate(labels):

        prediction_rows.append({
            "true_scalar": int(true_label),
            "predicted_scalar": int(
                best["predictions"][i]
            ),
            "circular_error": int(
                best["errors"][i]
            )
        })

    write_csv(
        OUT_DIR / "C13_predictions.csv",
        prediction_rows,
        [
            "true_scalar",
            "predicted_scalar",
            "circular_error"
        ]
    )

    write_csv(
        OUT_DIR / "C13_folds.csv",
        fold_rows,
        [
            "fold",
            "train_count",
            "test_count",
            "train_indices",
            "test_indices"
        ]
    )

    write_csv(
        OUT_DIR / "C13_permutations.csv",
        permutation_rows,
        [
            "permutation",
            "best_mae",
            "max_accuracy"
        ]
    )

    feature_fieldnames = list(
        feature_rows[0].keys()
    )

    write_csv(
        OUT_DIR / "C13_features.csv",
        feature_rows,
        feature_fieldnames
    )

    # --------------------------------------------------------
    # ТЕКСТОВЫЙ ОТЧЁТ
    # --------------------------------------------------------

    elapsed = time.time() - start_time

    report = []

    report.append("ЭКСПЕРИМЕНТ C13")
    report.append(
        "Проективная геометрия конфигурации кратных точек"
    )

    report.append("")

    report.append(f"p = {P}")
    report.append(f"Кривая: y^2 = x^3 + 7 mod {P}")
    report.append(f"G = {G}")
    report.append(f"Порядок G = {ORDER}")
    report.append(f"Ненулевых точек = {ORDER - 1}")

    report.append("")

    report.append(
        "Кратные точки: Q, 2Q, ..., 8Q"
    )

    report.append(
        f"Количество хорд: {len(list(combinations(range(8), 2)))}"
    )

    report.append(
        f"Количество cross-ratio: "
        f"{len(list(combinations(range(8), 4)))}"
    )

    report.append("")

    report.append("ЛУЧШАЯ МОДЕЛЬ")

    report.append(f"Модель: {best['model']}")
    report.append(f"Gamma: {best['gamma']}")
    report.append(f"Alpha: {best['alpha']}")
    report.append(f"MAE: {best['mae']:.6f}")
    report.append(f"Accuracy: {best['accuracy']:.6f}")

    report.append("")

    report.append("ПЕРЕСТАНОВОЧНЫЙ КОНТРОЛЬ")

    report.append(
        f"Перестановок: {PERMUTATIONS}"
    )

    report.append(
        f"Observed best MAE: {observed_best_mae:.6f}"
    )

    report.append(
        f"Null mean MAE: {np.mean(null_best_mae):.6f}"
    )

    report.append(
        f"Null minimum MAE: {np.min(null_best_mae):.6f}"
    )

    report.append(
        f"p-value MAE: {p_mae:.6f}"
    )

    report.append("")

    report.append(
        f"Observed max accuracy: {observed_max_accuracy:.6f}"
    )

    report.append(
        f"Null mean accuracy: "
        f"{np.mean(null_max_accuracy):.6f}"
    )

    report.append(
        f"p-value accuracy: {p_accuracy:.6f}"
    )

    report.append("")

    report.append(
        f"Случайный точный baseline: "
        f"1/72 = {exact_baseline:.6%}"
    )

    report.append("")

    report.append("ВСЕ МОДЕЛИ")

    for item in observed:

        report.append(
            f"{item['model']}, "
            f"gamma={item['gamma']}, "
            f"alpha={item['alpha']}, "
            f"accuracy={item['accuracy']:.6f}, "
            f"MAE={item['mae']:.6f}"
        )

    report.append("")
    report.append(
        f"Время выполнения: {elapsed:.2f} сек."
    )

    with open(
        OUT_DIR / "C13_report.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(report)
        )

    # --------------------------------------------------------
    # ZIP
    # --------------------------------------------------------

    print("\n[9/9] СОЗДАНИЕ ZIP-АРХИВА...")

    zip_path = BASE_DIR / "experimentC13.zip"

    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED
    ) as archive:

        for file_path in OUT_DIR.iterdir():

            if file_path.is_file():

                archive.write(
                    file_path,
                    arcname=file_path.name
                )

    # --------------------------------------------------------
    # ЗАВЕРШЕНИЕ
    # --------------------------------------------------------

    print("\n" + "=" * 78)
    print("C13 ЗАВЕРШЁН")
    print("=" * 78)

    print("\nРЕЗУЛЬТАТЫ:")
    print(OUT_DIR)

    print("\nZIP-АРХИВ:")
    print(zip_path)

    print("\nЛУЧШАЯ МОДЕЛЬ:")
    print(best["model"])

    print(f"Circular MAE: {best['mae']:.4f}")
    print(f"Accuracy: {best['accuracy']:.4f}")

    print(f"Permutation p-value MAE: {p_mae:.4f}")
    print(
        f"Permutation p-value Accuracy: "
        f"{p_accuracy:.4f}"
    )

    if p_mae < 0.05 or p_accuracy < 0.05:

        print(
            "\nОБНАРУЖЕН СТАТИСТИЧЕСКИЙ СИГНАЛ."
        )

        print(
            "Необходимо проверить его устойчивость "
            "на других генераторах и кривых."
        )

    else:

        print(
            "\nЗначимого улучшения аппроксимации "
            "не обнаружено."
        )

    print(
        "\nЭксперимент не использует прямой перебор "
        "кандидатов на скаляр."
    )

    print(
        "Результат на игрушечной кривой не является "
        "доказательством эффективного решения ECDLP."
    )


# ============================================================
# 14. ЗАПУСК
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception:

        print("\nОШИБКА ВЫПОЛНЕНИЯ C13")

        traceback.print_exc()

        input("\nНажми Enter для выхода...")
