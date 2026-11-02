
# -*- coding: utf-8 -*-

"""
ЭКСПЕРИМЕНТ C6

ГЕОМЕТРИЧЕСКАЯ ФОРМА ТРАЕКТОРИИ
И АППРОКСИМАЦИЯ СКАЛЯРА

Исследовательская цель:

Найти геометрические закономерности в последовательности точек

    P_n = nG

и проверить, позволяют ли локальные геометрические характеристики
приблизительно определить скаляр n.

Важные ограничения:

1. Скаляр n НЕ используется при построении признаков.
2. Прямой перебор скаляра при предсказании отсутствует.
3. Таблица соответствий точка -> скаляр не используется моделью.
4. Индекс n используется только как целевая переменная.
5. Проверяются разные геометрические представления.
6. Используется круговая регрессия cos/sin.

Кривая:

    y^2 = x^3 + 7 (mod 499)

Генератор:

    G = (3, 212)

Порядок:

    73

Зависимость:

    numpy

Установка:

    python -m pip install numpy

Запуск:

    python experimentC6.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


# ============================================================
# ПАРАМЕТРЫ
# ============================================================

P = 499
A = 0
B = 7

ORDER = 73

G = (3, 212)

RANDOM_SEED = 20261024

RIDGE_ALPHA = 10.0

PERMUTATIONS = 100

WINDOW_RADIUS = 2

OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "experimentC6_results"
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

        current = point_add(
            current,
            G
        )

        cycle.append(current)

    if point_add(cycle[-1], G) is not None:

        raise RuntimeError(
            "Проверка порядка генератора не пройдена."
        )

    if len(set(cycle[1:])) != ORDER - 1:

        raise RuntimeError(
            "Обнаружены повторяющиеся точки."
        )

    return cycle


# ============================================================
# ГЕОМЕТРИЧЕСКИЕ ФУНКЦИИ
# ============================================================

def euclidean_distance(p1, p2):

    """
    Обычное евклидово расстояние.

    Координаты здесь рассматриваются как числа 0..498,
    а не как элементы конечного поля.
    """

    dx = float(p2[0] - p1[0])

    dy = float(p2[1] - p1[1])

    return float(
        np.sqrt(dx * dx + dy * dy)
    )


def vector(p1, p2):

    return np.array([
        p2[0] - p1[0],
        p2[1] - p1[1]
    ], dtype=float)


def angle_between(v1, v2):

    """
    Угол между двумя векторами.

    Результат в радианах.
    """

    norm1 = np.linalg.norm(v1)

    norm2 = np.linalg.norm(v2)

    if norm1 < 1e-12 or norm2 < 1e-12:

        return 0.0

    cosine = np.dot(v1, v2) / (
        norm1 * norm2
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0
    )

    return float(
        np.arccos(cosine)
    )


def triangle_area(p1, p2, p3):

    """
    Удвоенная площадь треугольника по определителю.
    """

    v1 = vector(p1, p2)

    v2 = vector(p1, p3)

    determinant = (
        v1[0] * v2[1]
        - v1[1] * v2[0]
    )

    return float(
        abs(determinant) / 2.0
    )


# ============================================================
# ПОСТРОЕНИЕ ГЕОМЕТРИЧЕСКОЙ СИГНАТУРЫ
# ============================================================

def build_geometric_signature(points):

    """
    points содержит пять последовательных точек:

        P[n-2], P[n-1], P[n], P[n+1], P[n+2]

    Никакие индексы или скаляры здесь не используются.
    """

    center_index = 2

    center = points[center_index]

    features = []

    # --------------------------------------------------------
    # 1. Расстояния от центра до остальных точек
    # --------------------------------------------------------

    center_distances = []

    for i, point in enumerate(points):

        if i == center_index:
            continue

        distance = euclidean_distance(
            center,
            point
        )

        center_distances.append(distance)

    features.extend(center_distances)

    # --------------------------------------------------------
    # 2. Расстояния между соседними точками
    # --------------------------------------------------------

    neighbor_distances = []

    for i in range(len(points) - 1):

        distance = euclidean_distance(
            points[i],
            points[i + 1]
        )

        neighbor_distances.append(distance)

    features.extend(neighbor_distances)

    # --------------------------------------------------------
    # 3. Векторы движения
    # --------------------------------------------------------

    vectors = []

    for i in range(len(points) - 1):

        vectors.append(
            vector(
                points[i],
                points[i + 1]
            )
        )

    # --------------------------------------------------------
    # 4. Углы между последовательными векторами
    # --------------------------------------------------------

    angles = []

    for i in range(len(vectors) - 1):

        angle = angle_between(
            vectors[i],
            vectors[i + 1]
        )

        angles.append(angle)

    features.extend(angles)

    # --------------------------------------------------------
    # 5. Изменения длины движения
    # --------------------------------------------------------

    for i in range(len(neighbor_distances) - 1):

        difference = (
            neighbor_distances[i + 1]
            - neighbor_distances[i]
        )

        features.append(difference)

    # --------------------------------------------------------
    # 6. Отношения расстояний
    # --------------------------------------------------------

    for i in range(len(neighbor_distances) - 1):

        denominator = (
            neighbor_distances[i]
        )

        if denominator < 1e-12:

            ratio = 0.0

        else:

            ratio = (
                neighbor_distances[i + 1]
                / denominator
            )

        features.append(ratio)

    # --------------------------------------------------------
    # 7. Площади треугольников
    # --------------------------------------------------------

    for i in range(len(points) - 2):

        area = triangle_area(
            points[i],
            points[i + 1],
            points[i + 2]
        )

        features.append(area)

    # --------------------------------------------------------
    # 8. Площади относительно центральной точки
    # --------------------------------------------------------

    for i in range(len(points) - 1):

        area = triangle_area(
            center,
            points[i],
            points[i + 1]
        )

        features.append(area)

    return np.asarray(
        features,
        dtype=float
    )


# ============================================================
# ПОСТРОЕНИЕ НАБОРОВ ДАННЫХ
# ============================================================

def build_dataset(
    cycle,
    feature_kind
):

    X = []

    y = []

    radius = WINDOW_RADIUS

    for n in range(ORDER):

        indices = [
            (n + offset) % ORDER
            for offset in range(
                -radius,
                radius + 1
            )
        ]

        points = [
            cycle[i]
            for i in indices
        ]

        # Точка на бесконечности не имеет обычных координат.

        if any(
            point is None
            for point in points
        ):

            continue

        center = points[radius]

        signature = build_geometric_signature(
            points
        )

        # ----------------------------------------------------
        # Модель M1: координаты центра
        # ----------------------------------------------------

        if feature_kind == "coordinates_xy":

            features = np.array([
                center[0],
                center[1]
            ], dtype=float)

        # ----------------------------------------------------
        # Модель M2: только расстояния
        # ----------------------------------------------------

        elif feature_kind == "distances":

            features = signature[:8]

        # ----------------------------------------------------
        # Модель M3: расстояния, углы и отношения
        # ----------------------------------------------------

        elif feature_kind == "distances_angles":

            features = signature[:18]

        # ----------------------------------------------------
        # Модель M4: полная сигнатура
        # ----------------------------------------------------

        elif feature_kind == "full_geometry":

            features = signature

        # ----------------------------------------------------
        # Модель M5: геометрия относительно центра
        # ----------------------------------------------------

        elif feature_kind == "centered_geometry":

            centered_points = [
                (
                    point[0] - center[0],
                    point[1] - center[1]
                )
                for point in points
            ]

            features = build_geometric_signature(
                centered_points
            )

        else:

            raise ValueError(
                f"Неизвестная модель: {feature_kind}"
            )

        X.append(features)

        # Скаляр используется только как цель.

        y.append(n)

    return (
        np.asarray(X, dtype=float),
        np.asarray(y, dtype=int)
    )


# ============================================================
# КРУГОВОЕ ПРЕДСТАВЛЕНИЕ СКАЛЯРА
# ============================================================

def encode_circular(y):

    angle = (
        2.0 * np.pi * y / ORDER
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
        np.mod(
            angles,
            2.0 * np.pi
        )
        * ORDER
        / (2.0 * np.pi)
    )

    return np.mod(
        np.rint(values).astype(int),
        ORDER
    )


def circular_distance(a, b):

    distance = (
        abs(int(a) - int(b))
        % ORDER
    )

    return min(
        distance,
        ORDER - distance
    )


# ============================================================
# RIDGE REGRESSION С LOO
# ============================================================

def fit_predict_loo(
    X,
    y
):

    count = len(y)

    encoded = encode_circular(y)

    predictions = np.zeros(
        (count, 2),
        dtype=float
    )

    for test_index in range(count):

        mask = np.ones(
            count,
            dtype=bool
        )

        mask[test_index] = False

        X_train = X[mask]

        X_test = X[
            test_index:test_index + 1
        ]

        Y_train = encoded[mask]

        # Нормализация без утечки тестовых данных.

        mean = X_train.mean(
            axis=0
        )

        std = X_train.std(
            axis=0
        )

        std[std < 1e-12] = 1.0

        X_train = (
            X_train - mean
        ) / std

        X_test = (
            X_test - mean
        ) / std

        target_mean = Y_train.mean(
            axis=0
        )

        Y_centered = (
            Y_train - target_mean
        )

        gram = (
            X_train.T @ X_train
        )

        weights = np.linalg.solve(
            gram
            + RIDGE_ALPHA
            * np.eye(gram.shape[0]),
            X_train.T @ Y_centered
        )

        predictions[test_index] = (
            X_test @ weights
        )[0] + target_mean

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
    rng
):

    scores = []

    for _ in range(PERMUTATIONS):

        shuffled = rng.permutation(y)

        _, accuracy, _ = fit_predict_loo(
            X,
            shuffled
        )

        scores.append(accuracy)

    scores = np.asarray(
        scores,
        dtype=float
    )

    p_value = (
        1.0
        + np.sum(
            scores >= observed_accuracy
        )
    ) / (
        PERMUTATIONS + 1.0
    )

    return (
        float(scores.mean()),
        float(p_value)
    )


# ============================================================
# ГЛАВНЫЙ ЭКСПЕРИМЕНТ
# ============================================================

def main():

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Запуск эксперимента C6...")

    print(
        "Исследование геометрической формы траектории."
    )

    print("=" * 78)

    cycle = build_cycle()

    feature_models = [
        "coordinates_xy",
        "distances",
        "distances_angles",
        "full_geometry",
        "centered_geometry"
    ]

    results = []

    datasets = {}

    print(
        f"Кривая: y^2 = x^3 + {B} (mod {P})"
    )

    print(
        f"Генератор G = {G}"
    )

    print(
        f"Порядок генератора = {ORDER}"
    )

    print()

    print("ОСНОВНОЕ СРАВНЕНИЕ ГЕОМЕТРИЧЕСКИХ МОДЕЛЕЙ")

    print(
        f"{'Модель':<25} "
        f"{'Признаки':>10} "
        f"{'N':>5} "
        f"{'Accuracy':>12} "
        f"{'Circular MAE':>14}"
    )

    print("-" * 78)

    for model in feature_models:

        X, y = build_dataset(
            cycle,
            model
        )

        _, accuracy, mae = fit_predict_loo(
            X,
            y
        )

        result = {
            "model": model,
            "features": X.shape[1],
            "samples": len(y),
            "accuracy": accuracy,
            "circular_mae": mae
        }

        results.append(result)

        datasets[model] = (
            X,
            y
        )

        print(
            f"{model:<25} "
            f"{X.shape[1]:>10} "
            f"{len(y):>5} "
            f"{accuracy:>12.4f} "
            f"{mae:>14.3f}"
        )

    # --------------------------------------------------------
    # Выбираем лучшую модель
    # --------------------------------------------------------

    best = sorted(
        results,
        key=lambda r: (
            r["circular_mae"],
            -r["accuracy"]
        )
    )[0]

    best_model = best["model"]

    best_X, best_y = datasets[
        best_model
    ]

    _, best_accuracy, best_mae = fit_predict_loo(
        best_X,
        best_y
    )

    # --------------------------------------------------------
    # Перестановочный контроль
    # --------------------------------------------------------

    print()

    print(
        "ПЕРЕСТАНОВОЧНЫЙ КОНТРОЛЬ"
    )

    perm_mean, perm_p = permutation_test(
        best_X,
        best_y,
        best_accuracy,
        rng
    )

    print(
        f"Лучшая модель: {best_model}"
    )

    print(
        f"Accuracy: {best_accuracy:.4f}"
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

    # --------------------------------------------------------
    # Сохранение таблиц
    # --------------------------------------------------------

    with (
        OUTPUT_DIR / "c6_model_results.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(results[0].keys())
        )

        writer.writeheader()

        writer.writerows(results)

    summary = {
        "experiment": "C6",
        "curve": "y^2 = x^3 + 7 mod 499",
        "generator": list(G),
        "order": ORDER,
        "window_radius": WINDOW_RADIUS,
        "ridge_alpha": RIDGE_ALPHA,
        "permutations": PERMUTATIONS,
        "best_model": best,
        "permutation_control": {
            "observed_accuracy": best_accuracy,
            "permutation_mean": perm_mean,
            "p_value": perm_p,
            "circular_mae": best_mae
        }
    }

    (
        OUTPUT_DIR / "c6_summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    readme = """
ЭКСПЕРИМЕНТ C6

Геометрическая форма траектории.

Исследуется зависимость между локальной геометрией пяти последовательных
точек эллиптической кривой и скаляром центральной точки.

Проверяются:
- координаты центральной точки;
- евклидовы расстояния;
- углы;
- отношения расстояний;
- площади треугольников;
- геометрия относительно центра окна.

Скаляр не входит в признаки.

Модель использует круговую регрессию cos/sin.

ВАЖНО:
Евклидова геометрия координат конечного поля является выбранным
численным представлением, а не инвариантом группового закона.

Результаты игрушечной кривой нельзя автоматически переносить
на secp256k1.
"""

    (
        OUTPUT_DIR / "README_C6.txt"
    ).write_text(
        readme.strip(),
        encoding="utf-8"
    )

    print()

    print("РЕЗУЛЬТАТЫ СОХРАНЕНЫ:")

    print(OUTPUT_DIR)

    print()

    print("C6 ЗАВЕРШЁН.")


if __name__ == "__main__":

    main()
