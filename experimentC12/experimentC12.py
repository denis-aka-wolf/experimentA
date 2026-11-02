
# -*- coding: utf-8 -*-

"""
ЭКСПЕРИМЕНТ C12
Алгебраическая сигнатура точки через эллиптические полиномы деления.

Кривая:
    y^2 = x^3 + 7 (mod 499)

Генератор:
    G = (3, 212)

Порядок генератора:
    73

Цель:
    Аппроксимировать скаляр n по алгебраическим признакам точки Q=nG.

Запрещённые для модели операции:
    - прямой перебор скаляра n;
    - поиск Q в таблице дискретных логарифмов;
    - использование n при построении признаков.

Используются:
    - полиномы деления psi_m(Q), m=2..20;
    - центрированные остатки;
    - квадратичные характеры;
    - квадраты значений;
    - отношения соседних значений;
    - RBF Kernel Ridge Regression;
    - blocked cross-validation;
    - permutation test с поправкой на выбор лучшей модели.

Результаты:
    C12_model_summary.csv
    C12_predictions.csv
    C12_folds.csv
    C12_permutations.csv
    C12_features.csv
    C12_report.txt
    experimentC12.zip
"""

import os
import csv
import math
import time
import random
import zipfile
import traceback

from pathlib import Path

import numpy as np


# ============================================================
# 1. НАСТРОЙКИ
# ============================================================

P = 499
A = 0
B = 7

G = (3, 212)
ORDER = 73

MULTIPLIERS = list(range(2, 21))

GAMMAS = [0.1, 0.3, 1.0]
ALPHAS = [0.01, 0.1]

PERMUTATIONS = 99
PURGE_RADIUS = 2

RANDOM_SEED = 121212

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "experimentC12_results"

OUT_DIR.mkdir(exist_ok=True)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# 2. АРИФМЕТИКА ЭЛЛИПТИЧЕСКОЙ КРИВОЙ
# ============================================================

def inverse_mod(x, p=P):
    x %= p

    if x == 0:
        raise ZeroDivisionError("Обратного элемента не существует.")

    return pow(x, -1, p)


def point_add(P1, P2):
    """
    Сложение точек на кривой.

    None обозначает точку на бесконечности.
    """

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

    lam = numerator * inverse_mod(denominator) % P

    x3 = (lam * lam - x1 - x2) % P
    y3 = (lam * (x1 - x3) - y1) % P

    return x3, y3


def scalar_multiply(k, point):
    """
    Умножение точки на известное число k.

    В эксперименте эта функция применяется только
    для построения тестовой группы и проверки точек,
    а также для вычисления фиксированных кратных Q
    не используется при построении признаков.
    """

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

    left = (y * y) % P
    right = (x * x * x + A * x + B) % P

    return left == right


# ============================================================
# 3. ПОЛИНОМЫ ДЕЛЕНИЯ
# ============================================================

def division_polynomials(point, max_n=20):
    """
    Вычисляет psi_0 ... psi_max_n в поле F_P.

    Используются рекуррентные формулы полиномов деления.

    Базовые значения для y^2 = x^3 + 7:

        psi_0 = 0
        psi_1 = 1
        psi_2 = 2y
        psi_3 = 3x^4 + 12Bx
        psi_4 = 4y(x^6 + 5Bx^3 - B^2)

    Рекурсии:

    psi_(2m+1) =
        psi_(m+2) * psi_m^3
        - psi_(m-1) * psi_(m+1)^3

    psi_(2m) =
        psi_m *
        (psi_(m+2)*psi_(m-1)^2
        - psi_(m-2)*psi_(m+1)^2)
        / (2y)

    Все операции выполняются по модулю P.
    """

    if point is None:
        raise ValueError("Для точки на бесконечности признаки не строятся.")

    x, y = point

    x %= P
    y %= P

    if y == 0:
        raise ValueError(
            "Обнаружена точка с y=0. "
            "Формула рекурсии требует отдельной обработки."
        )

    memo = {}

    def psi(n):

        if n == 0:
            return 0

        if n == 1:
            return 1

        if n < 0:
            m = -n
            value = psi(m)

            # psi(-m) = (-1)^(m+1) * psi(m)
            if m % 2 == 0:
                return (-value) % P

            return value % P

        if n in memo:
            return memo[n]

        if n == 2:
            value = (2 * y) % P

        elif n == 3:
            value = (
                3 * pow(x, 4, P)
                + 12 * B * x
            ) % P

        elif n == 4:
            polynomial = (
                pow(x, 6, P)
                + 5 * B * pow(x, 3, P)
                - B * B
            ) % P

            value = (4 * y * polynomial) % P

        elif n % 2 == 1:

            m = (n - 1) // 2

            value = (
                psi(m + 2) * pow(psi(m), 3, P)
                -
                psi(m - 1) * pow(psi(m + 1), 3, P)
            ) % P

        else:

            m = n // 2

            numerator = (
                psi(m + 2) * pow(psi(m - 1), 2, P)
                -
                psi(m - 2) * pow(psi(m + 1), 2, P)
            ) % P

            denominator = (2 * y) % P

            value = (
                psi(m)
                * numerator
                * inverse_mod(denominator)
            ) % P

        memo[n] = value

        return value

    result = {}

    for n in range(max_n + 1):
        result[n] = psi(n)

    return result


# ============================================================
# 4. ПОСТРОЕНИЕ ИГРУШЕЧНОЙ ГРУППЫ
# ============================================================

def build_group():

    points = {}

    current = None

    for n in range(ORDER):

        points[n] = current
        current = point_add(current, G)

    if current is not None:
        raise RuntimeError("Порядок генератора не равен 73.")

    if points[0] is not None:
        raise RuntimeError("Неверная точка на бесконечности.")

    for n in range(1, ORDER):

        if points[n] is None:
            raise RuntimeError("Неожиданная точка на бесконечности.")

        if not is_on_curve(points[n]):
            raise RuntimeError("Точка не принадлежит кривой.")

    unique = set(points[n] for n in range(1, ORDER))

    if len(unique) != ORDER - 1:
        raise RuntimeError("В группе обнаружены повторяющиеся точки.")

    return points


# ============================================================
# 5. ПРЕОБРАЗОВАНИЯ ПОЛЕВЫХ ЗНАЧЕНИЙ
# ============================================================

def centered_residue(value):

    value %= P

    if value > P // 2:
        value -= P

    return value


def normalized_centered(value):

    return centered_residue(value) / (P // 2)


def legendre_symbol(value):

    value %= P

    if value == 0:
        return 0

    result = pow(value, (P - 1) // 2, P)

    if result == 1:
        return 1

    if result == P - 1:
        return -1

    raise RuntimeError("Неожиданное значение символа Лежандра.")


# ============================================================
# 6. ПОСТРОЕНИЕ АЛГЕБРАИЧЕСКИХ ПРИЗНАКОВ
# ============================================================

def build_features(points):

    names = [
        "xy_control",
        "psi_centered",
        "psi_legendre",
        "psi_squared",
        "psi_adjacent_ratio",
        "psi_centered_xy",
        "psi_mixed"
    ]

    features = {name: [] for name in names}

    feature_rows = []

    for n in range(1, ORDER):

        Q = points[n]

        x, y = Q

        psi_values = division_polynomials(
            Q,
            max_n=max(MULTIPLIERS)
        )

        raw = [
            psi_values[m]
            for m in MULTIPLIERS
        ]

        # Значения psi_m в центрированном представлении.
        centered = [
            normalized_centered(v)
            for v in raw
        ]

        # Квадратичные характеры.
        characters = [
            float(legendre_symbol(v))
            for v in raw
        ]

        # Квадраты значений.
        squares = [
            normalized_centered(v * v)
            for v in raw
        ]

        # Отношения соседних полиномов:
        # psi_m / psi_(m-1).
        ratios = []

        for m in range(3, max(MULTIPLIERS) + 1):

            numerator = psi_values[m]
            denominator = psi_values[m - 1]

            if denominator == 0:
                raise RuntimeError(
                    f"Нулевой знаменатель psi_{m - 1}."
                )

            ratio = (
                numerator * inverse_mod(denominator)
            ) % P

            ratios.append(normalized_centered(ratio))

        xy = [
            normalized_centered(x),
            normalized_centered(y)
        ]

        mixed = centered + characters + ratios

        features["xy_control"].append(xy)

        features["psi_centered"].append(centered)

        features["psi_legendre"].append(characters)

        features["psi_squared"].append(squares)

        features["psi_adjacent_ratio"].append(ratios)

        features["psi_centered_xy"].append(
            xy + centered
        )

        features["psi_mixed"].append(mixed)

        row = {
            "scalar": n,
            "x": x,
            "y": y
        }

        for m in MULTIPLIERS:
            row[f"psi_{m}"] = psi_values[m]

        feature_rows.append(row)

    for name in names:
        features[name] = np.asarray(
            features[name],
            dtype=float
        )

    return features, feature_rows


# ============================================================
# 7. ЦЕЛЕВОЕ КРУГОВОЕ ПРЕДСТАВЛЕНИЕ
# ============================================================

def encode_targets(labels):

    labels = np.asarray(labels, dtype=float)

    angles = 2.0 * np.pi * labels / ORDER

    return np.column_stack([
        np.cos(angles),
        np.sin(angles)
    ])


def decode_targets(predictions):

    """
    Декодирование выполняется сравнением с 72 допустимыми
    ненулевыми скалярами.

    Скаляр 0 не включён в множество допустимых ответов.
    """

    candidates = np.arange(1, ORDER)

    angles = 2.0 * np.pi * candidates / ORDER

    target_vectors = np.column_stack([
        np.cos(angles),
        np.sin(angles)
    ])

    scores = predictions @ target_vectors.T

    indices = np.argmax(scores, axis=1)

    return candidates[indices]


def circular_distance(a, b):

    d = abs(int(a) - int(b))

    return min(d, ORDER - d)


# ============================================================
# 8. ФИКСИРОВАННЫЕ BLOCKED CV-БЛОКИ
# ============================================================

def build_folds():

    all_indices = np.arange(ORDER - 1)

    blocks = np.array_split(all_indices, 6)

    folds = []

    for fold_id, test_indices in enumerate(blocks, start=1):

        test_set = set(int(i) for i in test_indices)

        excluded = set(test_set)

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
            f"test={len(test_indices)}, "
            f"train={len(train_indices)}"
        )

    return folds


# ============================================================
# 9. RBF KERNEL RIDGE REGRESSION
# ============================================================

def standardize_train_test(X_train, X_test):

    mean = np.mean(X_train, axis=0)

    std = np.std(X_train, axis=0)

    std[std < 1e-12] = 1.0

    return (
        (X_train - mean) / std,
        (X_test - mean) / std
    )


def rbf_kernel(X1, X2, gamma):

    d2 = (
        np.sum(X1 * X1, axis=1)[:, None]
        +
        np.sum(X2 * X2, axis=1)[None, :]
        -
        2.0 * X1 @ X2.T
    )

    d2 = np.maximum(d2, 0.0)

    # Деление на размерность делает gamma
    # более сопоставимым между разными наборами признаков.
    d2 /= max(1, X1.shape[1])

    return np.exp(-gamma * d2)


def evaluate_configuration(
    X,
    labels,
    folds,
    gamma,
    alpha
):

    target_vectors = encode_targets(labels)

    predictions_all = np.full(
        (len(labels), 2),
        np.nan,
        dtype=float
    )

    fold_records = []

    for fold_id, (train_idx, test_idx) in enumerate(
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

        prediction = K_test @ coefficients

        predictions_all[test_idx] = prediction

        predicted_labels = decode_targets(prediction)

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
                np.mean(predicted_labels == true_labels)
            ),
            "mae": float(np.mean(errors))
        })

    if np.isnan(predictions_all).any():
        raise RuntimeError("Не все тестовые точки получили прогноз.")

    predicted_labels = decode_targets(predictions_all)

    errors = np.array([
        circular_distance(a, b)
        for a, b in zip(labels, predicted_labels)
    ])

    accuracy = float(
        np.mean(predicted_labels == labels)
    )

    mae = float(np.mean(errors))

    return {
        "accuracy": accuracy,
        "mae": mae,
        "predictions": predicted_labels,
        "errors": errors,
        "fold_records": fold_records
    }


# ============================================================
# 10. ПОЛНЫЙ НАБОР МОДЕЛЕЙ
# ============================================================

def evaluate_all(features, labels, folds):

    configurations = []

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

                configurations.append({
                    "model": feature_name,
                    "gamma": gamma,
                    "alpha": alpha,
                    "accuracy": result["accuracy"],
                    "mae": result["mae"],
                    "predictions": result["predictions"],
                    "errors": result["errors"],
                    "fold_records": result["fold_records"]
                })

    configurations.sort(
        key=lambda item: (
            item["mae"],
            -item["accuracy"]
        )
    )

    return configurations


# ============================================================
# 11. CSV-ВЫВОД
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
# 12. ГЛАВНЫЙ ЭКСПЕРИМЕНТ
# ============================================================

def main():

    start_time = time.time()

    print("=" * 78)
    print("ЗАПУСК ЭКСПЕРИМЕНТА C12")
    print("Алгебраическая сигнатура через полиномы деления")
    print("=" * 78)

    # --------------------------------------------------------
    # ШАГ 1
    # --------------------------------------------------------

    print("\n[1/9] Построение группы...")

    points = build_group()

    print(f"Кривая: y^2 = x^3 + 7 (mod {P})")
    print(f"Генератор: {G}")
    print(f"Порядок: {ORDER}")
    print(f"Проверено ненулевых точек: {ORDER - 1}")

    # --------------------------------------------------------
    # ШАГ 2
    # --------------------------------------------------------

    print("\n[2/9] Проверка полиномов деления...")

    test_point = points[1]

    test_psi = division_polynomials(
        test_point,
        max_n=max(MULTIPLIERS)
    )

    for m in [2, 3, 4, 5, 10, 15, 20]:

        print(
            f"psi_{m}({test_point}) = "
            f"{test_psi[m]}"
        )

    # --------------------------------------------------------
    # ШАГ 3
    # --------------------------------------------------------

    print("\n[3/9] Построение признаков...")

    features, feature_rows = build_features(points)

    labels = np.arange(1, ORDER)

    print(f"Общая выборка: {len(labels)}")

    for name, X in features.items():

        print(
            f"{name:25s} "
            f"признаков={X.shape[1]:3d} "
            f"образцов={X.shape[0]:3d}"
        )

    # --------------------------------------------------------
    # ШАГ 4
    # --------------------------------------------------------

    print("\n[4/9] Построение фиксированных CV-блоков...")

    folds = build_folds()

    fold_rows = []

    for fold_id, (train_idx, test_idx) in enumerate(
        folds,
        start=1
    ):

        fold_rows.append({
            "fold": fold_id,
            "train_count": len(train_idx),
            "test_count": len(test_idx),
            "train_indices": ";".join(
                str(int(x)) for x in train_idx
            ),
            "test_indices": ";".join(
                str(int(x)) for x in test_idx
            )
        })

    # --------------------------------------------------------
    # ШАГ 5
    # --------------------------------------------------------

    print("\n[5/9] BLOCKED CROSS-VALIDATION...")

    observed = evaluate_all(
        features,
        labels,
        folds
    )

    print("\nРЕЗУЛЬТАТЫ МОДЕЛЕЙ")
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
        item["mae"] for item in observed
    )

    observed_max_accuracy = max(
        item["accuracy"] for item in observed
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

    print(f"Количество перестановок: {PERMUTATIONS}")

    print(
        "Все модели используют одну и ту же "
        "переставленную последовательность меток."
    )

    print("CV-блоки и purge-зоны остаются неизменными.")

    permutation_rows = []

    null_best_mae = []
    null_max_accuracy = []

    for permutation_id in range(1, PERMUTATIONS + 1):

        permuted_labels = np.random.permutation(labels)

        permutation_results = evaluate_all(
            features,
            permuted_labels,
            folds
        )

        best_permuted_mae = min(
            item["mae"]
            for item in permutation_results
        )

        max_permuted_accuracy = max(
            item["accuracy"]
            for item in permutation_results
        )

        null_best_mae.append(best_permuted_mae)

        null_max_accuracy.append(max_permuted_accuracy)

        permutation_rows.append({
            "permutation": permutation_id,
            "best_mae": best_permuted_mae,
            "max_accuracy": max_permuted_accuracy
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
        1 + np.sum(null_best_mae <= observed_best_mae)
    ) / (PERMUTATIONS + 1)

    p_accuracy = (
        1 + np.sum(
            null_max_accuracy >= observed_max_accuracy
        )
    ) / (PERMUTATIONS + 1)

    print("=" * 78)
    print("СТАТИСТИЧЕСКИЕ РЕЗУЛЬТАТЫ C12")
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
        f"Permutation p-value MAE: {p_mae:.4f}"
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

    # В выборке 72 допустимых ненулевых скаляра.
    exact_baseline = 1.0 / (ORDER - 1)

    print(
        f"Случайный ориентир Accuracy: "
        f"1/72 = {exact_baseline:.4%}"
    )

    print(
        "Примечание: 1/73 применялось бы при наличии "
        "73 равновероятных классов, включая скаляр 0."
    )

    # --------------------------------------------------------
    # ШАГ 8
    # --------------------------------------------------------

    print("\n[8/9] СОХРАНЕНИЕ РЕЗУЛЬТАТОВ...")

    # Сводка моделей.
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
        OUT_DIR / "C12_model_summary.csv",
        summary_rows,
        [
            "model",
            "gamma",
            "alpha",
            "accuracy",
            "circular_mae"
        ]
    )

    # Прогнозы лучшей модели.
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
        OUT_DIR / "C12_predictions.csv",
        prediction_rows,
        [
            "true_scalar",
            "predicted_scalar",
            "circular_error"
        ]
    )

    # CV-блоки.
    write_csv(
        OUT_DIR / "C12_folds.csv",
        fold_rows,
        [
            "fold",
            "train_count",
            "test_count",
            "train_indices",
            "test_indices"
        ]
    )

    # Перестановки.
    write_csv(
        OUT_DIR / "C12_permutations.csv",
        permutation_rows,
        [
            "permutation",
            "best_mae",
            "max_accuracy"
        ]
    )

    # Значения исходных признаков.
    feature_fieldnames = list(feature_rows[0].keys())

    write_csv(
        OUT_DIR / "C12_features.csv",
        feature_rows,
        feature_fieldnames
    )

    # --------------------------------------------------------
    # ТЕКСТОВЫЙ ОТЧЁТ
    # --------------------------------------------------------

    elapsed = time.time() - start_time

    report = []

    report.append("ЭКСПЕРИМЕНТ C12")
    report.append("Алгебраическая сигнатура через полиномы деления")
    report.append("")
    report.append(f"p = {P}")
    report.append(f"Кривая: y^2 = x^3 + 7 mod {P}")
    report.append(f"G = {G}")
    report.append(f"Порядок G = {ORDER}")
    report.append(f"Ненулевых точек = {ORDER - 1}")
    report.append("")
    report.append("ПОЛИНОМЫ ДЕЛЕНИЯ")
    report.append(f"Диапазон m: {min(MULTIPLIERS)}..{max(MULTIPLIERS)}")
    report.append("")
    report.append("ЛУЧШАЯ МОДЕЛЬ")
    report.append(f"Модель: {best['model']}")
    report.append(f"Gamma: {best['gamma']}")
    report.append(f"Alpha: {best['alpha']}")
    report.append(f"MAE: {best['mae']:.6f}")
    report.append(f"Accuracy: {best['accuracy']:.6f}")
    report.append("")
    report.append("ПЕРЕСТАНОВОЧНЫЙ КОНТРОЛЬ")
    report.append(f"Перестановок: {PERMUTATIONS}")
    report.append(f"Observed best MAE: {observed_best_mae:.6f}")
    report.append(f"Null mean MAE: {np.mean(null_best_mae):.6f}")
    report.append(f"Null minimum MAE: {np.min(null_best_mae):.6f}")
    report.append(f"p-value MAE: {p_mae:.6f}")
    report.append("")
    report.append(f"Observed max accuracy: {observed_max_accuracy:.6f}")
    report.append(f"Null mean accuracy: {np.mean(null_max_accuracy):.6f}")
    report.append(f"p-value accuracy: {p_accuracy:.6f}")
    report.append("")
    report.append(f"Случайный точный baseline: 1/72 = {exact_baseline:.6%}")
    report.append("")
    report.append("ОБЩИЙ СПИСОК МОДЕЛЕЙ")

    for item in observed:

        report.append(
            f"{item['model']}, "
            f"gamma={item['gamma']}, "
            f"alpha={item['alpha']}, "
            f"accuracy={item['accuracy']:.6f}, "
            f"MAE={item['mae']:.6f}"
        )

    report.append("")
    report.append(f"Время выполнения: {elapsed:.2f} сек.")

    report_text = "\n".join(report)

    with open(
        OUT_DIR / "C12_report.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(report_text)

    # --------------------------------------------------------
    # ZIP-АРХИВ
    # --------------------------------------------------------

    print("\n[9/9] СОЗДАНИЕ ZIP-АРХИВА...")

    zip_path = BASE_DIR / "experimentC12.zip"

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

    print("\n" + "=" * 78)
    print("C12 ЗАВЕРШЁН")
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
    print(f"Permutation p-value Accuracy: {p_accuracy:.4f}")

    print()

    if p_mae < 0.05 or p_accuracy < 0.05:

        print(
            "ОБНАРУЖЕН СТАТИСТИЧЕСКИЙ СИГНАЛ. "
            "Требуются независимые проверки устойчивости."
        )

    else:

        print(
            "Значимого улучшения аппроксимации "
            "по перестановочному контролю не обнаружено."
        )

    print("\nВажно:")
    print(
        "Даже статистически значимый результат на игрушечной "
        "кривой не доказывает существование эффективного метода "
        "решения ECDLP на криптографических кривых."
    )


# ============================================================
# 13. ЗАПУСК
# ============================================================

if __name__ == "__main__":

    try:
        main()

    except Exception:

        print("\nОШИБКА ВЫПОЛНЕНИЯ C12")
        traceback.print_exc()

        input("\nНажми Enter для выхода...")
