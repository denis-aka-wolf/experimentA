#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EXPERIMENT B5

Цель:
Исследовать, можно ли восстановить скаляр k из комплексного
отпечатка T_t(kG) без использования полной таблицы соответствий.

Кривая:
    y^2 = x^3 + 7 (mod 499)

Базовая точка:
    G = (3, 212)

Порядок G:
    73

Частоты:
    t = 174
    t = 1

Эксперименты:

1. Циклическая геометрия:
       D(k, delta) = |T(kG) - T((k+delta)G)|

2. Противоположные точки:
       kG и (73-k)G

3. Минимальные расстояния и теоретическая устойчивость.

4. Слепое восстановление k:
       - линейная ridge-регрессия;
       - квадратичная ridge-регрессия;
       - ближайший сосед среди обучающих отпечатков.

5. Сравнение с обычным перебором.

Разделение:
    train: k % 5 != 0
    test:  k % 5 == 0

Тестовые точки не используются при обучении моделей.

Все результаты сохраняются в CSV, JSON, README и ZIP.
"""

import os
import csv
import json
import math
import zipfile
from datetime import datetime, timezone


# ============================================================
# 1. ПАРАМЕТРЫ
# ============================================================

P = 499
B = 7

G = (3, 212)

ORDER = 73

PRIMITIVE_ROOT = 7

FREQUENCIES = (174, 1)

OUT_DIR = "experimentB5_results"

# Параметры моделей
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0)


# ============================================================
# 2. АРИФМЕТИКА ЭЛЛИПТИЧЕСКОЙ КРИВОЙ
# ============================================================

def inv_mod(a):
    return pow(a % P, -1, P)


def on_curve(point):

    if point is None:
        return True

    x, y = point

    return (
        y * y - x * x * x - B
    ) % P == 0


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

        lam = (
            3 * x1 * x1 * inv_mod(2 * y1)
        ) % P

    else:

        lam = (
            (y2 - y1) * inv_mod(x2 - x1)
        ) % P

    x3 = (
        lam * lam - x1 - x2
    ) % P

    y3 = (
        lam * (x1 - x3) - y1
    ) % P

    return x3, y3


def scalar_mult(k, point):

    result = None
    addend = point

    k = int(k)

    while k > 0:

        if k & 1:
            result = point_add(result, addend)

        addend = point_add(addend, addend)

        k >>= 1

    return result


# ============================================================
# 3. ПЕРЕЧИСЛЕНИЕ КРИВОЙ
# ============================================================

def enumerate_curve():

    points = []

    for x in range(P):

        rhs = (x ** 3 + B) % P

        for y in range(P):

            if (y * y) % P == rhs:
                points.append((x, y))

    return points


def build_subgroup():

    points = [
        scalar_mult(k, G)
        for k in range(ORDER)
    ]

    if points[0] is not None:
        raise RuntimeError("0G должен быть O")

    if points[-1] is None:
        raise RuntimeError("72G не должен быть O")

    if len(set(points)) != ORDER:
        raise RuntimeError("Порядок G не равен 73")

    if scalar_mult(ORDER, G) is not None:
        raise RuntimeError("73G должен быть O")

    if scalar_mult(371, G) != (449, 483):
        raise RuntimeError("Ошибка контрольной точки 371G")

    return points


# ============================================================
# 4. КУБИЧНЫЙ ХАРАКТЕР
# ============================================================

def build_log_table():

    table = {}

    value = 1

    for exponent in range(P - 1):

        if value in table:
            raise RuntimeError(
                "Примитивный корень некорректен"
            )

        table[value] = exponent

        value = (
            value * PRIMITIVE_ROOT
        ) % P

    if len(table) != P - 1 or value != 1:
        raise RuntimeError(
            "Ошибка построения таблицы логарифмов"
        )

    return table


def build_chi3(log_table):

    omega = complex(
        math.cos(2 * math.pi / 3),
        math.sin(2 * math.pi / 3)
    )

    def chi3(a):

        a %= P

        if a == 0:
            return 0j

        exponent = log_table[a] % 3

        return omega ** exponent

    return chi3


# ============================================================
# 5. ВЫЧИСЛЕНИЕ ОТПЕЧАТКОВ
# ============================================================

def build_fingerprints(curve_points, subgroup, t, chi3):

    exp_table = []

    for v in range(P):

        angle = (
            2 * math.pi * (t * v % P) / P
        )

        exp_table.append(
            complex(
                math.cos(angle),
                math.sin(angle)
            )
        )

    x_values = sorted({

        subgroup[k][0]

        for k in range(1, ORDER)

    })

    A_by_x = {}

    for x0 in x_values:

        total = 0j

        for u, v in curve_points:

            total += (
                chi3(u - x0) * exp_table[v]
            )

        A_by_x[x0] = total

    rows = []

    for k in range(1, ORDER):

        x, y = subgroup[k]

        phase = exp_table[(-y) % P]

        z = A_by_x[x] * phase

        rows.append({

            "k": k,

            "x": x,
            "y": y,

            "re": z.real,
            "im": z.imag,

            "abs": abs(z),

            "arg": math.atan2(z.imag, z.real),

            "cos_arg": math.cos(
                math.atan2(z.imag, z.real)
            ),

            "sin_arg": math.sin(
                math.atan2(z.imag, z.real)
            ),

            "A_re": A_by_x[x].real,
            "A_im": A_by_x[x].imag,

            "A_abs": abs(A_by_x[x]),

            "fingerprint": z
        })

    return rows


# ============================================================
# 6. CSV
# ============================================================

def write_csv(path, rows, fields=None):

    if not rows:
        return

    if fields is None:
        fields = list(rows[0].keys())

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore"
        )

        writer.writeheader()

        writer.writerows(rows)


# ============================================================
# 7. ЦИКЛИЧЕСКАЯ ГЕОМЕТРИЯ
# ============================================================

def analyze_cyclic_geometry(rows, t):

    by_k = {
        row["k"]: row["fingerprint"]
        for row in rows
    }

    results = []

    for delta in range(1, ORDER):

        distances = []

        for k in range(1, ORDER):

            # Настоящее сложение скаляров в группе:
            # (k + delta) mod ORDER.
            target_k = (k + delta) % ORDER

            # target_k == 0 означает точку на бесконечности O.
            # Для неё комплексный отпечаток в этом эксперименте
            # не определён, поэтому эту пару пропускаем.
            if target_k == 0:
                continue

            distance = abs(
                by_k[k] - by_k[target_k]
            )

            distances.append(distance)

        if not distances:
            raise RuntimeError(
                f"Нет расстояний для delta={delta}, t={t}"
            )

        results.append({

            "t": t,

            "delta_k": delta,

            "pair_count": len(distances),

            "min_distance": min(distances),

            "mean_distance": (
                sum(distances) / len(distances)
            ),

            "max_distance": max(distances)

        })

    return results


# ============================================================
# 8. ПРОТИВОПОЛОЖНЫЕ ТОЧКИ
# ============================================================

def analyze_antipodal(rows, t):

    by_k = {

        row["k"]: row

        for row in rows

    }

    results = []

    for k in range(1, 37):

        opposite_k = ORDER - k

        first = by_k[k]
        second = by_k[opposite_k]

        distance = abs(
            first["fingerprint"] -
            second["fingerprint"]
        )

        results.append({

            "t": t,

            "k": k,

            "opposite_k": opposite_k,

            "x1": first["x"],
            "y1": first["y"],

            "x2": second["x"],
            "y2": second["y"],

            "distance": distance,

            "abs_T_1": first["abs"],

            "abs_T_2": second["abs"],

            "abs_difference": abs(
                first["abs"] - second["abs"]
            )
        })

    return results


# ============================================================
# 9. МИНИМАЛЬНЫЕ РАССТОЯНИЯ
# ============================================================

def analyze_pairwise_distances(rows, t):

    distances = []

    for i in range(len(rows)):

        for j in range(i + 1, len(rows)):

            distance = abs(
                rows[i]["fingerprint"] -
                rows[j]["fingerprint"]
            )

            distances.append({

                "t": t,

                "k1": rows[i]["k"],

                "k2": rows[j]["k"],

                "delta_k_mod_73": min(

                    (rows[j]["k"] - rows[i]["k"]) % ORDER,

                    (rows[i]["k"] - rows[j]["k"]) % ORDER

                ),

                "distance": distance
            })

    distances.sort(
        key=lambda item: item["distance"]
    )

    return distances


# ============================================================
# 10. ТЕОРЕТИЧЕСКАЯ УСТОЙЧИВОСТЬ
# ============================================================

def analyze_separation(rows, t):

    distances = [

        abs(
            rows[i]["fingerprint"] -
            rows[j]["fingerprint"]
        )

        for i in range(len(rows))

        for j in range(i + 1, len(rows))
    ]

    minimum = min(distances)

    return {

        "t": t,

        "minimum_separation": minimum,

        "guaranteed_noise_radius": minimum / 2,

        "explanation": (
            "Если абсолютная ошибка каждого комплексного "
            "отпечатка строго меньше половины минимального "
            "расстояния между отпечатками, ближайший "
            "отпечаток остаётся правильным."
        )
    }


# ============================================================
# 11. ВЕКТОРЫ ПРИЗНАКОВ
# ============================================================

def base_features(row):

    return [

        row["re"],

        row["im"],

        row["abs"],

        row["cos_arg"],

        row["sin_arg"]

    ]


def standardize(train_features, all_features):

    dimension = len(train_features[0])

    means = []
    scales = []

    for j in range(dimension):

        values = [
            row[j]
            for row in train_features
        ]

        mean = sum(values) / len(values)

        variance = sum(
            (value - mean) ** 2
            for value in values
        ) / len(values)

        scale = math.sqrt(variance)

        if scale < 1e-12:
            scale = 1.0

        means.append(mean)
        scales.append(scale)

    transformed = []

    for row in all_features:

        transformed.append([

            (row[j] - means[j]) / scales[j]

            for j in range(dimension)

        ])

    return transformed, means, scales


# ============================================================
# 12. ФОРМИРОВАНИЕ ПОЛИНОМИАЛЬНЫХ ПРИЗНАКОВ
# ============================================================

def make_design_matrix(features, degree):

    result = []

    for row in features:

        vector = [1.0]

        # Линейные признаки

        vector.extend(row)

        if degree >= 2:

            # Квадратичные и смешанные признаки

            for i in range(len(row)):

                for j in range(i, len(row)):

                    vector.append(
                        row[i] * row[j]
                    )

        result.append(vector)

    return result


# ============================================================
# 13. РЕШЕНИЕ ЛИНЕЙНОЙ СИСТЕМЫ
# ============================================================

def solve_linear_system(matrix, vector):

    n = len(vector)

    a = [

        list(matrix[i]) + [vector[i]]

        for i in range(n)

    ]

    for col in range(n):

        pivot = max(

            range(col, n),

            key=lambda r: abs(a[r][col])

        )

        if abs(a[pivot][col]) < 1e-14:

            raise RuntimeError(
                "Система вырождена"
            )

        a[col], a[pivot] = (
            a[pivot],
            a[col]
        )

        divisor = a[col][col]

        for j in range(col, n + 1):

            a[col][j] /= divisor

        for r in range(n):

            if r == col:
                continue

            factor = a[r][col]

            for j in range(col, n + 1):

                a[r][j] -= (
                    factor * a[col][j]
                )

    return [
        a[i][n]
        for i in range(n)
    ]


# ============================================================
# 14. RIDGE-РЕГРЕССИЯ
# ============================================================

def fit_ridge(X, Y, alpha):

    """
    X: матрица признаков.
    Y: пары (cos(k), sin(k)).
    """

    n = len(X)
    m = len(X[0])

    xtx = [

        [0.0] * m

        for _ in range(m)

    ]

    xty = [

        [0.0, 0.0]

        for _ in range(m)

    ]

    for r in range(n):

        for i in range(m):

            xty[i][0] += (
                X[r][i] * Y[r][0]
            )

            xty[i][1] += (
                X[r][i] * Y[r][1]
            )

            for j in range(m):

                xtx[i][j] += (
                    X[r][i] * X[r][j]
                )

    # Интерсепт не штрафуем.

    for i in range(1, m):

        xtx[i][i] += alpha

    beta_cos = solve_linear_system(

        xtx,

        [
            row[0]
            for row in xty
        ]

    )

    beta_sin = solve_linear_system(

        xtx,

        [
            row[1]
            for row in xty
        ]

    )

    return beta_cos, beta_sin


def predict_ridge(X, model):

    beta_cos, beta_sin = model

    predictions = []

    for row in X:

        pred_cos = sum(

            row[j] * beta_cos[j]

            for j in range(len(row))

        )

        pred_sin = sum(

            row[j] * beta_sin[j]

            for j in range(len(row))

        )

        predictions.append(
            (pred_cos, pred_sin)
        )

    return predictions


# ============================================================
# 15. ПРЕОБРАЗОВАНИЕ ПРОГНОЗА В СКАЛЯР
# ============================================================

def decode_scalar(prediction):

    cos_value, sin_value = prediction

    angle = math.atan2(
        sin_value,
        cos_value
    )

    if angle < 0:
        angle += 2 * math.pi

    k_float = (
        angle * ORDER / (2 * math.pi)
    )

    return k_float % ORDER


def circular_error(predicted, actual):

    difference = abs(
        predicted - actual
    ) % ORDER

    return min(
        difference,
        ORDER - difference
    )


def nearest_integer_scalar(value):

    k = int(round(value)) % ORDER

    return k


# ============================================================
# 16. БЛИЖАЙШИЙ СОСЕД
# ============================================================

def nearest_neighbor_predict(
    query_features,
    train_features,
    train_k
):

    best_distance = None
    best_k = None

    for features, k in zip(
        train_features,
        train_k
    ):

        distance = math.sqrt(

            sum(
                (a - b) ** 2

                for a, b in zip(
                    query_features,
                    features
                )
            )

        )

        if (
            best_distance is None
            or distance < best_distance
        ):

            best_distance = distance
            best_k = k

    return best_k, best_distance


# ============================================================
# 17. СЛЕПОЕ ВОССТАНОВЛЕНИЕ
# ============================================================

def run_blind_test(rows, t):

    """
    Тестовые скаляры:
        k % 5 == 0

    Обучающие:
        k % 5 != 0

    Модели не получают тестовые скаляры
    во время обучения.
    """

    train_rows = [
        row for row in rows
        if row["k"] % 5 != 0
    ]

    test_rows = [
        row for row in rows
        if row["k"] % 5 == 0
    ]

    train_k = [
        row["k"]
        for row in train_rows
    ]

    train_raw = [
        base_features(row)
        for row in train_rows
    ]

    test_raw = [
        base_features(row)
        for row in test_rows
    ]

    all_raw = train_raw + test_raw

    normalized, means, scales = standardize(
        train_raw,
        all_raw
    )

    train_features = normalized[
        :len(train_rows)
    ]

    test_features = normalized[
        len(train_rows):
    ]

    train_targets = [

        (

            math.cos(
                2 * math.pi * row["k"] / ORDER
            ),

            math.sin(
                2 * math.pi * row["k"] / ORDER
            )

        )

        for row in train_rows

    ]

    results = []
    predictions_csv = []

    # --------------------------------------------------------
    # Ближайший сосед среди обучающих точек
    # --------------------------------------------------------

    nn_correct = 0
    nn_errors = []

    for row, features in zip(
        test_rows,
        test_features
    ):

        predicted_k, distance = nearest_neighbor_predict(

            features,

            train_features,

            train_k

        )

        error = circular_error(
            predicted_k,
            row["k"]
        )

        if error == 0:
            nn_correct += 1

        nn_errors.append(error)

        predictions_csv.append({

            "t": t,

            "method": "nearest_neighbor",

            "degree": 0,

            "alpha": 0,

            "actual_k": row["k"],

            "predicted_k": predicted_k,

            "circular_error": error,

            "correct": error == 0,

            "feature_distance": distance
        })

    results.append({

        "t": t,

        "method": "nearest_neighbor",

        "degree": 0,

        "alpha": 0,

        "train_count": len(train_rows),

        "test_count": len(test_rows),

        "exact_count": nn_correct,

        "accuracy": nn_correct / len(test_rows),

        "mean_circular_error": (
            sum(nn_errors) / len(nn_errors)
        ),

        "max_circular_error": max(nn_errors)
    })

    # --------------------------------------------------------
    # Ridge-регрессия
    # --------------------------------------------------------

    for degree in (1, 2):

        X_train = make_design_matrix(
            train_features,
            degree
        )

        X_test = make_design_matrix(
            test_features,
            degree
        )

        for alpha in RIDGE_ALPHAS:

            model = fit_ridge(

                X_train,

                train_targets,

                alpha

            )

            predictions = predict_ridge(
                X_test,
                model
            )

            errors = []
            exact_count = 0

            for row, prediction in zip(
                test_rows,
                predictions
            ):

                predicted_float = decode_scalar(
                    prediction
                )

                predicted_k = nearest_integer_scalar(
                    predicted_float
                )

                error = circular_error(
                    predicted_k,
                    row["k"]
                )

                errors.append(error)

                if error == 0:
                    exact_count += 1

                predictions_csv.append({

                    "t": t,

                    "method": "ridge",

                    "degree": degree,

                    "alpha": alpha,

                    "actual_k": row["k"],

                    "predicted_k": predicted_k,

                    "predicted_k_float": predicted_float,

                    "predicted_cos": prediction[0],

                    "predicted_sin": prediction[1],

                    "circular_error": error,

                    "correct": error == 0
                })

            results.append({

                "t": t,

                "method": "ridge",

                "degree": degree,

                "alpha": alpha,

                "train_count": len(train_rows),

                "test_count": len(test_rows),

                "exact_count": exact_count,

                "accuracy": (
                    exact_count / len(test_rows)
                ),

                "mean_circular_error": (
                    sum(errors) / len(errors)
                ),

                "max_circular_error": max(errors)
            })

    return results, predictions_csv


# ============================================================
# 18. СРАВНЕНИЕ С ПЕРЕБОРОМ
# ============================================================

def build_baseline(t):

    return [

        {

            "t": t,

            "method": "brute_force",

            "known_table_entries": 0,

            "candidate_checks_max": 72,

            "description": (
                "Последовательно проверяем kG=Q."
            )
        },

        {

            "t": t,

            "method": "full_fingerprint_table",

            "known_table_entries": 72,

            "candidate_checks_max": 72,

            "description": (
                "Таблица всех 72 отпечатков. "
                "Требует предварительного вычисления."
            )
        },

        {

            "t": t,

            "method": "blind_model",

            "known_table_entries": 58,

            "candidate_checks_max": 0,

            "description": (
                "Обучение на 58 точках и проверка "
                "на 14 исключённых точках. "
                "Количество точек зависит от разбиения."
            )
        }

    ]


# ============================================================
# 19. ГЛАВНАЯ ПРОГРАММА
# ============================================================

def main():

    os.makedirs(
        OUT_DIR,
        exist_ok=True
    )

    print("EXPERIMENT B5")

    print("Перечисление кривой...")

    curve_points = enumerate_curve()

    if len(curve_points) != 510:

        raise RuntimeError(
            f"Ожидалось 510 точек, получено {len(curve_points)}"
        )

    subgroup = build_subgroup()

    print("Кривая и подгруппа проверены.")

    log_table = build_log_table()

    chi3 = build_chi3(log_table)

    all_geometry = []
    all_antipodal = []
    all_pairwise = []
    all_separation = []

    all_model_results = []
    all_predictions = []
    all_baseline = []

    summaries = {}

    # --------------------------------------------------------
    # Эксперименты для каждой частоты
    # --------------------------------------------------------

    for t in FREQUENCIES:

        print()
        print("=" * 55)

        print(
            f"Вычисление для t={t}"
        )

        print("=" * 55)

        rows = build_fingerprints(

            curve_points,

            subgroup,

            t,

            chi3

        )

        # Отпечатки

        fingerprint_rows = [

            {

                key: row[key]

                for key in (

                    "k",
                    "x",
                    "y",

                    "re",
                    "im",

                    "abs",

                    "arg",
                    "cos_arg",
                    "sin_arg",

                    "A_re",
                    "A_im",
                    "A_abs"

                )

            }

            for row in rows

        ]

        write_csv(

            os.path.join(
                OUT_DIR,
                f"fingerprints_t{t}.csv"
            ),

            fingerprint_rows

        )

        # Циклическая геометрия

        geometry = analyze_cyclic_geometry(
            rows,
            t
        )

        all_geometry.extend(geometry)

        # Противоположные точки

        antipodal = analyze_antipodal(
            rows,
            t
        )

        all_antipodal.extend(antipodal)

        # Все попарные расстояния

        pairwise = analyze_pairwise_distances(
            rows,
            t
        )

        all_pairwise.extend(pairwise)

        # Минимальное разделение

        separation = analyze_separation(
            rows,
            t
        )

        all_separation.append(separation)

        # Слепое восстановление

        print(
            f"Слепой тест для t={t}..."
        )

        model_results, predictions = run_blind_test(
            rows,
            t
        )

        all_model_results.extend(model_results)

        all_predictions.extend(predictions)

        # Baseline

        all_baseline.extend(
            build_baseline(t)
        )

        # Сводка

        summaries[str(t)] = {

            "fingerprint_count": len(rows),

            "unique_fingerprints": len({

                (
                    row["re"],
                    row["im"]
                )

                for row in rows

            }),

            "minimum_separation": (
                separation["minimum_separation"]
            ),

            "guaranteed_noise_radius": (
                separation["guaranteed_noise_radius"]
            ),

            "minimum_cyclic_delta_distance": min(

                row["min_distance"]

                for row in geometry

            ),

            "blind_test": model_results
        }

    # ========================================================
    # 20. СОХРАНЕНИЕ CSV
    # ========================================================

    write_csv(

        os.path.join(
            OUT_DIR,
            "cyclic_geometry.csv"
        ),

        all_geometry

    )

    write_csv(

        os.path.join(
            OUT_DIR,
            "antipodal_pairs.csv"
        ),

        all_antipodal

    )

    write_csv(

        os.path.join(
            OUT_DIR,
            "pairwise_distances.csv"
        ),

        all_pairwise

    )

    write_csv(

        os.path.join(
            OUT_DIR,
            "separation.csv"
        ),

        all_separation

    )

    write_csv(

        os.path.join(
            OUT_DIR,
            "blind_model_results.csv"
        ),

        all_model_results

    )

    write_csv(

        os.path.join(
            OUT_DIR,
            "blind_predictions.csv"
        ),

        all_predictions

    )

    write_csv(

        os.path.join(
            OUT_DIR,
            "baseline_comparison.csv"
        ),

        all_baseline

    )

    # ========================================================
    # 21. JSON
    # ========================================================

    report = {

        "experiment": "B5",

        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "curve": "y^2 = x^3 + 7 mod 499",

        "p": P,

        "affine_curve_points": len(curve_points),

        "base_point": list(G),

        "order": ORDER,

        "control_371G": list(
            scalar_mult(371, G)
        ),

        "frequencies": list(FREQUENCIES),

        "train_rule": "k % 5 != 0",

        "test_rule": "k % 5 == 0",

        "train_count": 58,

        "test_count": 14,

        "feature_names": [

            "Re(T)",

            "Im(T)",

            "abs(T)",

            "cos(arg(T))",

            "sin(arg(T))"

        ],

        "ridge_alphas": list(RIDGE_ALPHAS),

        "results": summaries,

        "limitations": [

            (
                "Результаты относятся только к игрушечной "
                "кривой и данной подгруппе."
            ),

            (
                "Тестовая выборка содержит только 14 точек."
            ),

            (
                "Модели обучаются на конечном наборе "
                "скалярных значений."
            ),

            (
                "Наличие успешной регрессии не доказывает "
                "существование эффективного алгоритма ECDLP."
            ),

            (
                "Модели используют вещественные и комплексные "
                "числа с плавающей точкой."
            ),

            (
                "Ближайший сосед среди обучающих точек "
                "не эквивалентен полному табличному поиску."
            )

        ]

    }

    with open(

        os.path.join(
            OUT_DIR,
            "report.json"
        ),

        "w",

        encoding="utf-8"

    ) as f:

        json.dump(

            report,

            f,

            ensure_ascii=False,

            indent=2

        )

    # ========================================================
    # 22. README
    # ========================================================

    readme = """
EXPERIMENT B5
=============

Кривая:
    y^2 = x^3 + 7 mod 499

Базовая точка:
    G = (3, 212)

Порядок:
    73

Частоты:
    t = 174
    t = 1


ФАЙЛЫ
-----

fingerprints_t174.csv
fingerprints_t1.csv

    Комплексные отпечатки конечных точек.

cyclic_geometry.csv

    Циклические расстояния между k и k+delta mod 73.

antipodal_pairs.csv

    Расстояния между противоположными точками k и 73-k.

pairwise_distances.csv

    Все попарные расстояния между отпечатками.

separation.csv

    Минимальное расстояние и гарантированный радиус
    распознавания при ограниченной абсолютной ошибке.

blind_model_results.csv

    Итоги слепого тестирования моделей.

blind_predictions.csv

    Предсказания моделей для исключённых точек.

baseline_comparison.csv

    Сравнение с перебором и табличным поиском.

report.json

    Полный отчёт.


СЛЕПОЕ РАЗБИЕНИЕ
----------------

Обучение:
    k % 5 != 0

Тест:
    k % 5 == 0

Тестовые скаляры не участвуют в обучении.

Признаки:
    Re(T)
    Im(T)
    abs(T)
    cos(arg(T))
    sin(arg(T))

Модели:
    Линейная ridge-регрессия.
    Квадратичная ridge-регрессия.
    Ближайший сосед среди обучающих точек.


ВАЖНО
-----

Успешное предсказание скаляра на 14 исключённых
точках не доказывает возможность решения ECDLP
на больших группах.

Для этого необходимо показать масштабируемый
алгоритм, который не требует предварительного
обучения на скалярах целевой подгруппы.
"""

    with open(

        os.path.join(
            OUT_DIR,
            "README_results.txt"
        ),

        "w",

        encoding="utf-8"

    ) as f:

        f.write(readme)

    # ========================================================
    # 23. ZIP
    # ========================================================

    zip_path = os.path.join(
        OUT_DIR,
        "experimentB5.zip"
    )

    with zipfile.ZipFile(

        zip_path,

        "w",

        zipfile.ZIP_DEFLATED

    ) as archive:

        for filename in sorted(
            os.listdir(OUT_DIR)
        ):

            if filename == os.path.basename(zip_path):
                continue

            full_path = os.path.join(
                OUT_DIR,
                filename
            )

            if os.path.isfile(full_path):

                archive.write(
                    full_path,
                    arcname=filename
                )

    # ========================================================
    # 24. ИТОГ
    # ========================================================

    print()
    print("=" * 60)

    print("EXPERIMENT B5 RESULTS")

    print("=" * 60)

    print(
        "Affine curve points:",
        len(curve_points)
    )

    print(
        "Order of G:",
        ORDER
    )

    for t in FREQUENCIES:

        summary = summaries[str(t)]

        print()
        print(f"FREQUENCY t={t}")

        print(
            "Unique fingerprints:",
            summary["unique_fingerprints"],
            "/ 72"
        )

        print(
            "Minimum separation:",
            summary["minimum_separation"]
        )

        print(
            "Guaranteed noise radius:",
            summary["guaranteed_noise_radius"]
        )

        print()
        print("BLIND TEST:")

        for model in summary["blind_test"]:

            print(

                model["method"],

                "degree=",
                model["degree"],

                "alpha=",
                model["alpha"],

                "accuracy=",
                round(model["accuracy"], 4),

                "mean error=",
                round(model["mean_circular_error"], 4)

            )

    print()
    print("=" * 60)

    print(
        "Results directory:",
        OUT_DIR
    )

    print(
        "Archive created:",
        zip_path
    )

    print("=" * 60)


if __name__ == "__main__":
    main()