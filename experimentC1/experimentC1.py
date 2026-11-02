
# -*- coding: utf-8 -*-

"""
ЭКСПЕРИМЕНТ C1
Геометрическое предсказание скаляра эллиптической кривой.

Кривая:
    y^2 = x^3 + 7 (mod 499)

Базовая точка:
    G = (3, 212)

Порядок базовой точки:
    73

Цель:
    Проверить, можно ли предсказывать скаляр k по геометрическим
    характеристикам точки P = kG без полного перебора подгруппы.

Проверяемые признаки:
    1. x, y
    2. Целочисленный коэффициент n:
       n = (x^3 - y^2 + 7) / p
    3. x, y, n
    4. Полиномиальные признаки x, y до второй степени и n

Методы:
    1. Циклическая ridge-регрессия.
    2. Циклический kNN.
    3. Геометрический baseline.
    4. Baseline по среднему скаляру.
    5. Контроль со случайной перестановкой меток.
    6. Leave-one-out.

Метрики:
    - Точность точного предсказания скаляра.
    - Средняя циклическая ошибка.
    - Максимальная циклическая ошибка.

Дополнительно:
    Проверяется устойчивость коэффициента n при замене:
        x -> x + m*p
        y -> y + r*p

Все результаты сохраняются в папку experimentC1_results.
В конце создаётся ZIP-архив.

Используются только стандартные библиотеки Python.
"""

import os
import csv
import json
import math
import random
import zipfile
import statistics
from datetime import datetime


# ============================================================
# 1. ПАРАМЕТРЫ ЭКСПЕРИМЕНТА
# ============================================================

P = 499
A = 0
B = 7

G = (3, 212)

ORDER = 73

RANDOM_SEED = 20261022

NUMBER_OF_SPLITS = 100
TRAIN_RATIO = 0.75

RIDGE_LAMBDAS = [0.1, 1.0, 10.0]

KNN_VALUES = [1, 3, 5]

LIFT_RANGE = range(-2, 3)

OUTPUT_DIR = "experimentC1_results"

random.seed(RANDOM_SEED)


# ============================================================
# 2. АРИФМЕТИКА ЭЛЛИПТИЧЕСКОЙ КРИВОЙ
# ============================================================

def mod_inverse(value, modulus):
    """Обратный элемент по модулю."""

    value %= modulus

    if value == 0:
        raise ZeroDivisionError("Обратного элемента не существует.")

    return pow(value, -1, modulus)


def point_add(P1, P2):
    """
    Сложение точек эллиптической кривой.

    Бесконечно удалённая точка обозначается None.
    """

    if P1 is None:
        return P2

    if P2 is None:
        return P1

    x1, y1 = P1
    x2, y2 = P2

    # Противоположные точки.
    if x1 == x2 and (y1 + y2) % P == 0:
        return None

    # Удвоение точки.
    if P1 == P2:

        if y1 % P == 0:
            return None

        numerator = (3 * x1 * x1 + A) % P
        denominator = (2 * y1) % P

    else:

        numerator = (y2 - y1) % P
        denominator = (x2 - x1) % P

    slope = (
        numerator * mod_inverse(denominator, P)
    ) % P

    x3 = (slope * slope - x1 - x2) % P

    y3 = (
        slope * (x1 - x3) - y1
    ) % P

    return (x3, y3)


def scalar_multiply(k, point):
    """Вычисление k*P методом double-and-add."""

    result = None
    current = point

    while k > 0:

        if k & 1:
            result = point_add(result, current)

        current = point_add(current, current)

        k >>= 1

    return result


# ============================================================
# 3. ПРОВЕРКА КРИВОЙ И ПОСТРОЕНИЕ ПОДГРУППЫ
# ============================================================

def verify_curve():

    x, y = G

    left = (y * y) % P
    right = (x * x * x + A * x + B) % P

    if left != right:
        raise ValueError("Базовая точка не принадлежит кривой.")

    if scalar_multiply(ORDER, G) is not None:
        raise ValueError("Заявленный порядок точки неверен.")

    for k in range(1, ORDER):

        if scalar_multiply(k, G) is None:
            raise ValueError(
                "Порядок базовой точки меньше заявленного."
            )

    print("Проверка кривой: успешно.")
    print("Базовая точка:", G)
    print("Порядок базовой точки:", ORDER)


def build_dataset():

    dataset = []

    for k in range(1, ORDER):

        point = scalar_multiply(k, G)

        if point is None:
            raise ValueError("Неожиданная бесконечно удалённая точка.")

        x, y = point

        # Проверяем принадлежность кривой.
        if (y * y - x * x * x - B) % P != 0:
            raise ValueError("Ошибка построения точки.")

        # Целочисленный коэффициент.
        numerator = x ** 3 - y ** 2 + B

        if numerator % P != 0:
            raise ValueError("Коэффициент n не является целым.")

        n = numerator // P

        dataset.append({
            "k": k,
            "x": x,
            "y": y,
            "n": n,
            "n_mod_p": n % P
        })

    if len(dataset) != ORDER - 1:
        raise ValueError("Неверный размер подгруппы.")

    if len(set((row["x"], row["y"]) for row in dataset)) != len(dataset):
        raise ValueError("В подгруппе обнаружены повторяющиеся точки.")

    return dataset


# ============================================================
# 4. ФОРМИРОВАНИЕ ГЕОМЕТРИЧЕСКИХ ПРИЗНАКОВ
# ============================================================

def get_features(row, feature_type):

    x = row["x"]
    y = row["y"]
    n = row["n"]
    n_mod = row["n_mod_p"]

    if feature_type == "XY":

        return [
            float(x),
            float(y)
        ]

    if feature_type == "N":

        return [
            float(n)
        ]

    if feature_type == "XYN":

        return [
            float(x),
            float(y),
            float(n)
        ]

    if feature_type == "XYN_MOD":

        return [
            float(x),
            float(y),
            float(n),
            float(n_mod)
        ]

    if feature_type == "POLY2":

        return [
            float(x),
            float(y),
            float(x * x),
            float(x * y),
            float(y * y),
            float(n)
        ]

    raise ValueError("Неизвестный тип признаков.")


FEATURE_TYPES = [
    "XY",
    "N",
    "XYN",
    "XYN_MOD",
    "POLY2"
]


# ============================================================
# 5. ЦИКЛИЧЕСКАЯ МЕТРИКА СКАЛЯРОВ
# ============================================================

def circular_distance(k1, k2):

    difference = abs(k1 - k2)

    return min(
        difference,
        ORDER - difference
    )


def scalar_to_phase(k):

    angle = 2.0 * math.pi * k / ORDER

    return (
        math.cos(angle),
        math.sin(angle)
    )


def phase_to_scalar(cos_value, sin_value):

    angle = math.atan2(sin_value, cos_value)

    if angle < 0:
        angle += 2.0 * math.pi

    best_k = None
    best_distance = float("inf")

    for k in range(1, ORDER):

        target_angle = 2.0 * math.pi * k / ORDER

        difference = abs(angle - target_angle)

        difference = min(
            difference,
            2.0 * math.pi - difference
        )

        if difference < best_distance:

            best_distance = difference
            best_k = k

    return best_k


# ============================================================
# 6. СТАНДАРТИЗАЦИЯ ПРИЗНАКОВ
# ============================================================

def fit_scaler(X):

    columns = len(X[0])

    means = []
    deviations = []

    for j in range(columns):

        values = [row[j] for row in X]

        mean = sum(values) / len(values)

        variance = sum(
            (value - mean) ** 2
            for value in values
        ) / len(values)

        deviation = math.sqrt(variance)

        if deviation == 0:
            deviation = 1.0

        means.append(mean)
        deviations.append(deviation)

    return means, deviations


def apply_scaler(X, scaler):

    means, deviations = scaler

    result = []

    for row in X:

        result.append([
            (row[j] - means[j]) / deviations[j]
            for j in range(len(row))
        ])

    return result


# ============================================================
# 7. РЕШЕНИЕ СИСТЕМЫ ЛИНЕЙНЫХ УРАВНЕНИЙ
# ============================================================

def solve_linear_system(matrix, vector):
    """
    Метод Гаусса с частичным выбором главного элемента.
    """

    n = len(vector)

    a = [
        matrix[i][:] + [vector[i]]
        for i in range(n)
    ]

    for col in range(n):

        pivot = max(
            range(col, n),
            key=lambda row: abs(a[row][col])
        )

        if abs(a[pivot][col]) < 1e-12:
            raise ValueError("Вырожденная матрица.")

        a[col], a[pivot] = a[pivot], a[col]

        divisor = a[col][col]

        for j in range(col, n + 1):
            a[col][j] /= divisor

        for i in range(n):

            if i == col:
                continue

            factor = a[i][col]

            for j in range(col, n + 1):
                a[i][j] -= factor * a[col][j]

    return [
        a[i][n]
        for i in range(n)
    ]


# ============================================================
# 8. RIDGE-РЕГРЕССИЯ ДЛЯ ЦИКЛИЧЕСКОГО СКАЛЯРА
# ============================================================

def ridge_fit(X, targets, regularization):

    feature_count = len(X[0])

    # Первый коэффициент соответствует свободному члену.
    dimension = feature_count + 1

    matrix = [
        [0.0] * dimension
        for _ in range(dimension)
    ]

    vector = [
        [0.0] * dimension
        for _ in range(len(targets[0]))
    ]

    for row, target in zip(X, targets):

        values = [1.0] + row

        for i in range(dimension):

            for j in range(dimension):
                matrix[i][j] += values[i] * values[j]

            for t in range(len(target)):
                vector[t][i] += values[i] * target[t]

    # Не штрафуем свободный член.
    for i in range(1, dimension):
        matrix[i][i] += regularization

    coefficients = []

    for t in range(len(vector)):

        coefficients.append(
            solve_linear_system(
                matrix,
                vector[t]
            )
        )

    return coefficients


def ridge_predict(X, coefficients):

    predictions = []

    for row in X:

        values = [1.0] + row

        prediction = []

        for model in coefficients:

            result = sum(
                values[i] * model[i]
                for i in range(len(values))
            )

            prediction.append(result)

        predictions.append(prediction)

    return predictions


def make_targets(scalars):

    result = []

    for k in scalars:

        cosine, sine = scalar_to_phase(k)

        result.append([
            cosine,
            sine
        ])

    return result


# ============================================================
# 9. ЦИКЛИЧЕСКИЙ KNN
# ============================================================

def euclidean_distance(a, b):

    return math.sqrt(
        sum(
            (x - y) ** 2
            for x, y in zip(a, b)
        )
    )


def knn_predict(
    X_train,
    y_train,
    X_test,
    neighbors_count
):

    predictions = []

    for test_row in X_test:

        distances = []

        for index, train_row in enumerate(X_train):

            distance = euclidean_distance(
                test_row,
                train_row
            )

            distances.append((
                distance,
                index
            ))

        distances.sort(key=lambda item: item[0])

        selected = distances[:neighbors_count]

        cosine_sum = 0.0
        sine_sum = 0.0

        for distance, index in selected:

            cosine, sine = scalar_to_phase(
                y_train[index]
            )

            cosine_sum += cosine
            sine_sum += sine

        if abs(cosine_sum) < 1e-12 and abs(sine_sum) < 1e-12:

            prediction = y_train[selected[0][1]]

        else:

            prediction = phase_to_scalar(
                cosine_sum,
                sine_sum
            )

        predictions.append(prediction)

    return predictions


# ============================================================
# 10. МЕТРИКИ
# ============================================================

def calculate_metrics(actual, predicted):

    errors = [
        circular_distance(a, p)
        for a, p in zip(actual, predicted)
    ]

    return {
        "accuracy": sum(
            1 for error in errors if error == 0
        ) / len(errors),

        "circular_mae": sum(errors) / len(errors),

        "circular_max_error": max(errors)
    }


def mean_metrics(metrics_list):

    return {
        key: statistics.mean(
            item[key]
            for item in metrics_list
        )
        for key in metrics_list[0]
    }


# ============================================================
# 11. BASELINE: СРЕДНИЙ СКАЛЯР
# ============================================================

def circular_mean_prediction(scalars):

    targets = make_targets(scalars)

    cosine_sum = sum(
        row[0]
        for row in targets
    )

    sine_sum = sum(
        row[1]
        for row in targets
    )

    if abs(cosine_sum) < 1e-12 and abs(sine_sum) < 1e-12:
        return scalars[0]

    return phase_to_scalar(
        cosine_sum,
        sine_sum
    )


# ============================================================
# 12. СЛУЧАЙНОЕ РАЗБИЕНИЕ
# ============================================================

def make_split(dataset, rng):

    indices = list(range(len(dataset)))

    rng.shuffle(indices)

    train_size = int(
        len(indices) * TRAIN_RATIO
    )

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    return train_indices, test_indices


# ============================================================
# 13. ОСНОВНЫЕ ПОВТОРЯЕМЫЕ ЭКСПЕРИМЕНТЫ
# ============================================================

def run_repeated_experiments(dataset):

    all_results = []

    rng = random.Random(RANDOM_SEED)

    for split_number in range(NUMBER_OF_SPLITS):

        train_indices, test_indices = make_split(
            dataset,
            rng
        )

        train_rows = [
            dataset[i]
            for i in train_indices
        ]

        test_rows = [
            dataset[i]
            for i in test_indices
        ]

        y_train = [
            row["k"]
            for row in train_rows
        ]

        y_test = [
            row["k"]
            for row in test_rows
        ]

        for feature_type in FEATURE_TYPES:

            X_train_raw = [
                get_features(row, feature_type)
                for row in train_rows
            ]

            X_test_raw = [
                get_features(row, feature_type)
                for row in test_rows
            ]

            scaler = fit_scaler(X_train_raw)

            X_train = apply_scaler(
                X_train_raw,
                scaler
            )

            X_test = apply_scaler(
                X_test_raw,
                scaler
            )

            # ------------------------------------------------
            # RIDGE
            # ------------------------------------------------

            targets = make_targets(y_train)

            for regularization in RIDGE_LAMBDAS:

                coefficients = ridge_fit(
                    X_train,
                    targets,
                    regularization
                )

                raw_predictions = ridge_predict(
                    X_test,
                    coefficients
                )

                predictions = [
                    phase_to_scalar(row[0], row[1])
                    for row in raw_predictions
                ]

                metrics = calculate_metrics(
                    y_test,
                    predictions
                )

                all_results.append({
                    "split": split_number,
                    "feature_type": feature_type,
                    "method": "RIDGE",
                    "parameter": regularization,
                    **metrics
                })

            # ------------------------------------------------
            # KNN
            # ------------------------------------------------

            for k_neighbors in KNN_VALUES:

                predictions = knn_predict(
                    X_train,
                    y_train,
                    X_test,
                    k_neighbors
                )

                metrics = calculate_metrics(
                    y_test,
                    predictions
                )

                all_results.append({
                    "split": split_number,
                    "feature_type": feature_type,
                    "method": "KNN",
                    "parameter": k_neighbors,
                    **metrics
                })

            # ------------------------------------------------
            # BASELINE: СРЕДНИЙ СКАЛЯР
            # ------------------------------------------------

            baseline_prediction = circular_mean_prediction(
                y_train
            )

            predictions = [
                baseline_prediction
                for _ in y_test
            ]

            metrics = calculate_metrics(
                y_test,
                predictions
            )

            all_results.append({
                "split": split_number,
                "feature_type": feature_type,
                "method": "MEAN_BASELINE",
                "parameter": 0,
                **metrics
            })

            # ------------------------------------------------
            # BASELINE: БЛИЖАЙШАЯ ТОЧКА В ГЕОМЕТРИИ XY
            # ------------------------------------------------

            xy_train = [
                [float(row["x"]), float(row["y"])]
                for row in train_rows
            ]

            xy_test = [
                [float(row["x"]), float(row["y"])]
                for row in test_rows
            ]

            xy_scaler = fit_scaler(xy_train)

            xy_train_scaled = apply_scaler(
                xy_train,
                xy_scaler
            )

            xy_test_scaled = apply_scaler(
                xy_test,
                xy_scaler
            )

            predictions = knn_predict(
                xy_train_scaled,
                y_train,
                xy_test_scaled,
                1
            )

            metrics = calculate_metrics(
                y_test,
                predictions
            )

            all_results.append({
                "split": split_number,
                "feature_type": feature_type,
                "method": "XY_NEAREST",
                "parameter": 1,
                **metrics
            })

            # ------------------------------------------------
            # КОНТРОЛЬ: СЛУЧАЙНАЯ ПЕРЕСТАНОВКА МЕТОК
            # ------------------------------------------------

            shuffled_labels = y_train[:]

            rng.shuffle(shuffled_labels)

            for regularization in RIDGE_LAMBDAS:

                coefficients = ridge_fit(
                    X_train,
                    make_targets(shuffled_labels),
                    regularization
                )

                raw_predictions = ridge_predict(
                    X_test,
                    coefficients
                )

                predictions = [
                    phase_to_scalar(row[0], row[1])
                    for row in raw_predictions
                ]

                metrics = calculate_metrics(
                    y_test,
                    predictions
                )

                all_results.append({
                    "split": split_number,
                    "feature_type": feature_type,
                    "method": "RANDOM_LABEL_RIDGE",
                    "parameter": regularization,
                    **metrics
                })

        if (split_number + 1) % 10 == 0:

            print(
                "Выполнено разбиений:",
                split_number + 1,
                "/",
                NUMBER_OF_SPLITS
            )

    return all_results


# ============================================================
# 14. LEAVE-ONE-OUT
# ============================================================

def run_leave_one_out(dataset):

    results = []

    for feature_type in FEATURE_TYPES:

        actual = []
        predicted = []

        for test_index in range(len(dataset)):

            train_rows = [
                row
                for i, row in enumerate(dataset)
                if i != test_index
            ]

            test_row = dataset[test_index]

            X_train_raw = [
                get_features(row, feature_type)
                for row in train_rows
            ]

            X_test_raw = [
                get_features(test_row, feature_type)
            ]

            scaler = fit_scaler(X_train_raw)

            X_train = apply_scaler(
                X_train_raw,
                scaler
            )

            X_test = apply_scaler(
                X_test_raw,
                scaler
            )

            y_train = [
                row["k"]
                for row in train_rows
            ]

            y_test = test_row["k"]

            coefficients = ridge_fit(
                X_train,
                make_targets(y_train),
                1.0
            )

            raw_prediction = ridge_predict(
                X_test,
                coefficients
            )[0]

            prediction = phase_to_scalar(
                raw_prediction[0],
                raw_prediction[1]
            )

            actual.append(y_test)
            predicted.append(prediction)

        metrics = calculate_metrics(
            actual,
            predicted
        )

        results.append({
            "feature_type": feature_type,
            "method": "RIDGE_LOO",
            **metrics
        })

        # KNN leave-one-out.

        actual_knn = []
        predicted_knn = []

        for test_index in range(len(dataset)):

            train_rows = [
                row
                for i, row in enumerate(dataset)
                if i != test_index
            ]

            test_row = dataset[test_index]

            X_train_raw = [
                get_features(row, feature_type)
                for row in train_rows
            ]

            X_test_raw = [
                get_features(test_row, feature_type)
            ]

            scaler = fit_scaler(X_train_raw)

            X_train = apply_scaler(
                X_train_raw,
                scaler
            )

            X_test = apply_scaler(
                X_test_raw,
                scaler
            )

            y_train = [
                row["k"]
                for row in train_rows
            ]

            prediction = knn_predict(
                X_train,
                y_train,
                X_test,
                1
            )[0]

            actual_knn.append(test_row["k"])
            predicted_knn.append(prediction)

        metrics = calculate_metrics(
            actual_knn,
            predicted_knn
        )

        results.append({
            "feature_type": feature_type,
            "method": "KNN_LOO",
            **metrics
        })

    return results


# ============================================================
# 15. УСТОЙЧИВОСТЬ ЦЕЛОЧИСЛЕННОГО КОЭФФИЦИЕНТА n
# ============================================================

def calculate_lift_coefficient(x, y, m, r):

    lifted_x = x + m * P
    lifted_y = y + r * P

    numerator = (
        lifted_x ** 3
        - lifted_y ** 2
        + B
    )

    if numerator % P != 0:
        raise ValueError("Нарушена целочисленность коэффициента.")

    return numerator // P


def run_lift_experiment(dataset):

    results = []

    for row in dataset:

        values = []

        for m in LIFT_RANGE:

            for r in LIFT_RANGE:

                n_lifted = calculate_lift_coefficient(
                    row["x"],
                    row["y"],
                    m,
                    r
                )

                values.append(n_lifted)

        results.append({
            "k": row["k"],
            "x": row["x"],
            "y": row["y"],
            "canonical_n": row["n"],
            "canonical_n_mod_p": row["n_mod_p"],
            "lift_min": min(values),
            "lift_max": max(values),
            "lift_distinct_values": len(set(values)),
            "lift_variation": max(values) - min(values)
        })

    return results


# ============================================================
# 16. АГРЕГАЦИЯ РЕЗУЛЬТАТОВ
# ============================================================

def aggregate_results(results):

    groups = {}

    for row in results:

        key = (
            row["feature_type"],
            row["method"],
            row["parameter"]
        )

        groups.setdefault(key, []).append(row)

    aggregated = []

    for key, values in groups.items():

        feature_type, method, parameter = key

        aggregated.append({
            "feature_type": feature_type,
            "method": method,
            "parameter": parameter,

            "accuracy_mean": statistics.mean(
                row["accuracy"]
                for row in values
            ),

            "circular_mae_mean": statistics.mean(
                row["circular_mae"]
                for row in values
            ),

            "circular_max_error_mean": statistics.mean(
                row["circular_max_error"]
                for row in values
            ),

            "number_of_runs": len(values)
        })

    aggregated.sort(
        key=lambda row: (
            -row["accuracy_mean"],
            row["circular_mae_mean"]
        )
    )

    return aggregated


# ============================================================
# 17. СОХРАНЕНИЕ CSV
# ============================================================

def save_csv(path, rows):

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 18. СОХРАНЕНИЕ ОТЧЁТА
# ============================================================

def save_report(
    dataset,
    aggregate,
    loo_results,
    lift_results
):

    best_result = aggregate[0]

    report = {
        "experiment": "C1",
        "description": (
            "Геометрическое предсказание скаляра "
            "эллиптической кривой"
        ),

        "curve": {
            "p": P,
            "a": A,
            "b": B,
            "G": list(G),
            "order": ORDER
        },

        "dataset_size": len(dataset),

        "experiment_parameters": {
            "number_of_splits": NUMBER_OF_SPLITS,
            "train_ratio": TRAIN_RATIO,
            "ridge_lambdas": RIDGE_LAMBDAS,
            "knn_values": KNN_VALUES,
            "random_seed": RANDOM_SEED
        },

        "best_repeated_experiment": best_result,

        "repeated_experiments": aggregate,

        "leave_one_out": loo_results,

        "lift_experiment": {
            "number_of_points": len(lift_results),
            "average_distinct_lift_values": statistics.mean(
                row["lift_distinct_values"]
                for row in lift_results
            ),
            "average_lift_variation": statistics.mean(
                row["lift_variation"]
                for row in lift_results
            )
        },

        "important_limitations": [
            "Эксперимент проводится на малой игрушечной подгруппе.",
            "Успешное распознавание точки не означает решение ECDLP.",
            "Целочисленный коэффициент зависит от выбора представителей координат.",
            "Разбиение train/test не является доказательством криптографической стойкости.",
            "Полная таблица соответствий используется только для формирования меток и проверки."
        ]
    }

    path = os.path.join(
        OUTPUT_DIR,
        "experimentC1_report.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=4
        )


# ============================================================
# 19. README
# ============================================================

def save_readme():

    content = f"""
ЭКСПЕРИМЕНТ C1
==============================

Цель
----

Проверка возможности предсказывать скаляр k по геометрическим
характеристикам точки эллиптической кривой.

Кривая
------

y^2 = x^3 + 7 (mod 499)

Базовая точка:
G = (3, 212)

Порядок:
73

Размер исследуемой подгруппы:
72 ненулевые точки.

Основная гипотеза
-----------------

Геометрические признаки точки могут содержать информацию
о её скаляре k.

Особое внимание уделяется целочисленному коэффициенту:

n = (x^3 - y^2 + 7) / p

Методы
------

1. Циклическая ridge-регрессия.
2. Циклический kNN.
3. Ближайшая точка по координатам x,y.
4. Среднее значение скаляра.
5. Случайная перестановка меток.
6. Leave-one-out.

Метрики
-------

1. Точность точного предсказания.
2. Средняя циклическая ошибка.
3. Максимальная циклическая ошибка.

Циклическая ошибка:

d(k1,k2) = min(|k1-k2|, 73-|k1-k2|)

Дополнительный эксперимент
--------------------------

Проверяется зависимость n от выбора целочисленных представителей:

x -> x + m*p
y -> y + r*p

Диапазон m и r:
от -2 до 2.

Файлы
-----

dataset.csv
repeated_results.csv
aggregate_results.csv
leave_one_out.csv
lift_stability.csv
experimentC1_report.json
README.txt

Важное замечание
----------------

Эксперимент проверяет предсказуемость скаляра на игрушечной кривой.

Он не является доказательством ускорения решения ECDLP
и не должен интерпретироваться как криптографический алгоритм.
"""

    path = os.path.join(
        OUTPUT_DIR,
        "README.txt"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(content.strip())


# ============================================================
# 20. СОЗДАНИЕ ZIP-АРХИВА
# ============================================================

def create_archive():

    archive_path = OUTPUT_DIR + ".zip"

    with zipfile.ZipFile(
        archive_path,
        "w",
        zipfile.ZIP_DEFLATED
    ) as archive:

        for filename in os.listdir(OUTPUT_DIR):

            full_path = os.path.join(
                OUTPUT_DIR,
                filename
            )

            archive.write(
                full_path,
                arcname=os.path.join(
                    OUTPUT_DIR,
                    filename
                )
            )

    return archive_path


# ============================================================
# 21. ВЫВОД РЕЗУЛЬТАТОВ
# ============================================================

def print_results(aggregate, loo_results, lift_results):

    print()
    print("=" * 70)
    print("ИТОГИ ЭКСПЕРИМЕНТА C1")
    print("=" * 70)

    print()
    print("ЛУЧШИЕ РЕЗУЛЬТАТЫ ПОВТОРЯЕМЫХ ЭКСПЕРИМЕНТОВ")
    print("-" * 70)

    for row in aggregate[:15]:

        print(
            f'{row["feature_type"]:10s} '
            f'{row["method"]:20s} '
            f'параметр={row["parameter"]:6} '
            f'точность={row["accuracy_mean"]:.4f} '
            f'ошибка={row["circular_mae_mean"]:.4f}'
        )

    print()
    print("LEAVE-ONE-OUT")
    print("-" * 70)

    for row in loo_results:

        print(
            f'{row["feature_type"]:10s} '
            f'{row["method"]:12s} '
            f'точность={row["accuracy"]:.4f} '
            f'ошибка={row["circular_mae"]:.4f}'
        )

    print()
    print("УСТОЙЧИВОСТЬ КОЭФФИЦИЕНТА n")
    print("-" * 70)

    average_distinct = statistics.mean(
        row["lift_distinct_values"]
        for row in lift_results
    )

    average_variation = statistics.mean(
        row["lift_variation"]
        for row in lift_results
    )

    print(
        "Среднее число различных значений n при изменении lift:",
        round(average_distinct, 4)
    )

    print(
        "Средний диапазон изменения n:",
        round(average_variation, 4)
    )

    print()
    print("ВНИМАНИЕ:")
    print(
        "Высокая точность на игрушечной кривой "
        "сама по себе не означает ускорение ECDLP."
    )


# ============================================================
# 22. ГЛАВНАЯ ФУНКЦИЯ
# ============================================================

def main():

    print("=" * 70)
    print("ЭКСПЕРИМЕНТ C1")
    print("Геометрическое предсказание скаляра")
    print("=" * 70)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print()
    print("Этап 1. Проверка кривой.")

    verify_curve()

    print()
    print("Этап 2. Построение подгруппы.")

    dataset = build_dataset()

    print(
        "Количество точек:",
        len(dataset)
    )

    print()
    print("Этап 3. Сохранение исходных данных.")

    save_csv(
        os.path.join(
            OUTPUT_DIR,
            "dataset.csv"
        ),
        dataset
    )

    print()
    print("Этап 4. Повторяемые эксперименты.")

    repeated_results = run_repeated_experiments(
        dataset
    )

    save_csv(
        os.path.join(
            OUTPUT_DIR,
            "repeated_results.csv"
        ),
        repeated_results
    )

    print()
    print("Этап 5. Агрегация результатов.")

    aggregate = aggregate_results(
        repeated_results
    )

    save_csv(
        os.path.join(
            OUTPUT_DIR,
            "aggregate_results.csv"
        ),
        aggregate
    )

    print()
    print("Этап 6. Leave-one-out.")

    loo_results = run_leave_one_out(
        dataset
    )

    save_csv(
        os.path.join(
            OUTPUT_DIR,
            "leave_one_out.csv"
        ),
        loo_results
    )

    print()
    print("Этап 7. Проверка целочисленных подъёмов.")

    lift_results = run_lift_experiment(
        dataset
    )

    save_csv(
        os.path.join(
            OUTPUT_DIR,
            "lift_stability.csv"
        ),
        lift_results
    )

    print()
    print("Этап 8. Создание JSON-отчёта.")

    save_report(
        dataset,
        aggregate,
        loo_results,
        lift_results
    )

    print()
    print("Этап 9. Создание README.")

    save_readme()

    print()
    print("Этап 10. Создание ZIP-архива.")

    archive_path = create_archive()

    print_results(
        aggregate,
        loo_results,
        lift_results
    )

    print()
    print("=" * 70)
    print("ЭКСПЕРИМЕНТ ЗАВЕРШЁН")
    print("=" * 70)

    print()
    print("Папка результатов:", OUTPUT_DIR)
    print("ZIP-архив:", archive_path)

    print()
    print("Созданы файлы:")

    for filename in sorted(os.listdir(OUTPUT_DIR)):

        print(" -", filename)

    print()
    print("Размер архива:", os.path.getsize(archive_path), "байт")


if __name__ == "__main__":
    main()
