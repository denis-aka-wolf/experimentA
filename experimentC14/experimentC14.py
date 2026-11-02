# -*- coding: utf-8 -*-

import os
import csv
import math
import time
import json
import zipfile
import traceback
import platform
from datetime import datetime

import numpy as np


# ============================================================
# НАСТРОЙКИ ЭКСПЕРИМЕНТА
# ============================================================

FIELD_P = 499
CURVE_A = 0
CURVE_B = 7

GENERATOR = (3, 212)
GROUP_ORDER = 73

OFFSETS = [-3, -2, -1, 0, 1, 2, 3]
TRANSITION_COUNT = 6

PURGE_RADIUS = 6
FOLDS_COUNT = 6

PERMUTATIONS = 99
RANDOM_SEED = 140314

GAMMAS = [0.03, 0.1, 0.3]
ALPHAS = [0.01, 0.1]

MODEL_NAMES = [
    "xy_control",
    "local_orbit",
    "transition_core",
    "lift_dynamics",
    "slope_dynamics",
    "symmetry",
    "combined",
]


# ============================================================
# СЛУЖЕБНЫЕ ФУНКЦИИ
# ============================================================

def log(message):
    print(message, flush=True)


def ensure_directory(path):
    os.makedirs(path, exist_ok=True)


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def modular_signed(value, modulus):
    value = int(value) % modulus

    if value > modulus // 2:
        value -= modulus

    return value


def normalize_number(value, modulus):
    return float(value) / float(modulus)


def safe_float(value):
    value = float(value)

    if not math.isfinite(value):
        return 0.0

    return value


# ============================================================
# АРИФМЕТИКА ЭЛЛИПТИЧЕСКОЙ КРИВОЙ
# ============================================================

def mod_inverse(value, modulus):
    value = int(value) % modulus

    if value == 0:
        raise ZeroDivisionError("Обратного элемента для нуля не существует")

    return pow(value, -1, modulus)


def point_is_on_curve(point):
    if point is None:
        return True

    x, y = point

    left = (y * y) % FIELD_P
    right = (x * x * x + CURVE_A * x + CURVE_B) % FIELD_P

    return left == right


def point_neg(point):
    if point is None:
        return None

    x, y = point

    return x, (-y) % FIELD_P


def point_add(point1, point2):
    if point1 is None:
        return point2

    if point2 is None:
        return point1

    x1, y1 = point1
    x2, y2 = point2

    if x1 == x2 and (y1 + y2) % FIELD_P == 0:
        return None

    if point1 == point2:
        if y1 == 0:
            return None

        numerator = (3 * x1 * x1 + CURVE_A) % FIELD_P
        denominator = (2 * y1) % FIELD_P
        slope = (numerator * mod_inverse(denominator, FIELD_P)) % FIELD_P

    else:
        denominator = (x2 - x1) % FIELD_P

        if denominator == 0:
            return None

        numerator = (y2 - y1) % FIELD_P
        slope = (numerator * mod_inverse(denominator, FIELD_P)) % FIELD_P

    x3 = (slope * slope - x1 - x2) % FIELD_P
    y3 = (slope * (x1 - x3) - y1) % FIELD_P

    result = (x3, y3)

    if not point_is_on_curve(result):
        raise RuntimeError("Полученная точка не принадлежит кривой")

    return result


def point_mul(scalar, point):
    scalar = int(scalar)

    if scalar < 0:
        return point_mul(-scalar, point_neg(point))

    result = None
    addend = point

    while scalar > 0:
        if scalar & 1:
            result = point_add(result, addend)

        addend = point_add(addend, addend)
        scalar >>= 1

    return result


def build_group():
    log("Построение группы точек...")

    if not point_is_on_curve(GENERATOR):
        raise RuntimeError("Генератор не принадлежит эллиптической кривой")

    points = [None]

    current = None

    for scalar in range(1, GROUP_ORDER + 1):
        current = point_add(current, GENERATOR)

        if scalar < GROUP_ORDER:
            points.append(current)

    if points[1] != GENERATOR:
        raise RuntimeError("Ошибка проверки генератора")

    if point_add(points[GROUP_ORDER - 1], GENERATOR) is not None:
        raise RuntimeError("Заявленный порядок группы не подтверждён")

    unique_points = set(points[1:])

    if len(unique_points) != GROUP_ORDER - 1:
        raise RuntimeError("В группе обнаружены повторяющиеся точки")

    for point in points[1:]:
        if point is None or not point_is_on_curve(point):
            raise RuntimeError("В группе обнаружена некорректная точка")

    log("Группа построена.")
    log(f"Порядок группы: {GROUP_ORDER}")
    log(f"Количество ненулевых точек: {len(points) - 1}")

    return points


# ============================================================
# ПЕРЕХОД P1 -> P1 + G
# ============================================================

def transition_data(point1, point2):
    """
    Анализирует переход P1 -> P2, где P2 обязан равняться P1 + G.

    Наклон вычисляется именно для прямой, соединяющей P1 и G.
    """

    expected = point_add(point1, GENERATOR)

    if expected != point2:
        raise RuntimeError(
            "Несогласованность перехода: P2 не равна P1 + G"
        )

    result = {
        "slope": 0.0,
        "slope_mod": 0.0,
        "is_infinity_start": 0.0,
        "is_generator_start": 0.0,
        "is_inverse_generator_start": 0.0,
        "is_vertical": 0.0,
        "is_doubling": 0.0,
        "result_infinity": 0.0,
        "x1": 0.0,
        "y1": 0.0,
        "x2": 0.0,
        "y2": 0.0,
        "dx": 0.0,
        "dy": 0.0,
        "chord_residual": 0.0,
        "tangent_residual": 0.0,
    }

    if point1 is None:
        result["is_infinity_start"] = 1.0
        result["x2"] = normalize_number(point2[0], FIELD_P)
        result["y2"] = normalize_number(point2[1], FIELD_P)
        return result

    x1, y1 = point1

    result["x1"] = normalize_number(x1, FIELD_P)
    result["y1"] = normalize_number(y1, FIELD_P)

    if point2 is None:
        result["result_infinity"] = 1.0
        result["is_vertical"] = 1.0
        result["is_inverse_generator_start"] = 1.0
        return result

    x2, y2 = point2

    result["x2"] = normalize_number(x2, FIELD_P)
    result["y2"] = normalize_number(y2, FIELD_P)

    result["dx"] = normalize_number(
        modular_signed(x2 - x1, FIELD_P),
        FIELD_P
    )

    result["dy"] = normalize_number(
        modular_signed(y2 - y1, FIELD_P),
        FIELD_P
    )

    gx, gy = GENERATOR

    if point1 == GENERATOR:
        result["is_generator_start"] = 1.0
        result["is_doubling"] = 1.0

        if y1 == 0:
            result["is_vertical"] = 1.0
            return result

        numerator = (3 * x1 * x1 + CURVE_A) % FIELD_P
        denominator = (2 * y1) % FIELD_P

    else:
        if x1 == gx and (y1 + gy) % FIELD_P == 0:
            result["is_vertical"] = 1.0
            result["is_inverse_generator_start"] = 1.0
            return result

        numerator = (gy - y1) % FIELD_P
        denominator = (gx - x1) % FIELD_P

    if denominator == 0:
        result["is_vertical"] = 1.0
        return result

    slope_mod = (
        numerator * mod_inverse(denominator, FIELD_P)
    ) % FIELD_P

    slope_signed = modular_signed(slope_mod, FIELD_P)

    result["slope"] = normalize_number(slope_signed, FIELD_P)
    result["slope_mod"] = normalize_number(slope_mod, FIELD_P)

    # Проверяем, что прямая действительно проходит через исходные точки.
    result["chord_residual"] = normalize_number(
        modular_signed(
            y2 - (y1 + slope_mod * (x2 - x1)),
            FIELD_P
        ),
        FIELD_P
    )

    # Для удвоения отдельно сохраняем остаток проверки касательной.
    if result["is_doubling"]:
        tangent_y = (slope_mod * (x1 - x2) - y1) % FIELD_P

        result["tangent_residual"] = normalize_number(
            modular_signed(y2 - tangent_y, FIELD_P),
            FIELD_P
        )

    return result


def transition_vector(data):
    keys = [
        "slope",
        "slope_mod",
        "is_infinity_start",
        "is_generator_start",
        "is_inverse_generator_start",
        "is_vertical",
        "is_doubling",
        "result_infinity",
        "x1",
        "y1",
        "x2",
        "y2",
        "dx",
        "dy",
        "chord_residual",
        "tangent_residual",
    ]

    return [data[key] for key in keys]


# ============================================================
# ГЕОМЕТРИЧЕСКИЕ ПРИЗНАКИ
# ============================================================

def point_vector(point):
    if point is None:
        return [0.0, 0.0, 1.0]

    x, y = point

    return [
        normalize_number(x, FIELD_P),
        normalize_number(y, FIELD_P),
        0.0,
    ]


def point_difference(point1, point2):
    if point1 is None or point2 is None:
        return [0.0, 0.0]

    x1, y1 = point1
    x2, y2 = point2

    return [
        normalize_number(
            modular_signed(x2 - x1, FIELD_P),
            FIELD_P
        ),
        normalize_number(
            modular_signed(y2 - y1, FIELD_P),
            FIELD_P
        ),
    ]


def build_features_for_point(point):
    orbit = [
        point_add(point, point_mul(offset, GENERATOR))
        for offset in OFFSETS
    ]

    if len(orbit) != len(OFFSETS):
        raise RuntimeError("Неверная длина локальной орбиты")

    xy_features = point_vector(point)

    local_features = []

    for orbit_point in orbit:
        local_features.extend(point_vector(orbit_point))

    for index in range(len(orbit) - 1):
        local_features.extend(
            point_difference(orbit[index], orbit[index + 1])
        )

    transition_records = []

    for index in range(len(orbit) - 1):
        record = transition_data(
            orbit[index],
            orbit[index + 1]
        )

        transition_records.append(record)

    transition_vectors = [
        transition_vector(record)
        for record in transition_records
    ]

    transition_core = []

    for record in transition_records:
        transition_core.extend([
            record["slope"],
            record["slope_mod"],
            record["is_vertical"],
            record["is_doubling"],
            record["result_infinity"],
        ])

    lift_dynamics = []

    for index in range(len(transition_vectors)):
        lift_dynamics.extend(transition_vectors[index])

    for index in range(len(transition_vectors) - 1):
        current = np.asarray(
            transition_vectors[index],
            dtype=float
        )

        following = np.asarray(
            transition_vectors[index + 1],
            dtype=float
        )

        difference = following - current

        lift_dynamics.extend(difference.tolist())

    slope_values = [
        record["slope"]
        for record in transition_records
    ]

    slope_mod_values = [
        record["slope_mod"]
        for record in transition_records
    ]

    slope_dynamics = []

    slope_dynamics.extend(slope_values)
    slope_dynamics.extend(slope_mod_values)

    for index in range(len(slope_values) - 1):
        slope_dynamics.append(
            slope_values[index + 1] - slope_values[index]
        )

        slope_dynamics.append(
            slope_mod_values[index + 1] - slope_mod_values[index]
        )

    symmetry_features = []

    for index in range(3):
        left = orbit[index]
        right = orbit[-1 - index]

        symmetry_features.extend(
            point_difference(left, right)
        )

        symmetry_features.append(
            1.0 if left == right else 0.0
        )

        symmetry_features.append(
            1.0 if left == point_neg(right) else 0.0
        )

    symmetry_features.extend(
        point_difference(orbit[0], orbit[-1])
    )

    symmetry_features.extend(
        point_difference(orbit[1], orbit[-2])
    )

    feature_sets = {
        "xy_control": xy_features,
        "local_orbit": local_features,
        "transition_core": transition_core,
        "lift_dynamics": lift_dynamics,
        "slope_dynamics": slope_dynamics,
        "symmetry": symmetry_features,
    }

    feature_sets["combined"] = []

    for model_name in MODEL_NAMES:
        if model_name == "combined":
            continue

        feature_sets["combined"].extend(
            feature_sets[model_name]
        )

    raw = {
        "orbit": orbit,
        "transitions": transition_records,
    }

    return feature_sets, raw


def build_dataset(group_points):
    log("Построение набора признаков...")

    datasets = {
        model_name: []
        for model_name in MODEL_NAMES
    }

    labels = []
    raw_records = []

    for scalar in range(1, GROUP_ORDER):
        point = group_points[scalar]

        feature_sets, raw = build_features_for_point(point)

        labels.append(scalar)
        raw_records.append(raw)

        for model_name in MODEL_NAMES:
            datasets[model_name].append(
                feature_sets[model_name]
            )

    for model_name in MODEL_NAMES:
        datasets[model_name] = np.asarray(
            datasets[model_name],
            dtype=float
        )

        if not np.all(np.isfinite(datasets[model_name])):
            raise RuntimeError(
                f"Некорректные признаки в модели {model_name}"
            )

    labels = np.asarray(labels, dtype=int)

    log(f"Количество образцов: {len(labels)}")

    for model_name in MODEL_NAMES:
        log(
            f"Модель {model_name}: "
            f"{datasets[model_name].shape[1]} признаков"
        )

    return datasets, labels, raw_records


# ============================================================
# КРУГОВОЕ ПРЕДСТАВЛЕНИЕ СКАЛЯРА
# ============================================================

def labels_to_circle(labels):
    labels = np.asarray(labels, dtype=float)

    angles = labels * (2.0 * math.pi / GROUP_ORDER)

    return np.column_stack([
        np.cos(angles),
        np.sin(angles),
    ])


def decode_circle(prediction):
    """
    Декодирует в ближайшее допустимое направление.

    Нулевой скаляр исключён. Возможные ответы: 1..72.
    """

    prediction = np.asarray(prediction, dtype=float)

    if prediction.ndim == 1:
        prediction = prediction.reshape(1, -1)

    candidates = np.arange(1, GROUP_ORDER, dtype=int)
    candidate_angles = (
        candidates * (2.0 * math.pi / GROUP_ORDER)
    )

    candidate_vectors = np.column_stack([
        np.cos(candidate_angles),
        np.sin(candidate_angles),
    ])

    norms = np.linalg.norm(prediction, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0

    normalized = prediction / norms

    scores = normalized @ candidate_vectors.T

    best_indices = np.argmax(scores, axis=1)

    return candidates[best_indices]


def circular_distance(value1, value2):
    difference = abs(int(value1) - int(value2))

    return min(
        difference,
        GROUP_ORDER - difference
    )


def calculate_metrics(true_labels, predicted_labels):
    errors = np.asarray([
        circular_distance(true_value, predicted_value)
        for true_value, predicted_value
        in zip(true_labels, predicted_labels)
    ], dtype=float)

    mae = float(np.mean(errors))
    accuracy = float(np.mean(errors == 0.0))

    return mae, accuracy, errors


# ============================================================
# БЛОЧНАЯ ПЕРЕКРЁСТНАЯ ПРОВЕРКА
# ============================================================

def build_folds():
    indices = np.arange(GROUP_ORDER - 1)

    test_blocks = np.array_split(indices, FOLDS_COUNT)

    folds = []

    for fold_number, test_indices in enumerate(test_blocks, start=1):
        train_indices = []

        for candidate in indices:
            if candidate in test_indices:
                continue

            candidate_label = int(candidate) + 1

            too_close = False

            for test_index in test_indices:
                test_label = int(test_index) + 1

                if circular_distance(
                    candidate_label,
                    test_label
                ) <= PURGE_RADIUS:
                    too_close = True
                    break

            if not too_close:
                train_indices.append(int(candidate))

        train_indices = np.asarray(train_indices, dtype=int)
        test_indices = np.asarray(test_indices, dtype=int)

        if len(train_indices) == 0 or len(test_indices) == 0:
            raise RuntimeError("Пустая обучающая или тестовая выборка")

        folds.append({
            "number": fold_number,
            "train": train_indices,
            "test": test_indices,
        })

        log(
            f"Блок {fold_number}: "
            f"обучение {len(train_indices)}, "
            f"тест {len(test_indices)}"
        )

    return folds


# ============================================================
# RBF KERNEL RIDGE REGRESSION
# ============================================================

def standardize_train_test(train_x, test_x):
    mean = np.mean(train_x, axis=0)
    std = np.std(train_x, axis=0)

    std[std < 1e-12] = 1.0

    train_scaled = (train_x - mean) / std
    test_scaled = (test_x - mean) / std

    return train_scaled, test_scaled


def rbf_kernel(matrix1, matrix2, gamma):
    squared1 = np.sum(matrix1 * matrix1, axis=1).reshape(-1, 1)
    squared2 = np.sum(matrix2 * matrix2, axis=1).reshape(1, -1)

    distances = squared1 + squared2 - 2.0 * matrix1 @ matrix2.T
    distances = np.maximum(distances, 0.0)

    return np.exp(-gamma * distances)


def fit_predict_krr(train_x, train_y, test_x, gamma, alpha):
    train_scaled, test_scaled = standardize_train_test(
        train_x,
        test_x
    )

    kernel_train = rbf_kernel(
        train_scaled,
        train_scaled,
        gamma
    )

    kernel_test = rbf_kernel(
        test_scaled,
        train_scaled,
        gamma
    )

    regularized = kernel_train.copy()

    regularized.flat[::len(regularized) + 1] += alpha

    try:
        coefficients = np.linalg.solve(
            regularized,
            train_y
        )
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(
            regularized,
            train_y,
            rcond=None
        )[0]

    predictions = kernel_test @ coefficients

    return predictions


# ============================================================
# ОЦЕНКА ОДНОЙ КОНФИГУРАЦИИ
# ============================================================

def evaluate_configuration(
    features,
    labels,
    folds,
    gamma,
    alpha,
    collect_predictions=False
):
    target_vectors = labels_to_circle(labels)

    predicted_labels = np.zeros(
        len(labels),
        dtype=int
    )

    fold_metrics = []

    for fold in folds:
        train_indices = fold["train"]
        test_indices = fold["test"]

        train_x = features[train_indices]
        test_x = features[test_indices]

        train_y = target_vectors[train_indices]

        prediction_vectors = fit_predict_krr(
            train_x,
            train_y,
            test_x,
            gamma,
            alpha
        )

        fold_predictions = decode_circle(
            prediction_vectors
        )

        predicted_labels[test_indices] = fold_predictions

        true_fold_labels = labels[test_indices]

        fold_mae, fold_accuracy, _ = calculate_metrics(
            true_fold_labels,
            fold_predictions
        )

        fold_metrics.append({
            "fold": fold["number"],
            "mae": fold_mae,
            "accuracy": fold_accuracy,
            "train_size": len(train_indices),
            "test_size": len(test_indices),
        })

    mae, accuracy, errors = calculate_metrics(
        labels,
        predicted_labels
    )

    result = {
        "mae": mae,
        "accuracy": accuracy,
        "fold_metrics": fold_metrics,
    }

    if collect_predictions:
        result["predictions"] = predicted_labels
        result["errors"] = errors

    return result


# ============================================================
# ПОЛНАЯ ОЦЕНКА ВСЕХ МОДЕЛЕЙ
# ============================================================

def evaluate_all(
    datasets,
    labels,
    folds,
    collect_predictions=False
):
    results = []

    total = (
        len(MODEL_NAMES)
        * len(GAMMAS)
        * len(ALPHAS)
    )

    current = 0

    for model_name in MODEL_NAMES:
        features = datasets[model_name]

        for gamma in GAMMAS:
            for alpha in ALPHAS:
                current += 1

                result = evaluate_configuration(
                    features,
                    labels,
                    folds,
                    gamma,
                    alpha,
                    collect_predictions=collect_predictions
                )

                record = {
                    "model": model_name,
                    "gamma": gamma,
                    "alpha": alpha,
                    "mae": result["mae"],
                    "accuracy": result["accuracy"],
                    "fold_metrics": result["fold_metrics"],
                }

                if collect_predictions:
                    record["predictions"] = result["predictions"]
                    record["errors"] = result["errors"]

                results.append(record)

                log(
                    f"[{current}/{total}] "
                    f"{model_name}, "
                    f"гамма={gamma}, альфа={alpha}, "
                    f"MAE={result['mae']:.6f}, "
                    f"точность={result['accuracy']:.6f}"
                )

    return results


def select_best(results):
    return min(
        results,
        key=lambda item: (
            item["mae"],
            -item["accuracy"]
        )
    )


# ============================================================
# ПЕРЕСТАНОВОЧНЫЙ ТЕСТ
# ============================================================

def permutation_test(
    datasets,
    labels,
    folds,
    observed_best,
    random_generator
):
    log("")
    log("Запуск перестановочного теста...")
    log(
        "В каждой перестановке одна и та же "
        "переставленная последовательность меток "
        "используется для всех моделей."
    )

    null_best_mae = []
    null_best_accuracy = []

    for permutation_number in range(1, PERMUTATIONS + 1):
        permuted_labels = random_generator.permutation(labels)

        permutation_results = evaluate_all(
            datasets,
            permuted_labels,
            folds,
            collect_predictions=False
        )

        best_permutation = select_best(
            permutation_results
        )

        null_best_mae.append(
            best_permutation["mae"]
        )

        null_best_accuracy.append(
            best_permutation["accuracy"]
        )

        log(
            f"Перестановка {permutation_number}/"
            f"{PERMUTATIONS}: "
            f"лучший MAE={best_permutation['mae']:.6f}, "
            f"точность={best_permutation['accuracy']:.6f}"
        )

    null_best_mae = np.asarray(null_best_mae, dtype=float)
    null_best_accuracy = np.asarray(
        null_best_accuracy,
        dtype=float
    )

    p_mae = (
        1.0
        + np.sum(null_best_mae <= observed_best["mae"])
    ) / (PERMUTATIONS + 1.0)

    p_accuracy = (
        1.0
        + np.sum(
            null_best_accuracy >= observed_best["accuracy"]
        )
    ) / (PERMUTATIONS + 1.0)

    statistics = {
        "observed_mae": observed_best["mae"],
        "observed_accuracy": observed_best["accuracy"],
        "null_mean_mae": float(np.mean(null_best_mae)),
        "null_std_mae": float(np.std(null_best_mae)),
        "null_min_mae": float(np.min(null_best_mae)),
        "null_max_mae": float(np.max(null_best_mae)),
        "null_mean_accuracy": float(
            np.mean(null_best_accuracy)
        ),
        "null_max_accuracy": float(
            np.max(null_best_accuracy)
        ),
        "p_value_mae": float(p_mae),
        "p_value_accuracy": float(p_accuracy),
        "permutations": PERMUTATIONS,
    }

    return statistics, null_best_mae, null_best_accuracy


# ============================================================
# СОХРАНЕНИЕ CSV
# ============================================================

def save_group_csv(path, group_points):
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file, delimiter=";")

        writer.writerow([
            "scalar",
            "x",
            "y",
            "is_infinity",
        ])

        writer.writerow([0, "", "", 1])

        for scalar in range(1, GROUP_ORDER):
            x, y = group_points[scalar]

            writer.writerow([
                scalar,
                x,
                y,
                0,
            ])


def save_results_csv(path, results):
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file, delimiter=";")

        writer.writerow([
            "model",
            "gamma",
            "alpha",
            "mae",
            "accuracy",
        ])

        for result in sorted(
            results,
            key=lambda item: (
                item["mae"],
                -item["accuracy"]
            )
        ):
            writer.writerow([
                result["model"],
                result["gamma"],
                result["alpha"],
                f"{result['mae']:.10f}",
                f"{result['accuracy']:.10f}",
            ])


def save_predictions_csv(path, labels, best_result):
    predictions = best_result["predictions"]
    errors = best_result["errors"]

    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file, delimiter=";")

        writer.writerow([
            "true_scalar",
            "predicted_scalar",
            "circular_error",
            "is_correct",
        ])

        for true_value, predicted_value, error in zip(
            labels,
            predictions,
            errors
        ):
            writer.writerow([
                int(true_value),
                int(predicted_value),
                int(error),
                int(error == 0),
            ])


def save_permutation_csv(
    path,
    null_best_mae,
    null_best_accuracy
):
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file, delimiter=";")

        writer.writerow([
            "permutation",
            "best_mae",
            "best_accuracy",
        ])

        for index, (mae, accuracy) in enumerate(
            zip(null_best_mae, null_best_accuracy),
            start=1
        ):
            writer.writerow([
                index,
                f"{mae:.10f}",
                f"{accuracy:.10f}",
            ])


def save_fold_metrics_csv(path, results):
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file, delimiter=";")

        writer.writerow([
            "model",
            "gamma",
            "alpha",
            "fold",
            "mae",
            "accuracy",
            "train_size",
            "test_size",
        ])

        for result in results:
            for fold in result["fold_metrics"]:
                writer.writerow([
                    result["model"],
                    result["gamma"],
                    result["alpha"],
                    fold["fold"],
                    f"{fold['mae']:.10f}",
                    f"{fold['accuracy']:.10f}",
                    fold["train_size"],
                    fold["test_size"],
                ])


# ============================================================
# ОТЧЁТ И КОНФИГУРАЦИЯ
# ============================================================

def build_configuration():
    return {
        "experiment": "C14",
        "title": "Геометрическая кривизна алгебраических переходов",
        "field_p": FIELD_P,
        "curve_a": CURVE_A,
        "curve_b": CURVE_B,
        "generator": list(GENERATOR),
        "group_order": GROUP_ORDER,
        "offsets": OFFSETS,
        "transition_count": TRANSITION_COUNT,
        "purge_radius": PURGE_RADIUS,
        "folds_count": FOLDS_COUNT,
        "permutations": PERMUTATIONS,
        "random_seed": RANDOM_SEED,
        "gammas": GAMMAS,
        "alphas": ALPHAS,
        "models": MODEL_NAMES,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
    }


def write_report(
    path,
    results,
    best_result,
    permutation_statistics,
    folds,
    elapsed
):
    with open(path, "w", encoding="utf-8") as file:
        file.write(
            "ЭКСПЕРИМЕНТ C14\n"
            "Геометрическая кривизна алгебраических переходов\n"
        )

        file.write("=" * 72 + "\n\n")

        file.write("1. ПАРАМЕТРЫ КРИВОЙ\n")
        file.write(f"Поле: p = {FIELD_P}\n")
        file.write(
            f"Кривая: y^2 = x^3 + {CURVE_B} (mod {FIELD_P})\n"
        )
        file.write(f"Генератор: G = {GENERATOR}\n")
        file.write(f"Порядок группы: {GROUP_ORDER}\n\n")

        file.write("2. ПАРАМЕТРЫ ЭКСПЕРИМЕНТА\n")
        file.write(f"Локальные смещения: {OFFSETS}\n")
        file.write(f"Радиус исключения: {PURGE_RADIUS}\n")
        file.write(f"Количество блоков: {FOLDS_COUNT}\n")
        file.write(f"Перестановок: {PERMUTATIONS}\n")
        file.write(f"Гамма: {GAMMAS}\n")
        file.write(f"Альфа: {ALPHAS}\n\n")

        file.write("3. БЛОЧНАЯ ПРОВЕРКА\n")

        for fold in folds:
            file.write(
                f"Блок {fold['number']}: "
                f"обучение={len(fold['train'])}, "
                f"тест={len(fold['test'])}\n"
            )

        file.write("\n4. РЕЗУЛЬТАТЫ ВСЕХ КОНФИГУРАЦИЙ\n")

        sorted_results = sorted(
            results,
            key=lambda item: (
                item["mae"],
                -item["accuracy"]
            )
        )

        for index, result in enumerate(sorted_results, start=1):
            file.write(
                f"{index:02d}. "
                f"{result['model']}, "
                f"gamma={result['gamma']}, "
                f"alpha={result['alpha']}, "
                f"MAE={result['mae']:.8f}, "
                f"точность={result['accuracy']:.8f}\n"
            )

        file.write("\n5. ЛУЧШАЯ КОНФИГУРАЦИЯ\n")
        file.write(f"Модель: {best_result['model']}\n")
        file.write(f"Гамма: {best_result['gamma']}\n")
        file.write(f"Альфа: {best_result['alpha']}\n")
        file.write(f"MAE: {best_result['mae']:.8f}\n")
        file.write(
            f"Точность: {best_result['accuracy']:.8f}\n"
        )

        file.write("\n6. ПЕРЕСТАНОВОЧНЫЙ ТЕСТ\n")

        for key, value in permutation_statistics.items():
            file.write(f"{key}: {value}\n")

        file.write("\n7. ИНТЕРПРЕТАЦИЯ\n")

        p_mae = permutation_statistics["p_value_mae"]
        p_accuracy = permutation_statistics["p_value_accuracy"]

        if p_mae < 0.05 or p_accuracy < 0.05:
            file.write(
                "Обнаружен результат, требующий дополнительной "
                "проверки: хотя бы одно перестановочное значение "
                "p меньше 0.05. Это не является доказательством "
                "эффективного решения задачи дискретного логарифма.\n"
            )
        else:
            file.write(
                "Статистически значимого преимущества над "
                "перестановочным контролем не обнаружено. "
                "На данном наборе данных выбранные признаки "
                "не демонстрируют устойчивого преимущества "
                "для восстановления скаляра.\n"
            )

        file.write(
            "\nДаже положительный результат на игрушечной кривой "
            "не доказывает возможность эффективного решения "
            "ECDLP на криптографических кривых.\n"
        )

        file.write(
            f"\nВремя выполнения: {elapsed:.3f} секунд\n"
        )


def save_raw_features(path, datasets, labels):
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file, delimiter=";")

        header = ["scalar", "model", "feature_index", "value"]
        writer.writerow(header)

        for model_name in MODEL_NAMES:
            matrix = datasets[model_name]

            for row_index, scalar in enumerate(labels):
                for feature_index, value in enumerate(
                    matrix[row_index]
                ):
                    writer.writerow([
                        int(scalar),
                        model_name,
                        feature_index,
                        f"{float(value):.12g}",
                    ])


def create_archive(results_directory, archive_path):
    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for root, _, filenames in os.walk(results_directory):
            for filename in filenames:
                full_path = os.path.join(root, filename)

                relative_path = os.path.relpath(
                    full_path,
                    os.path.dirname(results_directory)
                )

                archive.write(
                    full_path,
                    arcname=relative_path
                )


# ============================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================

def main():
    start_time = time.time()

    run_stamp = timestamp()

    results_directory = os.path.abspath(
        f"experimentC14_results_{run_stamp}"
    )

    ensure_directory(results_directory)

    log("=" * 72)
    log("ЭКСПЕРИМЕНТ C14")
    log("Геометрическая кривизна алгебраических переходов")
    log("=" * 72)

    log(f"Каталог результатов: {results_directory}")

    random_generator = np.random.default_rng(
        RANDOM_SEED
    )

    configuration = build_configuration()

    configuration_path = os.path.join(
        results_directory,
        "C14_configuration.json"
    )

    with open(
        configuration_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            configuration,
            file,
            ensure_ascii=False,
            indent=4
        )

    group_points = build_group()

    datasets, labels, raw_records = build_dataset(
        group_points
    )

    folds = build_folds()

    group_csv_path = os.path.join(
        results_directory,
        "C14_group.csv"
    )

    save_group_csv(
        group_csv_path,
        group_points
    )

    log("")
    log("Оценка исходных данных...")
    log("")

    results = evaluate_all(
        datasets,
        labels,
        folds,
        collect_predictions=False
    )

    best_result = select_best(results)

    log("")
    log("ЛУЧШАЯ КОНФИГУРАЦИЯ:")
    log(f"Модель: {best_result['model']}")
    log(f"Гамма: {best_result['gamma']}")
    log(f"Альфа: {best_result['alpha']}")
    log(f"MAE: {best_result['mae']:.8f}")
    log(f"Точность: {best_result['accuracy']:.8f}")

    log("")
    log("Повторное вычисление предсказаний лучшей конфигурации...")

    best_prediction_result = evaluate_configuration(
        datasets[best_result["model"]],
        labels,
        folds,
        best_result["gamma"],
        best_result["alpha"],
        collect_predictions=True
    )

    best_result["predictions"] = (
        best_prediction_result["predictions"]
    )

    best_result["errors"] = (
        best_prediction_result["errors"]
    )

    best_result["fold_metrics"] = (
        best_prediction_result["fold_metrics"]
    )

    permutation_statistics, null_best_mae, null_best_accuracy = (
        permutation_test(
            datasets,
            labels,
            folds,
            best_result,
            random_generator
        )
    )

    results_csv_path = os.path.join(
        results_directory,
        "C14_results.csv"
    )

    predictions_csv_path = os.path.join(
        results_directory,
        "C14_predictions.csv"
    )

    permutation_csv_path = os.path.join(
        results_directory,
        "C14_permutations.csv"
    )

    fold_csv_path = os.path.join(
        results_directory,
        "C14_fold_metrics.csv"
    )

    features_csv_path = os.path.join(
        results_directory,
        "C14_features.csv"
    )

    save_results_csv(
        results_csv_path,
        results
    )

    save_predictions_csv(
        predictions_csv_path,
        labels,
        best_result
    )

    save_permutation_csv(
        permutation_csv_path,
        null_best_mae,
        null_best_accuracy
    )

    save_fold_metrics_csv(
        fold_csv_path,
        results
    )

    save_raw_features(
        features_csv_path,
        datasets,
        labels
    )

    elapsed = time.time() - start_time

    report_path = os.path.join(
        results_directory,
        "C14_report.txt"
    )

    write_report(
        report_path,
        results,
        best_result,
        permutation_statistics,
        folds,
        elapsed
    )

    log("")
    log("=" * 72)
    log("ИТОГИ ПЕРЕСТАНОВОЧНОГО ТЕСТА")
    log("=" * 72)

    log(
        f"Наблюдаемый MAE: "
        f"{permutation_statistics['observed_mae']:.8f}"
    )

    log(
        f"Средний нулевой MAE: "
        f"{permutation_statistics['null_mean_mae']:.8f}"
    )

    log(
        f"Минимальный нулевой MAE: "
        f"{permutation_statistics['null_min_mae']:.8f}"
    )

    log(
        f"p-value для MAE: "
        f"{permutation_statistics['p_value_mae']:.6f}"
    )

    log(
        f"p-value для точности: "
        f"{permutation_statistics['p_value_accuracy']:.6f}"
    )

    log("")
    log(f"Время выполнения: {elapsed:.2f} секунд")

    archive_path = os.path.abspath(
        f"experimentC14_{run_stamp}.zip"
    )

    create_archive(
        results_directory,
        archive_path
    )

    log("")
    log("ЭКСПЕРИМЕНТ ЗАВЕРШЁН")
    log(f"Каталог результатов: {results_directory}")
    log(f"ZIP-архив: {archive_path}")
    log("=" * 72)


if __name__ == "__main__":
    try:
        main()

    except Exception:
        log("")
        log("ОШИБКА ВО ВРЕМЯ ВЫПОЛНЕНИЯ ЭКСПЕРИМЕНТА")
        log(traceback.format_exc())
        raise