
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

RESULTS_DIRECTORY = None
LOG_FILE = None


# ============================================================
# ЖУРНАЛИРОВАНИЕ
# ============================================================

def log(message):
    text = str(message)
    print(text, flush=True)

    if LOG_FILE is not None:
        LOG_FILE.write(text + "\n")
        LOG_FILE.flush()


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_directory(path):
    os.makedirs(path, exist_ok=True)


# ============================================================
# МОДУЛЬНАЯ АРИФМЕТИКА
# ============================================================

def modular_signed(value, modulus):
    value = int(value) % modulus

    if value > modulus // 2:
        value -= modulus

    return value


def mod_inverse(value, modulus):
    value = int(value) % modulus

    if value == 0:
        raise ZeroDivisionError(
            "Обратного элемента для нуля не существует"
        )

    return pow(value, -1, modulus)


def normalize_number(value, modulus):
    return float(value) / float(modulus)


# ============================================================
# АРИФМЕТИКА ЭЛЛИПТИЧЕСКОЙ КРИВОЙ
# ============================================================

def point_is_on_curve(point):
    if point is None:
        return True

    x, y = point

    left = (y * y) % FIELD_P
    right = (
        x * x * x
        + CURVE_A * x
        + CURVE_B
    ) % FIELD_P

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

        numerator = (
            3 * x1 * x1 + CURVE_A
        ) % FIELD_P

        denominator = (2 * y1) % FIELD_P

        slope = (
            numerator
            * mod_inverse(denominator, FIELD_P)
        ) % FIELD_P

    else:
        denominator = (x2 - x1) % FIELD_P

        if denominator == 0:
            return None

        numerator = (y2 - y1) % FIELD_P

        slope = (
            numerator
            * mod_inverse(denominator, FIELD_P)
        ) % FIELD_P

    x3 = (
        slope * slope - x1 - x2
    ) % FIELD_P

    y3 = (
        slope * (x1 - x3) - y1
    ) % FIELD_P

    result = (x3, y3)

    if not point_is_on_curve(result):
        raise RuntimeError(
            "Полученная точка не принадлежит кривой"
        )

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
        raise RuntimeError(
            "Генератор не принадлежит кривой"
        )

    points = [None]
    current = None

    for scalar in range(1, GROUP_ORDER + 1):
        current = point_add(current, GENERATOR)

        if scalar < GROUP_ORDER:
            points.append(current)

    if points[1] != GENERATOR:
        raise RuntimeError(
            "Ошибка проверки генератора"
        )

    if point_add(
        points[GROUP_ORDER - 1],
        GENERATOR
    ) is not None:
        raise RuntimeError(
            "Заявленный порядок группы не подтверждён"
        )

    if len(set(points[1:])) != GROUP_ORDER - 1:
        raise RuntimeError(
            "В группе обнаружены повторяющиеся точки"
        )

    for point in points[1:]:
        if point is None or not point_is_on_curve(point):
            raise RuntimeError(
                "В группе обнаружена некорректная точка"
            )

    log("Группа построена.")
    log(f"Порядок группы: {GROUP_ORDER}")
    log(f"Количество ненулевых точек: {len(points) - 1}")

    return points


# ============================================================
# ПЕРЕХОД P1 -> P1 + G
# ============================================================

def transition_data(point1, point2):
    expected = point_add(point1, GENERATOR)

    if expected != point2:
        raise RuntimeError(
            "Несогласованность перехода: "
            "P2 не равна P1 + G"
        )

    data = {
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
        "addition_residual_x": 0.0,
        "addition_residual_y": 0.0,
    }

    if point1 is None:
        data["is_infinity_start"] = 1.0

        if point2 is not None:
            data["x2"] = normalize_number(
                point2[0], FIELD_P
            )
            data["y2"] = normalize_number(
                point2[1], FIELD_P
            )

        return data

    x1, y1 = point1
    gx, gy = GENERATOR

    data["x1"] = normalize_number(x1, FIELD_P)
    data["y1"] = normalize_number(y1, FIELD_P)

    if point2 is None:
        data["result_infinity"] = 1.0
        data["is_vertical"] = 1.0
        data["is_inverse_generator_start"] = 1.0
        return data

    x2, y2 = point2

    data["x2"] = normalize_number(x2, FIELD_P)
    data["y2"] = normalize_number(y2, FIELD_P)

    data["dx"] = normalize_number(
        modular_signed(x2 - x1, FIELD_P),
        FIELD_P
    )

    data["dy"] = normalize_number(
        modular_signed(y2 - y1, FIELD_P),
        FIELD_P
    )

    if point1 == GENERATOR:
        data["is_generator_start"] = 1.0
        data["is_doubling"] = 1.0

        numerator = (
            3 * x1 * x1 + CURVE_A
        ) % FIELD_P

        denominator = (2 * y1) % FIELD_P

    else:
        if (
            x1 == gx
            and (y1 + gy) % FIELD_P == 0
        ):
            data["is_vertical"] = 1.0
            data["is_inverse_generator_start"] = 1.0
            return data

        numerator = (gy - y1) % FIELD_P
        denominator = (gx - x1) % FIELD_P

    if denominator == 0:
        data["is_vertical"] = 1.0
        return data

    slope_mod = (
        numerator
        * mod_inverse(denominator, FIELD_P)
    ) % FIELD_P

    slope_signed = modular_signed(
        slope_mod,
        FIELD_P
    )

    data["slope"] = normalize_number(
        slope_signed,
        FIELD_P
    )

    data["slope_mod"] = normalize_number(
        slope_mod,
        FIELD_P
    )

    # Независимо восстанавливаем координаты результата сложения.
    if data["is_doubling"]:
        calculated_x = (
            slope_mod * slope_mod - 2 * x1
        ) % FIELD_P
    else:
        calculated_x = (
            slope_mod * slope_mod - x1 - gx
        ) % FIELD_P

    calculated_y = (
        slope_mod * (x1 - calculated_x) - y1
    ) % FIELD_P

    data["addition_residual_x"] = normalize_number(
        modular_signed(
            calculated_x - x2,
            FIELD_P
        ),
        FIELD_P
    )

    data["addition_residual_y"] = normalize_number(
        modular_signed(
            calculated_y - y2,
            FIELD_P
        ),
        FIELD_P
    )

    if (
        calculated_x != x2
        or calculated_y != y2
    ):
        raise RuntimeError(
            "Независимая проверка сложения не пройдена"
        )

    return data


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
        "addition_residual_x",
        "addition_residual_y",
    ]

    return [data[key] for key in keys]


# ============================================================
# ПОСТРОЕНИЕ ГЕОМЕТРИЧЕСКИХ ПРИЗНАКОВ
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
        point_add(
            point,
            point_mul(offset, GENERATOR)
        )
        for offset in OFFSETS
    ]

    xy_features = point_vector(point)

    local_features = []

    for orbit_point in orbit:
        local_features.extend(
            point_vector(orbit_point)
        )

    for index in range(len(orbit) - 1):
        local_features.extend(
            point_difference(
                orbit[index],
                orbit[index + 1]
            )
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

    for vector in transition_vectors:
        lift_dynamics.extend(vector)

    for index in range(len(transition_vectors) - 1):
        current = np.asarray(
            transition_vectors[index],
            dtype=float
        )

        following = np.asarray(
            transition_vectors[index + 1],
            dtype=float
        )

        lift_dynamics.extend(
            (following - current).tolist()
        )

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
            slope_values[index + 1]
            - slope_values[index]
        )

        slope_dynamics.append(
            slope_mod_values[index + 1]
            - slope_mod_values[index]
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

    combined = []

    for model_name in MODEL_NAMES:
        if model_name != "combined":
            combined.extend(feature_sets[model_name])

    feature_sets["combined"] = combined

    return feature_sets


def build_dataset(group_points):
    log("Построение набора признаков...")

    datasets = {
        model_name: []
        for model_name in MODEL_NAMES
    }

    labels = []

    for scalar in range(1, GROUP_ORDER):
        point = group_points[scalar]

        feature_sets = build_features_for_point(point)

        labels.append(scalar)

        for model_name in MODEL_NAMES:
            datasets[model_name].append(
                feature_sets[model_name]
            )

    for model_name in MODEL_NAMES:
        datasets[model_name] = np.asarray(
            datasets[model_name],
            dtype=float
        )

        if not np.all(
            np.isfinite(datasets[model_name])
        ):
            raise RuntimeError(
                f"Некорректные признаки модели {model_name}"
            )

    labels = np.asarray(labels, dtype=int)

    log(f"Количество образцов: {len(labels)}")

    for model_name in MODEL_NAMES:
        log(
            f"Модель {model_name}: "
            f"{datasets[model_name].shape[1]} признаков"
        )

    return datasets, labels


# ============================================================
# КРУГОВОЕ ПРЕДСТАВЛЕНИЕ СКАЛЯРА
# ============================================================

def labels_to_circle(labels):
    labels = np.asarray(labels, dtype=float)

    angles = (
        labels
        * (2.0 * math.pi / GROUP_ORDER)
    )

    return np.column_stack([
        np.cos(angles),
        np.sin(angles),
    ])


def decode_circle(prediction):
    prediction = np.asarray(
        prediction,
        dtype=float
    )

    if prediction.ndim == 1:
        prediction = prediction.reshape(1, -1)

    candidates = np.arange(
        1,
        GROUP_ORDER,
        dtype=int
    )

    candidate_angles = (
        candidates
        * (2.0 * math.pi / GROUP_ORDER)
    )

    candidate_vectors = np.column_stack([
        np.cos(candidate_angles),
        np.sin(candidate_angles),
    ])

    norms = np.linalg.norm(
        prediction,
        axis=1,
        keepdims=True
    )

    norms[norms == 0.0] = 1.0

    normalized = prediction / norms

    scores = normalized @ candidate_vectors.T

    best_indices = np.argmax(
        scores,
        axis=1
    )

    return candidates[best_indices]


def circular_distance(value1, value2):
    difference = abs(
        int(value1) - int(value2)
    )

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

    test_blocks = np.array_split(
        indices,
        FOLDS_COUNT
    )

    folds = []

    for fold_number, test_indices in enumerate(
        test_blocks,
        start=1
    ):
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

        train_indices = np.asarray(
            train_indices,
            dtype=int
        )

        test_indices = np.asarray(
            test_indices,
            dtype=int
        )

        if (
            len(train_indices) == 0
            or len(test_indices) == 0
        ):
            raise RuntimeError(
                "Пустая обучающая или тестовая выборка"
            )

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
    mean = np.mean(
        train_x,
        axis=0
    )

    std = np.std(
        train_x,
        axis=0
    )

    std[std < 1e-12] = 1.0

    train_scaled = (
        train_x - mean
    ) / std

    test_scaled = (
        test_x - mean
    ) / std

    return train_scaled, test_scaled


def rbf_kernel(matrix1, matrix2, gamma):
    squared1 = np.sum(
        matrix1 * matrix1,
        axis=1
    ).reshape(-1, 1)

    squared2 = np.sum(
        matrix2 * matrix2,
        axis=1
    ).reshape(1, -1)

    distances = (
        squared1
        + squared2
        - 2.0 * matrix1 @ matrix2.T
    )

    distances = np.maximum(
        distances,
        0.0
    )

    return np.exp(
        -gamma * distances
    )


def fit_predict_krr(
    train_x,
    train_y,
    test_x,
    gamma,
    alpha
):
    train_scaled, test_scaled = (
        standardize_train_test(
            train_x,
            test_x
        )
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

    regularized.flat[
        ::len(regularized) + 1
    ] += alpha

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

    return kernel_test @ coefficients


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

        prediction_vectors = fit_predict_krr(
            features[train_indices],
            target_vectors[train_indices],
            features[test_indices],
            gamma,
            alpha
        )

        fold_predictions = decode_circle(
            prediction_vectors
        )

        predicted_labels[test_indices] = (
            fold_predictions
        )

        fold_mae, fold_accuracy, _ = (
            calculate_metrics(
                labels[test_indices],
                fold_predictions
            )
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
# ОЦЕНКА ВСЕХ 42 КОНФИГУРАЦИЙ
# ============================================================

def evaluate_all(
    datasets,
    labels,
    folds,
    verbose=True
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
                    collect_predictions=False
                )

                record = {
                    "model": model_name,
                    "gamma": gamma,
                    "alpha": alpha,
                    "mae": result["mae"],
                    "accuracy": result["accuracy"],
                    "fold_metrics": result["fold_metrics"],
                }

                results.append(record)

                if verbose:
                    log(
                        f"[{current}/{total}] "
                        f"{model_name}, "
                        f"гамма={gamma}, "
                        f"альфа={alpha}, "
                        f"MAE={result['mae']:.6f}, "
                        f"точность={result['accuracy']:.6f}"
                    )

    if len(results) != total:
        raise RuntimeError(
            "Получено неверное количество конфигураций"
        )

    return results


def select_best(results):
    if not results:
        raise RuntimeError(
            "Невозможно выбрать результат из пустого списка"
        )

    return min(
        results,
        key=lambda item: (
            item["mae"],
            -item["accuracy"]
        )
    )


def verify_observed_results(results, best_result):
    expected_count = (
        len(MODEL_NAMES)
        * len(GAMMAS)
        * len(ALPHAS)
    )

    if len(results) != expected_count:
        raise RuntimeError(
            "Неверное количество наблюдаемых конфигураций"
        )

    actual_best = select_best(results)

    if not math.isclose(
        actual_best["mae"],
        best_result["mae"],
        rel_tol=0.0,
        abs_tol=1e-12
    ):
        raise RuntimeError(
            "Несогласованность наблюдаемого MAE: "
            f"минимум таблицы={actual_best['mae']}, "
            f"переданное значение={best_result['mae']}"
        )

    if actual_best["model"] != best_result["model"]:
        raise RuntimeError(
            "Несогласованность названия лучшей модели"
        )

    if actual_best["gamma"] != best_result["gamma"]:
        raise RuntimeError(
            "Несогласованность параметра гамма"
        )

    if actual_best["alpha"] != best_result["alpha"]:
        raise RuntimeError(
            "Несогласованность параметра альфа"
        )

    return actual_best


# ============================================================
# ПЕРЕСТАНОВОЧНЫЙ ТЕСТ
# ============================================================

def permutation_test(
    datasets,
    labels,
    folds,
    observed_results,
    random_generator
):
    log("")
    log("Запуск перестановочного теста...")

    # Независимо восстанавливаем наблюдаемый минимум.
    observed_best = select_best(
        observed_results
    )

    verified_best = verify_observed_results(
        observed_results,
        observed_best
    )

    observed_mae = float(
        verified_best["mae"]
    )

    observed_accuracy = float(
        verified_best["accuracy"]
    )

    log(
        f"Контрольный наблюдаемый MAE: "
        f"{observed_mae:.8f}"
    )

    log(
        f"Контрольная наблюдаемая точность: "
        f"{observed_accuracy:.8f}"
    )

    null_best_mae = []
    null_best_accuracy = []
    null_best_models = []

    for permutation_number in range(
        1,
        PERMUTATIONS + 1
    ):
        permuted_labels = random_generator.permutation(
            labels
        )

        permutation_results = evaluate_all(
            datasets,
            permuted_labels,
            folds,
            verbose=False
        )

        permutation_best = select_best(
            permutation_results
        )

        null_best_mae.append(
            permutation_best["mae"]
        )

        null_best_accuracy.append(
            permutation_best["accuracy"]
        )

        null_best_models.append(
            permutation_best["model"]
        )

        log(
            f"Перестановка {permutation_number}/"
            f"{PERMUTATIONS}: "
            f"лучший MAE={permutation_best['mae']:.6f}, "
            f"точность={permutation_best['accuracy']:.6f}, "
            f"модель={permutation_best['model']}"
        )

    null_best_mae = np.asarray(
        null_best_mae,
        dtype=float
    )

    null_best_accuracy = np.asarray(
        null_best_accuracy,
        dtype=float
    )

    p_mae = (
        1.0
        + np.sum(null_best_mae <= observed_mae)
    ) / (PERMUTATIONS + 1.0)

    p_accuracy = (
        1.0
        + np.sum(
            null_best_accuracy >= observed_accuracy
        )
    ) / (PERMUTATIONS + 1.0)

    statistics = {
        "observed_model": verified_best["model"],
        "observed_gamma": verified_best["gamma"],
        "observed_alpha": verified_best["alpha"],
        "observed_mae": observed_mae,
        "observed_accuracy": observed_accuracy,
        "null_mean_mae": float(
            np.mean(null_best_mae)
        ),
        "null_std_mae": float(
            np.std(null_best_mae)
        ),
        "null_min_mae": float(
            np.min(null_best_mae)
        ),
        "null_max_mae": float(
            np.max(null_best_mae)
        ),
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

    # Финальная проверка: наблюдаемый MAE не должен измениться.
    if not math.isclose(
        statistics["observed_mae"],
        observed_mae,
        rel_tol=0.0,
        abs_tol=1e-12
    ):
        raise RuntimeError(
            "Наблюдаемый MAE изменился во время теста"
        )

    return (
        statistics,
        null_best_mae,
        null_best_accuracy,
        null_best_models
    )


# ============================================================
# СОХРАНЕНИЕ ТАБЛИЦ
# ============================================================

def save_group_csv(path, group_points):
    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(
            file,
            delimiter=";"
        )

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
    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(
            file,
            delimiter=";"
        )

        writer.writerow([
            "model",
            "gamma",
            "alpha",
            "mae",
            "accuracy",
        ])

        sorted_results = sorted(
            results,
            key=lambda item: (
                item["mae"],
                -item["accuracy"]
            )
        )

        for result in sorted_results:
            writer.writerow([
                result["model"],
                result["gamma"],
                result["alpha"],
                f"{result['mae']:.10f}",
                f"{result['accuracy']:.10f}",
            ])


def save_predictions_csv(path, labels, best_result):
    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(
            file,
            delimiter=";"
        )

        writer.writerow([
            "true_scalar",
            "predicted_scalar",
            "circular_error",
            "is_correct",
        ])

        for true_value, predicted_value, error in zip(
            labels,
            best_result["predictions"],
            best_result["errors"]
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
    null_best_accuracy,
    null_best_models
):
    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(
            file,
            delimiter=";"
        )

        writer.writerow([
            "permutation",
            "best_mae",
            "best_accuracy",
            "best_model",
        ])

        for index in range(len(null_best_mae)):
            writer.writerow([
                index + 1,
                f"{null_best_mae[index]:.10f}",
                f"{null_best_accuracy[index]:.10f}",
                null_best_models[index],
            ])


def save_fold_metrics_csv(path, results):
    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(
            file,
            delimiter=";"
        )

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


def save_raw_features(path, datasets, labels):
    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(
            file,
            delimiter=";"
        )

        writer.writerow([
            "scalar",
            "model",
            "feature_index",
            "value",
        ])

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


def save_summary_csv(path, statistics):
    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(
            file,
            delimiter=";"
        )

        writer.writerow([
            "metric",
            "value",
        ])

        for key, value in statistics.items():
            writer.writerow([
                key,
                value,
            ])


# ============================================================
# ОТЧЁТ
# ============================================================

def build_configuration():
    return {
        "experiment": "C14_2",
        "title": "Исправленная геометрическая кривизна переходов",
        "field_p": FIELD_P,
        "curve_a": CURVE_A,
        "curve_b": CURVE_B,
        "generator": list(GENERATOR),
        "group_order": GROUP_ORDER,
        "offsets": OFFSETS,
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
    statistics,
    folds,
    elapsed
):
    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(
            "ЭКСПЕРИМЕНТ C14_2\n"
            "Исправленная геометрическая кривизна переходов\n"
        )

        file.write("=" * 72 + "\n\n")

        file.write("1. ПАРАМЕТРЫ КРИВОЙ\n")
        file.write(f"Поле: p = {FIELD_P}\n")
        file.write(
            f"Кривая: y^2 = x^3 + {CURVE_B} "
            f"(mod {FIELD_P})\n"
        )
        file.write(f"Генератор: G = {GENERATOR}\n")
        file.write(f"Порядок группы: {GROUP_ORDER}\n\n")

        file.write("2. ПАРАМЕТРЫ ЭКСПЕРИМЕНТА\n")
        file.write(f"Локальные смещения: {OFFSETS}\n")
        file.write(f"Радиус исключения: {PURGE_RADIUS}\n")
        file.write(f"Количество блоков: {FOLDS_COUNT}\n")
        file.write(f"Количество перестановок: {PERMUTATIONS}\n")
        file.write(f"Гамма: {GAMMAS}\n")
        file.write(f"Альфа: {ALPHAS}\n\n")

        file.write("3. БЛОЧНАЯ ПРОВЕРКА\n")

        for fold in folds:
            file.write(
                f"Блок {fold['number']}: "
                f"обучение={len(fold['train'])}, "
                f"тест={len(fold['test'])}\n"
            )

        file.write("\n4. РЕЗУЛЬТАТЫ КОНФИГУРАЦИЙ\n")

        sorted_results = sorted(
            results,
            key=lambda item: (
                item["mae"],
                -item["accuracy"]
            )
        )

        for index, result in enumerate(
            sorted_results,
            start=1
        ):
            file.write(
                f"{index:02d}. "
                f"{result['model']}, "
                f"гамма={result['gamma']}, "
                f"альфа={result['alpha']}, "
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

        for key, value in statistics.items():
            file.write(f"{key}: {value}\n")

        file.write("\n7. КОНТРОЛЬ СОГЛАСОВАННОСТИ\n")

        actual_minimum = min(
            result["mae"]
            for result in results
        )

        file.write(
            f"Минимум таблицы конфигураций: "
            f"{actual_minimum:.10f}\n"
        )

        file.write(
            f"Наблюдаемый MAE в тесте: "
            f"{statistics['observed_mae']:.10f}\n"
        )

        consistent = math.isclose(
            actual_minimum,
            statistics["observed_mae"],
            rel_tol=0.0,
            abs_tol=1e-12
        )

        file.write(
            f"Согласованность: "
            f"{'ПОДТВЕРЖДЕНА' if consistent else 'ОШИБКА'}\n"
        )

        file.write("\n8. ИНТЕРПРЕТАЦИЯ\n")

        if (
            statistics["p_value_mae"] < 0.05
            or statistics["p_value_accuracy"] < 0.05
        ):
            file.write(
                "Обнаружен результат, требующий дополнительной "
                "проверки: одно из перестановочных значений p "
                "меньше 0.05. Это не доказывает возможность "
                "эффективного решения ECDLP.\n"
            )
        else:
            file.write(
                "Статистически значимого преимущества над "
                "перестановочным контролем не обнаружено.\n"
            )

        file.write(
            "\nДаже положительный результат на игрушечной "
            "кривой не доказывает возможность эффективного "
            "решения ECDLP на криптографических кривых.\n"
        )

        file.write(
            f"\nВремя выполнения: {elapsed:.3f} секунд\n"
        )


# ============================================================
# СОЗДАНИЕ ZIP-АРХИВА
# ============================================================

def create_archive(results_directory, archive_path):
    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for root, _, filenames in os.walk(
            results_directory
        ):
            for filename in filenames:
                full_path = os.path.join(
                    root,
                    filename
                )

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
    global RESULTS_DIRECTORY
    global LOG_FILE

    start_time = time.time()
    run_stamp = timestamp()

    RESULTS_DIRECTORY = os.path.abspath(
        f"experimentC14_2_results_{run_stamp}"
    )

    ensure_directory(
        RESULTS_DIRECTORY
    )

    log_path = os.path.join(
        RESULTS_DIRECTORY,
        "C14_2_log.txt"
    )

    LOG_FILE = open(
        log_path,
        "w",
        encoding="utf-8"
    )

    log("=" * 72)
    log("ЭКСПЕРИМЕНТ C14_2")
    log("Исправленная геометрическая кривизна переходов")
    log("=" * 72)

    log(
        f"Каталог результатов: {RESULTS_DIRECTORY}"
    )

    random_generator = np.random.default_rng(
        RANDOM_SEED
    )

    configuration = build_configuration()

    configuration_path = os.path.join(
        RESULTS_DIRECTORY,
        "C14_2_configuration.json"
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

    datasets, labels = build_dataset(
        group_points
    )

    folds = build_folds()

    group_csv_path = os.path.join(
        RESULTS_DIRECTORY,
        "C14_2_group.csv"
    )

    save_group_csv(
        group_csv_path,
        group_points
    )

    log("")
    log("Оценка наблюдаемых данных...")
    log("")

    observed_results = evaluate_all(
        datasets,
        labels,
        folds,
        verbose=True
    )

    best_result = select_best(
        observed_results
    )

    # Независимая проверка лучшей конфигурации.
    verified_best = verify_observed_results(
        observed_results,
        best_result
    )

    log("")
    log("ЛУЧШАЯ НАБЛЮДАЕМАЯ КОНФИГУРАЦИЯ")
    log(f"Модель: {verified_best['model']}")
    log(f"Гамма: {verified_best['gamma']}")
    log(f"Альфа: {verified_best['alpha']}")
    log(f"MAE: {verified_best['mae']:.10f}")
    log(f"Точность: {verified_best['accuracy']:.10f}")

    log("")
    log("Повторная проверка предсказаний лучшей конфигурации...")

    prediction_result = evaluate_configuration(
        datasets[verified_best["model"]],
        labels,
        folds,
        verified_best["gamma"],
        verified_best["alpha"],
        collect_predictions=True
    )

    # Сверяем повторное вычисление с исходной таблицей.
    if not math.isclose(
        prediction_result["mae"],
        verified_best["mae"],
        rel_tol=0.0,
        abs_tol=1e-12
    ):
        raise RuntimeError(
            "Повторное вычисление предсказаний дало "
            "другой MAE"
        )

    if not math.isclose(
        prediction_result["accuracy"],
        verified_best["accuracy"],
        rel_tol=0.0,
        abs_tol=1e-12
    ):
        raise RuntimeError(
            "Повторное вычисление предсказаний дало "
            "другую точность"
        )

    best_result = dict(verified_best)

    best_result["predictions"] = (
        prediction_result["predictions"]
    )

    best_result["errors"] = (
        prediction_result["errors"]
    )

    best_result["fold_metrics"] = (
        prediction_result["fold_metrics"]
    )

    log("Повторная проверка предсказаний пройдена.")

    log("")
    log("Проверка перед перестановочным тестом...")

    verify_observed_results(
        observed_results,
        best_result
    )

    (
        permutation_statistics,
        null_best_mae,
        null_best_accuracy,
        null_best_models
    ) = permutation_test(
        datasets,
        labels,
        folds,
        observed_results,
        random_generator
    )

    # Сохраняем все результаты.
    save_results_csv(
        os.path.join(
            RESULTS_DIRECTORY,
            "C14_2_results.csv"
        ),
        observed_results
    )

    save_predictions_csv(
        os.path.join(
            RESULTS_DIRECTORY,
            "C14_2_predictions.csv"
        ),
        labels,
        best_result
    )

    save_permutation_csv(
        os.path.join(
            RESULTS_DIRECTORY,
            "C14_2_permutations.csv"
        ),
        null_best_mae,
        null_best_accuracy,
        null_best_models
    )

    save_fold_metrics_csv(
        os.path.join(
            RESULTS_DIRECTORY,
            "C14_2_fold_metrics.csv"
        ),
        observed_results
    )

    save_raw_features(
        os.path.join(
            RESULTS_DIRECTORY,
            "C14_2_features.csv"
        ),
        datasets,
        labels
    )

    save_summary_csv(
        os.path.join(
            RESULTS_DIRECTORY,
            "C14_2_summary.csv"
        ),
        permutation_statistics
    )

    elapsed = time.time() - start_time

    write_report(
        os.path.join(
            RESULTS_DIRECTORY,
            "C14_2_report.txt"
        ),
        observed_results,
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
        f"{permutation_statistics['observed_mae']:.10f}"
    )

    log(
        f"Средний нулевой MAE: "
        f"{permutation_statistics['null_mean_mae']:.10f}"
    )

    log(
        f"Минимальный нулевой MAE: "
        f"{permutation_statistics['null_min_mae']:.10f}"
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
    log("КОНТРОЛЬ СОГЛАСОВАННОСТИ")

    actual_minimum = min(
        result["mae"]
        for result in observed_results
    )

    log(
        f"Минимум среди 42 конфигураций: "
        f"{actual_minimum:.10f}"
    )

    log(
        f"Наблюдаемый MAE теста: "
        f"{permutation_statistics['observed_mae']:.10f}"
    )

    if not math.isclose(
        actual_minimum,
        permutation_statistics["observed_mae"],
        rel_tol=0.0,
        abs_tol=1e-12
    ):
        raise RuntimeError(
            "Финальная проверка согласованности не пройдена"
        )

    log("Согласованность подтверждена.")

    elapsed = time.time() - start_time

    log(f"Время выполнения: {elapsed:.2f} секунд")

    archive_path = os.path.abspath(
        f"experimentC14_2_{run_stamp}.zip"
    )

    log("")
    log("Подготовка ZIP-архива...")

    # Закрываем журнал перед архивированием.
    LOG_FILE.close()
    LOG_FILE = None

    create_archive(
        RESULTS_DIRECTORY,
        archive_path
    )

    print("")
    print("=" * 72)
    print("ЭКСПЕРИМЕНТ C14_2 ЗАВЕРШЁН")
    print(f"Каталог результатов: {RESULTS_DIRECTORY}")
    print(f"ZIP-архив: {archive_path}")
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()

    except Exception:
        error_text = traceback.format_exc()

        print("")
        print("ОШИБКА ВО ВРЕМЯ ВЫПОЛНЕНИЯ ЭКСПЕРИМЕНТА")
        print(error_text)

        if LOG_FILE is not None:
            LOG_FILE.write("\nОШИБКА ВО ВРЕМЯ ВЫПОЛНЕНИЯ\n")
            LOG_FILE.write(error_text)
            LOG_FILE.close()
            LOG_FILE = None

        raise
