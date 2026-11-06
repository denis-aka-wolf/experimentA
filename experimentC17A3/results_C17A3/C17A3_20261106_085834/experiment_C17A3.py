
# -*- coding: utf-8 -*-

"""
Эксперимент C17A3.
Исследование геометрических зависимостей скалярной разности
на игрушечной эллиптической кривой y^2 = x^3 + 7 (mod 499).

Скрипт:
1. Строит все ненулевые точки циклической подгруппы.
2. Создает все упорядоченные пары различных точек.
3. Исследует геометрические признаки и круговую скалярную разность.
4. Проводит проверку с разделением по точкам.
5. Проводит перестановочный тест на фиксированной выборке.
6. Сохраняет CSV, JSON, TXT, исходный код и ZIP-архив.

Скалярные значения используются только как целевые метки.
Операция сложения точек не используется при предсказании.
"""

import os
import json
import math
import shutil
import zipfile
import random
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor


# ============================================================
# НАСТРОЙКИ ЭКСПЕРИМЕНТА
# ============================================================

EXPERIMENT_NAME = "C17A3"

FIELD_P = 499
CURVE_A = 0
CURVE_B = 7

GENERATOR_X = 3
GENERATOR_Y = 212

GROUP_ORDER = 73
POINT_COUNT = GROUP_ORDER - 1

RANDOM_SEED = 17303

REPEAT_COUNT = 10
TEST_POINT_COUNT = 18

PERMUTATION_COUNT = 199

TREE_COUNT = 100
MAX_DEPTH = None
MIN_SAMPLES_LEAF = 2

N_JOBS = -1

OUTPUT_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results_C17A3"
)


# ============================================================
# ОБЩИЕ ФУНКЦИИ
# ============================================================

def print_message(message):
    print(message, flush=True)


def ensure_directory(path):
    os.makedirs(path, exist_ok=True)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def modular_inverse(value, modulus):
    return pow(int(value) % modulus, -1, modulus)


# ============================================================
# АРИФМЕТИКА ЭЛЛИПТИЧЕСКОЙ КРИВОЙ
# ============================================================

def point_add(point_a, point_b):
    if point_a is None:
        return point_b

    if point_b is None:
        return point_a

    x1, y1 = point_a
    x2, y2 = point_b

    if x1 == x2 and (y1 + y2) % FIELD_P == 0:
        return None

    if point_a == point_b:
        if y1 % FIELD_P == 0:
            return None

        numerator = (
            3 * x1 * x1 + CURVE_A
        ) % FIELD_P

        denominator = (
            2 * y1
        ) % FIELD_P

    else:
        numerator = (
            y2 - y1
        ) % FIELD_P

        denominator = (
            x2 - x1
        ) % FIELD_P

    slope = (
        numerator * modular_inverse(denominator, FIELD_P)
    ) % FIELD_P

    x3 = (
        slope * slope - x1 - x2
    ) % FIELD_P

    y3 = (
        slope * (x1 - x3) - y1
    ) % FIELD_P

    return int(x3), int(y3)


def scalar_multiply(scalar, point):
    result = None
    current = point
    value = int(scalar)

    while value > 0:
        if value & 1:
            result = point_add(result, current)

        current = point_add(current, current)
        value >>= 1

    return result


def generate_points():
    generator = (GENERATOR_X, GENERATOR_Y)

    points = []

    for scalar in range(1, GROUP_ORDER):
        point = scalar_multiply(scalar, generator)

        if point is None:
            raise RuntimeError(
                "Получена бесконечно удаленная точка раньше ожидаемого."
            )

        points.append({
            "point_id": scalar,
            "scalar": scalar,
            "x": point[0],
            "y": point[1]
        })

    unique_points = {
        (item["x"], item["y"])
        for item in points
    }

    if len(unique_points) != POINT_COUNT:
        raise RuntimeError(
            "Количество уникальных точек не совпадает с ожидаемым."
        )

    if scalar_multiply(GROUP_ORDER, generator) is not None:
        raise RuntimeError(
            "Порядок генератора не совпадает с настройкой GROUP_ORDER."
        )

    return points


# ============================================================
# СОЗДАНИЕ УПОРЯДОЧЕННЫХ ПАР
# ============================================================

def generate_pairs(points, scalar_assignment=None):
    if scalar_assignment is None:
        scalar_assignment = {
            item["point_id"]: item["scalar"]
            for item in points
        }

    records = []

    for point_a in points:
        for point_b in points:
            if point_a["point_id"] == point_b["point_id"]:
                continue

            scalar_a = scalar_assignment[point_a["point_id"]]
            scalar_b = scalar_assignment[point_b["point_id"]]

            delta_scalar = (
                scalar_b - scalar_a
            ) % GROUP_ORDER

            records.append({
                "point_a": point_a["point_id"],
                "point_b": point_b["point_id"],
                "xa": point_a["x"],
                "ya": point_a["y"],
                "xb": point_b["x"],
                "yb": point_b["y"],
                "scalar_a": scalar_a,
                "scalar_b": scalar_b,
                "delta_scalar": delta_scalar
            })

    return pd.DataFrame(records)


# ============================================================
# ГЕОМЕТРИЧЕСКИЕ ПРИЗНАКИ
# ============================================================

FEATURE_NAMES = {
    "xy_control": [
        "xa", "ya", "xb", "yb"
    ],
    "displacement": [
        "dx", "dy"
    ],
    "midpoint": [
        "mx", "my"
    ],
    "polar": [
        "distance", "direction_x", "direction_y"
    ],
    "displacement_midpoint": [
        "dx", "dy", "mx", "my"
    ],
    "combined": [
        "xa", "ya", "xb", "yb",
        "dx", "dy",
        "mx", "my",
        "distance",
        "direction_x",
        "direction_y"
    ]
}


FEATURE_FAMILIES = list(FEATURE_NAMES.keys())


def add_geometric_columns(pairs):
    result = pairs.copy()

    result["dx"] = result["xb"] - result["xa"]
    result["dy"] = result["yb"] - result["ya"]

    result["mx"] = (
        result["xa"] + result["xb"]
    ) / 2.0

    result["my"] = (
        result["ya"] + result["yb"]
    ) / 2.0

    result["distance"] = np.sqrt(
        result["dx"] ** 2 + result["dy"] ** 2
    )

    safe_distance = result["distance"].replace(0, np.nan)

    result["direction_x"] = (
        result["dx"] / safe_distance
    )

    result["direction_y"] = (
        result["dy"] / safe_distance
    )

    result["direction_x"] = result["direction_x"].fillna(0)
    result["direction_y"] = result["direction_y"].fillna(0)

    result["angle"] = np.arctan2(
        result["dy"],
        result["dx"]
    )

    return result


def get_features(data, family):
    columns = FEATURE_NAMES[family]
    return data[columns].to_numpy(dtype=float)


# ============================================================
# КРУГОВАЯ ЦЕЛЕВАЯ ПЕРЕМЕННАЯ
# ============================================================

def encode_circular_target(delta_values):
    angles = (
        2.0 * np.pi * np.asarray(delta_values, dtype=float)
        / GROUP_ORDER
    )

    return np.column_stack([
        np.cos(angles),
        np.sin(angles)
    ])


def decode_circular_prediction(predictions):
    angles = np.arctan2(
        predictions[:, 1],
        predictions[:, 0]
    )

    angles = np.mod(angles, 2.0 * np.pi)

    values = (
        angles * GROUP_ORDER / (2.0 * np.pi)
    )

    return np.mod(
        np.rint(values).astype(int),
        GROUP_ORDER
    )


def circular_error(predicted, actual):
    difference = np.abs(
        np.asarray(predicted, dtype=int)
        - np.asarray(actual, dtype=int)
    )

    return np.minimum(
        difference,
        GROUP_ORDER - difference
    )


def calculate_metrics(predicted, actual):
    errors = circular_error(predicted, actual)

    return {
        "cyclic_mae": float(np.mean(errors)),
        "cyclic_median_error": float(np.median(errors)),
        "exact_accuracy": float(np.mean(errors == 0)),
        "within_1_accuracy": float(np.mean(errors <= 1)),
        "within_3_accuracy": float(np.mean(errors <= 3)),
        "max_error": int(np.max(errors))
    }


# ============================================================
# ОБУЧЕНИЕ И АНТИСИММЕТРИЧНОЕ ПРЕДСКАЗАНИЕ
# ============================================================

def create_model(seed):
    return ExtraTreesRegressor(
        n_estimators=TREE_COUNT,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=int(seed),
        n_jobs=N_JOBS
    )


def predict_symmetrically(model, test_data, family):
    forward_features = get_features(
        test_data,
        family
    )

    reverse_data = test_data.copy()

    reverse_data[["xa", "xb"]] = (
        test_data[["xb", "xa"]].to_numpy()
    )

    reverse_data[["ya", "yb"]] = (
        test_data[["yb", "ya"]].to_numpy()
    )

    reverse_data = add_geometric_columns(reverse_data)

    reverse_features = get_features(
        reverse_data,
        family
    )

    prediction_forward = model.predict(
        forward_features
    )

    prediction_reverse = model.predict(
        reverse_features
    )

    # Антисимметризация кругового вектора.
    prediction = (
        prediction_forward - prediction_reverse
    ) / 2.0

    norms = np.linalg.norm(
        prediction,
        axis=1,
        keepdims=True
    )

    norms[norms == 0] = 1.0

    prediction = prediction / norms

    return decode_circular_prediction(prediction)


def fit_and_predict(train_data, test_data, family, seed):
    model = create_model(seed)

    x_train = get_features(
        train_data,
        family
    )

    y_train = encode_circular_target(
        train_data["delta_scalar"].to_numpy()
    )

    model.fit(x_train, y_train)

    prediction = predict_symmetrically(
        model,
        test_data,
        family
    )

    importance = {}

    for name, value in zip(
        FEATURE_NAMES[family],
        model.feature_importances_
    ):
        importance[name] = float(value)

    return prediction, importance


# ============================================================
# РАЗДЕЛЕНИЕ ПО ТОЧКАМ
# ============================================================

def split_point_ids(points, test_count, seed):
    generator = np.random.default_rng(seed)

    point_ids = np.array([
        item["point_id"]
        for item in points
    ])

    shuffled = generator.permutation(point_ids)

    test_ids = set(
        int(value)
        for value in shuffled[:test_count]
    )

    train_ids = set(
        int(value)
        for value in shuffled[test_count:]
    )

    return train_ids, test_ids


def select_internal_pairs(pairs, point_ids):
    mask = (
        pairs["point_a"].isin(point_ids)
        & pairs["point_b"].isin(point_ids)
    )

    return pairs.loc[mask].copy().reset_index(drop=True)


def validate_disjoint_split(train_data, test_data):
    train_points = (
        set(train_data["point_a"])
        | set(train_data["point_b"])
    )

    test_points = (
        set(test_data["point_a"])
        | set(test_data["point_b"])
    )

    if train_points.intersection(test_points):
        raise RuntimeError(
            "Обнаружена утечка точек между обучением и тестом."
        )


# ============================================================
# ПОВТОРЯЕМАЯ ПРОВЕРКА
# ============================================================

def run_repeated_validation(points, pairs):
    metric_records = []
    importance_records = []

    for repeat_index in range(REPEAT_COUNT):
        seed = RANDOM_SEED + repeat_index

        train_ids, test_ids = split_point_ids(
            points,
            TEST_POINT_COUNT,
            seed
        )

        train_data = select_internal_pairs(
            pairs,
            train_ids
        )

        test_data = select_internal_pairs(
            pairs,
            test_ids
        )

        validate_disjoint_split(
            train_data,
            test_data
        )

        for family_index, family in enumerate(FEATURE_FAMILIES):
            model_seed = (
                RANDOM_SEED
                + repeat_index * 100
                + family_index
            )

            prediction, importance = fit_and_predict(
                train_data,
                test_data,
                family,
                model_seed
            )

            actual = test_data[
                "delta_scalar"
            ].to_numpy()

            metrics = calculate_metrics(
                prediction,
                actual
            )

            record = {
                "repeat": repeat_index + 1,
                "family": family,
                "train_point_count": len(train_ids),
                "test_point_count": len(test_ids),
                "train_pair_count": len(train_data),
                "test_pair_count": len(test_data),
                **metrics
            }

            metric_records.append(record)

            for feature_name, feature_importance in importance.items():
                importance_records.append({
                    "repeat": repeat_index + 1,
                    "family": family,
                    "feature": feature_name,
                    "importance": feature_importance
                })

        print_message(
            f"Повторная проверка: {repeat_index + 1}/{REPEAT_COUNT}"
        )

    return (
        pd.DataFrame(metric_records),
        pd.DataFrame(importance_records)
    )


# ============================================================
# АНАЛИЗ СВЯЗЕЙ ПРИЗНАКОВ С КРУГОВОЙ ЦЕЛЬЮ
# ============================================================

def calculate_feature_correlations(pairs):
    feature_columns = [
        "dx", "dy",
        "mx", "my",
        "distance",
        "direction_x", "direction_y",
        "angle",
        "xa", "ya", "xb", "yb"
    ]

    target = encode_circular_target(
        pairs["delta_scalar"].to_numpy()
    )

    data = pairs.copy()

    data["target_cos"] = target[:, 0]
    data["target_sin"] = target[:, 1]

    records = []

    for feature in feature_columns:
        for target_name in ["target_cos", "target_sin"]:
            pearson_value = data[feature].corr(
                data[target_name],
                method="pearson"
            )

            spearman_value = data[feature].corr(
                data[target_name],
                method="spearman"
            )

            records.append({
                "feature": feature,
                "target_component": target_name,
                "pearson": float(pearson_value),
                "spearman": float(spearman_value)
            })

    return pd.DataFrame(records)


def calculate_binned_statistics(pairs):
    records = []

    columns = [
        "dx", "dy",
        "mx", "my",
        "distance"
    ]

    for column in columns:
        try:
            bins = pd.qcut(
                pairs[column],
                q=8,
                duplicates="drop"
            )
        except ValueError:
            continue

        temporary = pairs.copy()
        temporary["bin"] = bins

        for bin_value, group in temporary.groupby(
            "bin",
            observed=True
        ):
            target = encode_circular_target(
                group["delta_scalar"].to_numpy()
            )

            mean_cos = float(np.mean(target[:, 0]))
            mean_sin = float(np.mean(target[:, 1]))

            concentration = math.sqrt(
                mean_cos ** 2 + mean_sin ** 2
            )

            mean_angle = math.atan2(
                mean_sin,
                mean_cos
            )

            mean_delta = (
                mean_angle * GROUP_ORDER
                / (2.0 * math.pi)
            ) % GROUP_ORDER

            records.append({
                "feature": column,
                "bin": str(bin_value),
                "count": int(len(group)),
                "mean_cos": mean_cos,
                "mean_sin": mean_sin,
                "circular_concentration": concentration,
                "circular_mean_delta": mean_delta
            })

    return pd.DataFrame(records)


# ============================================================
# ФИКСИРОВАННЫЙ HOLDOUT И ПЕРЕСТАНОВОЧНЫЙ ТЕСТ
# ============================================================

def make_scalar_assignment(points, permutation=None):
    point_ids = [
        item["point_id"]
        for item in points
    ]

    scalar_values = np.arange(
        1,
        GROUP_ORDER,
        dtype=int
    )

    if permutation is not None:
        scalar_values = np.asarray(
            permutation,
            dtype=int
        )

    return {
        int(point_id): int(scalar)
        for point_id, scalar in zip(
            point_ids,
            scalar_values
        )
    }


def evaluate_fixed_holdout(points, train_ids, test_ids, assignment, seed):
    all_pairs = generate_pairs(
        points,
        scalar_assignment=assignment
    )

    all_pairs = add_geometric_columns(all_pairs)

    train_data = select_internal_pairs(
        all_pairs,
        train_ids
    )

    test_data = select_internal_pairs(
        all_pairs,
        test_ids
    )

    validate_disjoint_split(
        train_data,
        test_data
    )

    predictions = {}
    metrics = {}

    for family in ["xy_control", "displacement"]:
        prediction, _ = fit_and_predict(
            train_data,
            test_data,
            family,
            seed
        )

        predictions[family] = prediction

        metrics[family] = calculate_metrics(
            prediction,
            test_data["delta_scalar"].to_numpy()
        )

    improvement = (
        metrics["xy_control"]["cyclic_mae"]
        - metrics["displacement"]["cyclic_mae"]
    )

    return {
        "metrics": metrics,
        "improvement": float(improvement),
        "test_data": test_data,
        "predictions": predictions
    }


def run_permutation_test(points, pairs):
    train_ids, test_ids = split_point_ids(
        points,
        TEST_POINT_COUNT,
        RANDOM_SEED + 9000
    )

    observed_assignment = make_scalar_assignment(
        points
    )

    observed = evaluate_fixed_holdout(
        points,
        train_ids,
        test_ids,
        observed_assignment,
        RANDOM_SEED + 9100
    )

    observed_xy_error = (
        observed["metrics"]["xy_control"]["cyclic_mae"]
    )

    observed_displacement_error = (
        observed["metrics"]["displacement"]["cyclic_mae"]
    )

    observed_improvement = observed["improvement"]

    permutation_records = []

    base_scalars = np.arange(
        1,
        GROUP_ORDER,
        dtype=int
    )

    for permutation_index in range(PERMUTATION_COUNT):
        generator = np.random.default_rng(
            RANDOM_SEED + 10000 + permutation_index
        )

        shuffled_scalars = generator.permutation(
            base_scalars
        )

        assignment = make_scalar_assignment(
            points,
            shuffled_scalars
        )

        result = evaluate_fixed_holdout(
            points,
            train_ids,
            test_ids,
            assignment,
            RANDOM_SEED + 20000 + permutation_index
        )

        xy_error = (
            result["metrics"]["xy_control"]["cyclic_mae"]
        )

        displacement_error = (
            result["metrics"]["displacement"]["cyclic_mae"]
        )

        permutation_records.append({
            "permutation": permutation_index + 1,
            "xy_control_mae": xy_error,
            "displacement_mae": displacement_error,
            "improvement": result["improvement"]
        })

        if (
            (permutation_index + 1) % 10 == 0
            or permutation_index + 1 == PERMUTATION_COUNT
        ):
            print_message(
                "Перестановочный тест: "
                f"{permutation_index + 1}/{PERMUTATION_COUNT}"
            )

    permutation_frame = pd.DataFrame(
        permutation_records
    )

    null_displacement = permutation_frame[
        "displacement_mae"
    ].to_numpy()

    null_improvement = permutation_frame[
        "improvement"
    ].to_numpy()

    # Малое значение ошибки считается более сильным результатом.
    p_displacement = (
        1 + np.sum(
            null_displacement <= observed_displacement_error
        )
    ) / (PERMUTATION_COUNT + 1)

    # Большое улучшение относительно контроля считается более сильным.
    p_improvement = (
        1 + np.sum(
            null_improvement >= observed_improvement
        )
    ) / (PERMUTATION_COUNT + 1)

    summary = {
        "train_point_count": len(train_ids),
        "test_point_count": len(test_ids),
        "train_pair_count": int(
            len(select_internal_pairs(pairs, train_ids))
        ),
        "test_pair_count": int(
            len(select_internal_pairs(pairs, test_ids))
        ),
        "observed_xy_control_mae": float(observed_xy_error),
        "observed_displacement_mae": float(
            observed_displacement_error
        ),
        "observed_improvement": float(
            observed_improvement
        ),
        "permutation_count": PERMUTATION_COUNT,
        "null_displacement_mean": float(
            np.mean(null_displacement)
        ),
        "null_displacement_std": float(
            np.std(null_displacement, ddof=1)
        ),
        "null_displacement_min": float(
            np.min(null_displacement)
        ),
        "null_displacement_max": float(
            np.max(null_displacement)
        ),
        "null_improvement_mean": float(
            np.mean(null_improvement)
        ),
        "null_improvement_std": float(
            np.std(null_improvement, ddof=1)
        ),
        "p_value_displacement": float(p_displacement),
        "p_value_improvement": float(p_improvement)
    }

    prediction_records = []

    for index, (_, row) in enumerate(
        observed["test_data"].iterrows()
    ):
        prediction_records.append({
            "point_a": int(row["point_a"]),
            "point_b": int(row["point_b"]),
            "actual_delta": int(row["delta_scalar"]),
            "xy_control_prediction": int(
                observed["predictions"]["xy_control"][index]
            ),
            "displacement_prediction": int(
                observed["predictions"]["displacement"][index]
            )
        })

    return (
        permutation_frame,
        summary,
        pd.DataFrame(prediction_records)
    )


# ============================================================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ============================================================

def build_report(
    points,
    pairs,
    repeat_metrics,
    importance,
    correlations,
    binned_statistics,
    permutation_summary
):
    lines = []

    lines.append("ОТЧЕТ ЭКСПЕРИМЕНТА C17A3")
    lines.append("=" * 70)
    lines.append("")

    lines.append("ПАРАМЕТРЫ КРИВОЙ")
    lines.append(f"Поле: {FIELD_P}")
    lines.append(f"Кривая: y^2 = x^3 + {CURVE_B} (mod {FIELD_P})")
    lines.append(
        f"Генератор: ({GENERATOR_X}, {GENERATOR_Y})"
    )
    lines.append(f"Порядок подгруппы: {GROUP_ORDER}")
    lines.append(f"Количество ненулевых точек: {len(points)}")
    lines.append(f"Количество упорядоченных пар: {len(pairs)}")
    lines.append("")

    lines.append("ПОВТОРЯЕМАЯ ПРОВЕРКА ПО ТОЧКАМ")
    lines.append("-" * 70)

    summary = repeat_metrics.groupby(
        "family"
    ).agg({
        "cyclic_mae": ["mean", "std", "min", "max"],
        "exact_accuracy": ["mean", "std"],
        "within_1_accuracy": ["mean", "std"],
        "within_3_accuracy": ["mean", "std"]
    })

    lines.append(summary.to_string())
    lines.append("")

    lines.append("ПЕРЕСТАНОВОЧНЫЙ ТЕСТ")
    lines.append("-" * 70)

    for key, value in permutation_summary.items():
        lines.append(f"{key}: {value}")

    lines.append("")

    lines.append("КОРРЕЛЯЦИИ")
    lines.append("-" * 70)

    lines.append(
        correlations.to_string(index=False)
    )

    lines.append("")

    lines.append("ВАЖНОЕ МЕТОДОЛОГИЧЕСКОЕ ОГРАНИЧЕНИЕ")
    lines.append("-" * 70)
    lines.append(
        "Геометрические признаки являются преобразованиями "
        "исходных координат."
    )
    lines.append(
        "Улучшение модели может означать более удобное представление "
        "тех же данных, а не появление новой информации."
    )
    lines.append(
        "Перестановочный тест проверяет связь назначенных скаляров "
        "с геометрией в рамках данной игрушечной кривой."
    )
    lines.append(
        "Результат не является решением задачи дискретного логарифма "
        "и не доказывает применимость к secp256k1."
    )

    return "\n".join(lines)


def save_all_results(
    output_directory,
    points,
    pairs,
    repeat_metrics,
    importance,
    correlations,
    binned_statistics,
    permutation_frame,
    permutation_summary,
    holdout_predictions
):
    ensure_directory(output_directory)

    pd.DataFrame(points).to_csv(
        os.path.join(output_directory, "points.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    pairs.to_csv(
        os.path.join(output_directory, "all_pairs.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    repeat_metrics.to_csv(
        os.path.join(output_directory, "repeat_metrics.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    importance.to_csv(
        os.path.join(output_directory, "feature_importance.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    correlations.to_csv(
        os.path.join(output_directory, "feature_correlations.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    binned_statistics.to_csv(
        os.path.join(output_directory, "binned_statistics.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    permutation_frame.to_csv(
        os.path.join(output_directory, "permutation_results.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    holdout_predictions.to_csv(
        os.path.join(output_directory, "holdout_predictions.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    save_json(
        permutation_summary,
        os.path.join(output_directory, "permutation_summary.json")
    )

    config = {
        "experiment": EXPERIMENT_NAME,
        "field_p": FIELD_P,
        "curve_a": CURVE_A,
        "curve_b": CURVE_B,
        "generator": [GENERATOR_X, GENERATOR_Y],
        "group_order": GROUP_ORDER,
        "point_count": POINT_COUNT,
        "repeat_count": REPEAT_COUNT,
        "test_point_count": TEST_POINT_COUNT,
        "permutation_count": PERMUTATION_COUNT,
        "tree_count": TREE_COUNT,
        "max_depth": MAX_DEPTH,
        "min_samples_leaf": MIN_SAMPLES_LEAF,
        "random_seed": RANDOM_SEED,
        "feature_families": FEATURE_NAMES,
        "target_encoding": "cos(2*pi*delta/73), sin(2*pi*delta/73)",
        "prediction_method": "antisymmetric forward-reverse averaging"
    }

    save_json(
        config,
        os.path.join(output_directory, "config.json")
    )

    report = build_report(
        points,
        pairs,
        repeat_metrics,
        importance,
        correlations,
        binned_statistics,
        permutation_summary
    )

    with open(
        os.path.join(output_directory, "report.txt"),
        "w",
        encoding="utf-8"
    ) as file:
        file.write(report)

    return report


def create_archive(output_directory):
    archive_path = output_directory + ".zip"

    if os.path.exists(archive_path):
        os.remove(archive_path)

    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for root, _, files in os.walk(output_directory):
            for filename in files:
                full_path = os.path.join(root, filename)

                relative_path = os.path.relpath(
                    full_path,
                    os.path.dirname(output_directory)
                )

                archive.write(
                    full_path,
                    arcname=relative_path
                )

    return archive_path


# ============================================================
# ЗАПУСК
# ============================================================

def main():
    start_time = datetime.now()

    timestamp = start_time.strftime("%Y%m%d_%H%M%S")

    output_directory = os.path.join(
        OUTPUT_ROOT,
        f"{EXPERIMENT_NAME}_{timestamp}"
    )

    ensure_directory(output_directory)

    print_message("=" * 65)
    print_message("ЗАПУСК ЭКСПЕРИМЕНТА C17A3")
    print_message("=" * 65)

    print_message("Построение точек кривой...")

    points = generate_points()

    print_message(
        f"Построено точек: {len(points)}"
    )

    print_message("Создание упорядоченных пар...")

    pairs = generate_pairs(points)
    pairs = add_geometric_columns(pairs)

    print_message(
        f"Создано пар: {len(pairs)}"
    )

    print_message("Расчет корреляций...")

    correlations = calculate_feature_correlations(
        pairs
    )

    print_message("Расчет круговой статистики по интервалам...")

    binned_statistics = calculate_binned_statistics(
        pairs
    )

    print_message("Повторная проверка с разделением по точкам...")

    repeat_metrics, importance = run_repeated_validation(
        points,
        pairs
    )

    print_message("Запуск фиксированного перестановочного теста...")

    (
        permutation_frame,
        permutation_summary,
        holdout_predictions
    ) = run_permutation_test(
        points,
        pairs
    )

    print_message("Сохранение результатов...")

    report = save_all_results(
        output_directory,
        points,
        pairs,
        repeat_metrics,
        importance,
        correlations,
        binned_statistics,
        permutation_frame,
        permutation_summary,
        holdout_predictions
    )

    source_path = os.path.abspath(__file__)

    shutil.copy2(
        source_path,
        os.path.join(
            output_directory,
            "experiment_C17A3.py"
        )
    )

    archive_path = create_archive(
        output_directory
    )

    elapsed = (
        datetime.now() - start_time
    ).total_seconds()

    print_message("")
    print_message("=" * 65)
    print_message("ЭКСПЕРИМЕНТ C17A3 ЗАВЕРШЕН")
    print_message("=" * 65)

    print_message(
        f"Папка результатов: {output_directory}"
    )

    print_message(
        f"ZIP-архив: {archive_path}"
    )

    print_message(
        f"Время выполнения: {elapsed:.2f} сек."
    )

    print_message("")
    print_message("КРАТКИЙ ОТЧЕТ")

    summary = repeat_metrics.groupby(
        "family"
    )["cyclic_mae"].agg(
        ["mean", "std"]
    ).sort_values("mean")

    print(summary.to_string())

    print_message("")
    print_message(
        "Перестановочный тест:"
    )

    print_message(
        f"MAE исходных координат: "
        f"{permutation_summary['observed_xy_control_mae']:.6f}"
    )

    print_message(
        f"MAE смещения: "
        f"{permutation_summary['observed_displacement_mae']:.6f}"
    )

    print_message(
        f"Улучшение: "
        f"{permutation_summary['observed_improvement']:.6f}"
    )

    print_message(
        f"p-value смещения: "
        f"{permutation_summary['p_value_displacement']:.6f}"
    )

    print_message(
        f"p-value улучшения: "
        f"{permutation_summary['p_value_improvement']:.6f}"
    )

    print_message("")
    print_message("Полный отчет сохранен в report.txt")
    print_message("Архив содержит все таблицы и исходный код.")


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print_message("")
        print_message("ОШИБКА ВО ВРЕМЯ ЭКСПЕРИМЕНТА")
        print_message(str(error))
        print_message(traceback.format_exc())
        raise
