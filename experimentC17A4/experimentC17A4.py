
# -*- coding: utf-8 -*-

"""
Эксперимент C17A4.
Строгая проверка полярного геометрического представления.

Кривая:
y^2 = x^3 + 7 (mod 499)

Генератор:
G = (3, 212)

Порядок подгруппы:
73

Сравниваются два представления:
1. xy_control: исходные координаты двух точек.
2. polar: расстояние и нормированное направление между точками.

Целевая переменная:
delta = (scalar_b - scalar_a) mod 73.

Используется круговое кодирование цели через синус и косинус.
Предсказание симметризуется по направлению пары.

Скрипт проводит повторную проверку по точкам и перестановочный тест.
Каждая перестановка меняет назначение скаляров точкам целиком,
сохраняя зависимость между парами, содержащими общие точки.

Скрипт сохраняет таблицы, отчет, настройки, исходный код и ZIP-архив.
"""

import os
import json
import shutil
import zipfile
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor


# ============================================================
# НАСТРОЙКИ
# ============================================================

EXPERIMENT_NAME = "C17A4"

FIELD_P = 499
CURVE_A = 0
CURVE_B = 7

GENERATOR_X = 3
GENERATOR_Y = 212

GROUP_ORDER = 73
POINT_COUNT = GROUP_ORDER - 1

RANDOM_SEED = 17404

REPEAT_COUNT = 10
TEST_POINT_COUNT = 18

PERMUTATION_COUNT = 999
PERMUTATION_PROGRESS_STEP = 25

TREE_COUNT = 100
MAX_DEPTH = None
MIN_SAMPLES_LEAF = 2
N_JOBS = -1

OUTPUT_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results_C17A4"
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
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2
        )


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
    generator = (
        GENERATOR_X,
        GENERATOR_Y
    )

    points = []

    for scalar in range(1, GROUP_ORDER):
        point = scalar_multiply(
            scalar,
            generator
        )

        if point is None:
            raise RuntimeError(
                "Получена бесконечно удаленная точка."
            )

        points.append({
            "point_id": scalar,
            "original_scalar": scalar,
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

    if scalar_multiply(
        GROUP_ORDER,
        generator
    ) is not None:
        raise RuntimeError(
            "Порядок генератора не совпадает с настройками."
        )

    return points


# ============================================================
# СОЗДАНИЕ ПАР И ГЕОМЕТРИЧЕСКИХ ПРИЗНАКОВ
# ============================================================

def generate_pairs(points, scalar_assignment):
    records = []

    for point_a in points:
        for point_b in points:
            if point_a["point_id"] == point_b["point_id"]:
                continue

            scalar_a = scalar_assignment[
                point_a["point_id"]
            ]

            scalar_b = scalar_assignment[
                point_b["point_id"]
            ]

            delta_scalar = (
                scalar_b - scalar_a
            ) % GROUP_ORDER

            xa = point_a["x"]
            ya = point_a["y"]
            xb = point_b["x"]
            yb = point_b["y"]

            dx = xb - xa
            dy = yb - ya

            distance = float(
                np.sqrt(dx * dx + dy * dy)
            )

            if distance == 0:
                direction_x = 0.0
                direction_y = 0.0
            else:
                direction_x = dx / distance
                direction_y = dy / distance

            records.append({
                "point_a": point_a["point_id"],
                "point_b": point_b["point_id"],
                "xa": xa,
                "ya": ya,
                "xb": xb,
                "yb": yb,
                "dx": dx,
                "dy": dy,
                "distance": distance,
                "direction_x": direction_x,
                "direction_y": direction_y,
                "scalar_a": scalar_a,
                "scalar_b": scalar_b,
                "delta_scalar": delta_scalar
            })

    return pd.DataFrame(records)


# ============================================================
# ПРИЗНАКИ И КРУГОВАЯ ЦЕЛЬ
# ============================================================

FEATURE_NAMES = {
    "xy_control": [
        "xa",
        "ya",
        "xb",
        "yb"
    ],
    "polar": [
        "distance",
        "direction_x",
        "direction_y"
    ]
}


def get_features(data, family):
    return data[
        FEATURE_NAMES[family]
    ].to_numpy(dtype=float)


def encode_circular_target(delta_values):
    angles = (
        2.0
        * np.pi
        * np.asarray(delta_values, dtype=float)
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

    angles = np.mod(
        angles,
        2.0 * np.pi
    )

    values = (
        angles
        * GROUP_ORDER
        / (2.0 * np.pi)
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
    errors = circular_error(
        predicted,
        actual
    )

    return {
        "cyclic_mae": float(
            np.mean(errors)
        ),
        "cyclic_median_error": float(
            np.median(errors)
        ),
        "exact_accuracy": float(
            np.mean(errors == 0)
        ),
        "within_1_accuracy": float(
            np.mean(errors <= 1)
        ),
        "within_3_accuracy": float(
            np.mean(errors <= 3)
        ),
        "max_error": int(
            np.max(errors)
        )
    }


# ============================================================
# ОБУЧЕНИЕ МОДЕЛИ
# ============================================================

def create_model(seed):
    return ExtraTreesRegressor(
        n_estimators=TREE_COUNT,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=int(seed),
        n_jobs=N_JOBS
    )


def reverse_pairs(data):
    result = data.copy()

    result[["xa", "xb"]] = (
        data[["xb", "xa"]].to_numpy()
    )

    result[["ya", "yb"]] = (
        data[["yb", "ya"]].to_numpy()
    )

    result["dx"] = (
        result["xb"] - result["xa"]
    )

    result["dy"] = (
        result["yb"] - result["ya"]
    )

    result["distance"] = np.sqrt(
        result["dx"] ** 2
        + result["dy"] ** 2
    )

    safe_distance = result["distance"].replace(
        0,
        np.nan
    )

    result["direction_x"] = (
        result["dx"] / safe_distance
    ).fillna(0)

    result["direction_y"] = (
        result["dy"] / safe_distance
    ).fillna(0)

    return result


def predict_symmetrically(model, test_data, family):
    forward_features = get_features(
        test_data,
        family
    )

    reversed_data = reverse_pairs(
        test_data
    )

    reverse_features = get_features(
        reversed_data,
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
        prediction_forward
        - prediction_reverse
    ) / 2.0

    norms = np.linalg.norm(
        prediction,
        axis=1,
        keepdims=True
    )

    norms[norms == 0] = 1.0

    prediction = prediction / norms

    return decode_circular_prediction(
        prediction
    )


def fit_and_predict(
    train_data,
    test_data,
    family,
    seed
):
    model = create_model(seed)

    x_train = get_features(
        train_data,
        family
    )

    y_train = encode_circular_target(
        train_data["delta_scalar"].to_numpy()
    )

    model.fit(
        x_train,
        y_train
    )

    prediction = predict_symmetrically(
        model,
        test_data,
        family
    )

    importance = {
        name: float(value)
        for name, value in zip(
            FEATURE_NAMES[family],
            model.feature_importances_
        )
    }

    return prediction, importance


# ============================================================
# РАЗДЕЛЕНИЕ ДАННЫХ ПО ТОЧКАМ
# ============================================================

def split_point_ids(points, test_count, seed):
    generator = np.random.default_rng(seed)

    point_ids = np.array([
        item["point_id"]
        for item in points
    ])

    shuffled = generator.permutation(
        point_ids
    )

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

    return pairs.loc[
        mask
    ].copy().reset_index(drop=True)


def validate_disjoint_split(
    train_data,
    test_data
):
    train_points = (
        set(train_data["point_a"])
        | set(train_data["point_b"])
    )

    test_points = (
        set(test_data["point_a"])
        | set(test_data["point_b"])
    )

    overlap = train_points.intersection(
        test_points
    )

    if overlap:
        raise RuntimeError(
            "Обнаружена утечка точек между обучением и тестом."
        )


# ============================================================
# ПОВТОРНАЯ ПРОВЕРКА
# ============================================================

def run_repeated_validation(points):
    original_assignment = {
        item["point_id"]: item["original_scalar"]
        for item in points
    }

    pairs = generate_pairs(
        points,
        original_assignment
    )

    metric_records = []
    importance_records = []
    prediction_records = []

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

        for family_index, family in enumerate(
            ["xy_control", "polar"]
        ):
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

            metric_records.append({
                "repeat": repeat_index + 1,
                "family": family,
                "train_point_count": len(train_ids),
                "test_point_count": len(test_ids),
                "train_pair_count": len(train_data),
                "test_pair_count": len(test_data),
                **metrics
            })

            for feature_name, value in importance.items():
                importance_records.append({
                    "repeat": repeat_index + 1,
                    "family": family,
                    "feature": feature_name,
                    "importance": value
                })

            for index, (_, row) in enumerate(
                test_data.iterrows()
            ):
                prediction_records.append({
                    "repeat": repeat_index + 1,
                    "family": family,
                    "point_a": int(row["point_a"]),
                    "point_b": int(row["point_b"]),
                    "actual_delta": int(
                        row["delta_scalar"]
                    ),
                    "predicted_delta": int(
                        prediction[index]
                    )
                })

        print_message(
            f"Повторная проверка: "
            f"{repeat_index + 1}/{REPEAT_COUNT}"
        )

    return (
        pd.DataFrame(metric_records),
        pd.DataFrame(importance_records),
        pd.DataFrame(prediction_records)
    )


# ============================================================
# ФИКСИРОВАННЫЙ HOLDOUT
# ============================================================

def make_scalar_assignment(
    points,
    scalar_values=None
):
    point_ids = [
        item["point_id"]
        for item in points
    ]

    if scalar_values is None:
        scalar_values = [
            item["original_scalar"]
            for item in points
        ]

    return {
        int(point_id): int(scalar)
        for point_id, scalar in zip(
            point_ids,
            scalar_values
        )
    }


def evaluate_fixed_holdout(
    points,
    train_ids,
    test_ids,
    assignment,
    seed,
    save_predictions=False
):
    pairs = generate_pairs(
        points,
        assignment
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

    predictions = {}
    metrics = {}
    importances = {}

    for family_index, family in enumerate(
        ["xy_control", "polar"]
    ):
        prediction, importance = fit_and_predict(
            train_data,
            test_data,
            family,
            seed + family_index
        )

        predictions[family] = prediction
        importances[family] = importance

        metrics[family] = calculate_metrics(
            prediction,
            test_data["delta_scalar"].to_numpy()
        )

    improvement = (
        metrics["xy_control"]["cyclic_mae"]
        - metrics["polar"]["cyclic_mae"]
    )

    result = {
        "metrics": metrics,
        "improvement": float(improvement),
        "train_pair_count": len(train_data),
        "test_pair_count": len(test_data),
        "importances": importances
    }

    if save_predictions:
        prediction_records = []

        for index, (_, row) in enumerate(
            test_data.iterrows()
        ):
            prediction_records.append({
                "point_a": int(row["point_a"]),
                "point_b": int(row["point_b"]),
                "actual_delta": int(
                    row["delta_scalar"]
                ),
                "xy_control_prediction": int(
                    predictions["xy_control"][index]
                ),
                "polar_prediction": int(
                    predictions["polar"][index]
                ),
                "xy_control_error": int(
                    circular_error(
                        predictions["xy_control"][index],
                        row["delta_scalar"]
                    )
                ),
                "polar_error": int(
                    circular_error(
                        predictions["polar"][index],
                        row["delta_scalar"]
                    )
                )
            })

        result["predictions"] = pd.DataFrame(
            prediction_records
        )

    return result


# ============================================================
# ПЕРЕСТАНОВОЧНЫЙ ТЕСТ
# ============================================================

def run_permutation_test(points):
    train_ids, test_ids = split_point_ids(
        points,
        TEST_POINT_COUNT,
        RANDOM_SEED + 9000
    )

    original_values = np.array([
        item["original_scalar"]
        for item in points
    ])

    observed_assignment = make_scalar_assignment(
        points,
        original_values
    )

    print_message(
        "Оценка исходного назначения скаляров..."
    )

    observed = evaluate_fixed_holdout(
        points,
        train_ids,
        test_ids,
        observed_assignment,
        RANDOM_SEED + 9100,
        save_predictions=True
    )

    observed_control_mae = (
        observed["metrics"]["xy_control"]["cyclic_mae"]
    )

    observed_polar_mae = (
        observed["metrics"]["polar"]["cyclic_mae"]
    )

    observed_improvement = (
        observed["improvement"]
    )

    permutation_records = []

    for permutation_index in range(
        PERMUTATION_COUNT
    ):
        generator = np.random.default_rng(
            RANDOM_SEED
            + 10000
            + permutation_index
        )

        shuffled_values = generator.permutation(
            original_values
        )

        assignment = make_scalar_assignment(
            points,
            shuffled_values
        )

        result = evaluate_fixed_holdout(
            points,
            train_ids,
            test_ids,
            assignment,
            RANDOM_SEED + 20000 + permutation_index
        )

        control_mae = (
            result["metrics"]["xy_control"]["cyclic_mae"]
        )

        polar_mae = (
            result["metrics"]["polar"]["cyclic_mae"]
        )

        permutation_records.append({
            "permutation": permutation_index + 1,
            "xy_control_mae": control_mae,
            "polar_mae": polar_mae,
            "improvement": result["improvement"]
        })

        current = permutation_index + 1

        if (
            current % PERMUTATION_PROGRESS_STEP == 0
            or current == PERMUTATION_COUNT
        ):
            print_message(
                f"Перестановочный тест: "
                f"{current}/{PERMUTATION_COUNT}"
            )

    permutation_frame = pd.DataFrame(
        permutation_records
    )

    null_control = permutation_frame[
        "xy_control_mae"
    ].to_numpy()

    null_polar = permutation_frame[
        "polar_mae"
    ].to_numpy()

    null_improvement = permutation_frame[
        "improvement"
    ].to_numpy()

    # Проверка абсолютной ошибки полярной модели.
    p_polar_error = (
        1 + np.sum(
            null_polar <= observed_polar_mae
        )
    ) / (PERMUTATION_COUNT + 1)

    # Проверка преимущества polar над xy_control.
    p_improvement = (
        1 + np.sum(
            null_improvement >= observed_improvement
        )
    ) / (PERMUTATION_COUNT + 1)

    # Контрольная проверка для исходных координат.
    p_control_error = (
        1 + np.sum(
            null_control <= observed_control_mae
        )
    ) / (PERMUTATION_COUNT + 1)

    summary = {
        "train_point_count": len(train_ids),
        "test_point_count": len(test_ids),
        "train_pair_count": observed["train_pair_count"],
        "test_pair_count": observed["test_pair_count"],
        "observed_xy_control_mae": float(
            observed_control_mae
        ),
        "observed_polar_mae": float(
            observed_polar_mae
        ),
        "observed_improvement": float(
            observed_improvement
        ),
        "observed_xy_control_accuracy": float(
            observed["metrics"]["xy_control"]["exact_accuracy"]
        ),
        "observed_polar_accuracy": float(
            observed["metrics"]["polar"]["exact_accuracy"]
        ),
        "permutation_count": PERMUTATION_COUNT,
        "null_xy_control_mean": float(
            np.mean(null_control)
        ),
        "null_xy_control_std": float(
            np.std(null_control, ddof=1)
        ),
        "null_polar_mean": float(
            np.mean(null_polar)
        ),
        "null_polar_std": float(
            np.std(null_polar, ddof=1)
        ),
        "null_polar_min": float(
            np.min(null_polar)
        ),
        "null_polar_max": float(
            np.max(null_polar)
        ),
        "null_improvement_mean": float(
            np.mean(null_improvement)
        ),
        "null_improvement_std": float(
            np.std(null_improvement, ddof=1)
        ),
        "null_improvement_min": float(
            np.min(null_improvement)
        ),
        "null_improvement_max": float(
            np.max(null_improvement)
        ),
        "p_value_polar_error": float(
            p_polar_error
        ),
        "p_value_improvement": float(
            p_improvement
        ),
        "p_value_control_error": float(
            p_control_error
        ),
        "polar_feature_importance": observed[
            "importances"
        ]["polar"]
    }

    return (
        permutation_frame,
        summary,
        observed["predictions"],
        observed
    )


# ============================================================
# ОТЧЕТ
# ============================================================

def build_report(
    points,
    repeat_metrics,
    importance,
    permutation_summary
):
    lines = []

    lines.append("ОТЧЕТ ЭКСПЕРИМЕНТА C17A4")
    lines.append("=" * 70)
    lines.append("")

    lines.append("ПАРАМЕТРЫ КРИВОЙ")
    lines.append(
        f"Поле: {FIELD_P}"
    )
    lines.append(
        f"Кривая: y^2 = x^3 + {CURVE_B} (mod {FIELD_P})"
    )
    lines.append(
        f"Генератор: ({GENERATOR_X}, {GENERATOR_Y})"
    )
    lines.append(
        f"Порядок подгруппы: {GROUP_ORDER}"
    )
    lines.append(
        f"Количество ненулевых точек: {len(points)}"
    )
    lines.append("")

    lines.append("ПОВТОРНАЯ ПРОВЕРКА")
    lines.append("-" * 70)

    metrics_summary = repeat_metrics.groupby(
        "family"
    ).agg({
        "cyclic_mae": ["mean", "std", "min", "max"],
        "exact_accuracy": ["mean", "std"],
        "within_1_accuracy": ["mean", "std"],
        "within_3_accuracy": ["mean", "std"]
    })

    lines.append(
        metrics_summary.to_string()
    )
    lines.append("")

    lines.append("ВАЖНОСТЬ ПРИЗНАКОВ POLAR")
    lines.append("-" * 70)

    polar_importance = importance[
        importance["family"] == "polar"
    ].groupby(
        "feature"
    )["importance"].agg(
        ["mean", "std"]
    )

    lines.append(
        polar_importance.to_string()
    )
    lines.append("")

    lines.append("ФИКСИРОВАННЫЙ ПЕРЕСТАНОВОЧНЫЙ ТЕСТ")
    lines.append("-" * 70)

    for key, value in permutation_summary.items():
        lines.append(
            f"{key}: {value}"
        )

    lines.append("")

    lines.append("ИНТЕРПРЕТАЦИЯ")
    lines.append("-" * 70)
    lines.append(
        "Положительное improvement означает, что polar имеет "
        "меньшую циклическую MAE, чем xy_control."
    )
    lines.append(
        "p_value_polar_error проверяет абсолютную ошибку polar "
        "относительно перестановочного распределения."
    )
    lines.append(
        "p_value_improvement проверяет преимущество polar "
        "над xy_control относительно перестановочного распределения."
    )
    lines.append(
        "Перестановка выполняется на уровне назначения скаляров точкам, "
        "а не на уровне отдельных пар."
    )
    lines.append(
        "Тест относится к одной игрушечной кривой и одному "
        "фиксированному разбиению точек."
    )
    lines.append(
        "Геометрические признаки являются преобразованиями координат "
        "и не содержат дополнительной информации относительно них."
    )
    lines.append(
        "Результат не является решением задачи дискретного логарифма "
        "и не доказывает применимость к secp256k1."
    )

    return "\n".join(lines)


# ============================================================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ============================================================

def save_results(
    output_directory,
    points,
    repeat_metrics,
    importance,
    repeat_predictions,
    permutation_frame,
    permutation_summary,
    holdout_predictions,
    observed_result
):
    ensure_directory(output_directory)

    pd.DataFrame(points).to_csv(
        os.path.join(
            output_directory,
            "points.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    repeat_metrics.to_csv(
        os.path.join(
            output_directory,
            "repeat_metrics.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    importance.to_csv(
        os.path.join(
            output_directory,
            "feature_importance.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    repeat_predictions.to_csv(
        os.path.join(
            output_directory,
            "repeat_predictions.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    permutation_frame.to_csv(
        os.path.join(
            output_directory,
            "permutation_results.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    holdout_predictions.to_csv(
        os.path.join(
            output_directory,
            "holdout_predictions.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    save_json(
        permutation_summary,
        os.path.join(
            output_directory,
            "permutation_summary.json"
        )
    )

    config = {
        "experiment": EXPERIMENT_NAME,
        "field_p": FIELD_P,
        "curve_a": CURVE_A,
        "curve_b": CURVE_B,
        "generator": [
            GENERATOR_X,
            GENERATOR_Y
        ],
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
        "target_encoding": (
            "cos(2*pi*delta/73), sin(2*pi*delta/73)"
        ),
        "prediction_method": (
            "antisymmetric forward-reverse averaging"
        ),
        "permutation_unit": (
            "scalar assignment to points"
        )
    }

    save_json(
        config,
        os.path.join(
            output_directory,
            "config.json"
        )
    )

    report = build_report(
        points,
        repeat_metrics,
        importance,
        permutation_summary
    )

    with open(
        os.path.join(
            output_directory,
            "report.txt"
        ),
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
        for root, _, files in os.walk(
            output_directory
        ):
            for filename in files:
                full_path = os.path.join(
                    root,
                    filename
                )

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
# ЗАПУСК ЭКСПЕРИМЕНТА
# ============================================================

def main():
    start_time = datetime.now()

    timestamp = start_time.strftime(
        "%Y%m%d_%H%M%S"
    )

    output_directory = os.path.join(
        OUTPUT_ROOT,
        f"{EXPERIMENT_NAME}_{timestamp}"
    )

    ensure_directory(output_directory)

    print_message("=" * 65)
    print_message("ЗАПУСК ЭКСПЕРИМЕНТА C17A4")
    print_message("=" * 65)

    print_message(
        "Построение точек кривой..."
    )

    points = generate_points()

    print_message(
        f"Построено точек: {len(points)}"
    )

    print_message(
        "Повторная проверка полярной гипотезы..."
    )

    (
        repeat_metrics,
        importance,
        repeat_predictions
    ) = run_repeated_validation(points)

    print_message(
        "Запуск фиксированного перестановочного теста..."
    )

    (
        permutation_frame,
        permutation_summary,
        holdout_predictions,
        observed_result
    ) = run_permutation_test(points)

    print_message(
        "Сохранение результатов..."
    )

    save_results(
        output_directory,
        points,
        repeat_metrics,
        importance,
        repeat_predictions,
        permutation_frame,
        permutation_summary,
        holdout_predictions,
        observed_result
    )

    source_path = os.path.abspath(
        __file__
    )

    shutil.copy2(
        source_path,
        os.path.join(
            output_directory,
            "experimentC17A4.py"
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
    print_message("ЭКСПЕРИМЕНТ C17A4 ЗАВЕРШЕН")
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
    print_message(
        "СРЕДНИЕ РЕЗУЛЬТАТЫ ПОВТОРНОЙ ПРОВЕРКИ"
    )

    summary = repeat_metrics.groupby(
        "family"
    ).agg({
        "cyclic_mae": ["mean", "std"],
        "exact_accuracy": ["mean", "std"]
    })

    print(summary.to_string())

    print_message("")
    print_message(
        "РЕЗУЛЬТАТЫ ПЕРЕСТАНОВОЧНОГО ТЕСТА"
    )

    print_message(
        f"MAE исходных координат: "
        f"{permutation_summary['observed_xy_control_mae']:.6f}"
    )

    print_message(
        f"MAE polar: "
        f"{permutation_summary['observed_polar_mae']:.6f}"
    )

    print_message(
        f"Преимущество polar: "
        f"{permutation_summary['observed_improvement']:.6f}"
    )

    print_message(
        f"Среднее преимущество в перестановках: "
        f"{permutation_summary['null_improvement_mean']:.6f}"
    )

    print_message(
        f"p-value ошибки polar: "
        f"{permutation_summary['p_value_polar_error']:.6f}"
    )

    print_message(
        f"p-value преимущества polar: "
        f"{permutation_summary['p_value_improvement']:.6f}"
    )

    print_message("")
    print_message(
        "Полный отчет сохранен в report.txt"
    )

    print_message(
        "ZIP-архив содержит все таблицы и исходный код."
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print_message("")
        print_message(
            "ОШИБКА ВО ВРЕМЯ ЭКСПЕРИМЕНТА"
        )
        print_message(
            str(error)
        )
        print_message(
            traceback.format_exc()
        )
        raise
