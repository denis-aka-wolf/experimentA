# -*- coding: utf-8 -*-

import json
import math
import random
import shutil
import sys
import traceback
import zipfile

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier


# ============================================================
# НАСТРОЙКИ ЭКСПЕРИМЕНТА
# ============================================================

P = 499
CURVE_A = 0
CURVE_B = 7

GENERATOR = (3, 212)
GROUP_ORDER = 73

RANDOM_SEED = 20261103

NUMBER_OF_FOLDS = 6
NODES_PER_FOLD = 12

MAX_TRAIN_PAIRS = 900

OUTPUT_PREFIX = "experimentC16"

REGRESSION_MODELS = [
    "knn_15",
    "extra_trees_160",
    "random_forest_140",
]

CLASSIFICATION_MODELS = [
    "knn_classifier_15",
    "extra_trees_classifier_180",
]


# ============================================================
# АРИФМЕТИКА ЭЛЛИПТИЧЕСКОЙ КРИВОЙ
# ============================================================

def point_negate(point):
    if point is None:
        return None

    x, y = point

    return x, (-y) % P


def point_add(point_1, point_2):
    if point_1 is None:
        return point_2

    if point_2 is None:
        return point_1

    x1, y1 = point_1
    x2, y2 = point_2

    if x1 == x2 and (y1 + y2) % P == 0:
        return None

    if point_1 == point_2:
        denominator = (2 * y1) % P

        if denominator == 0:
            return None

        numerator = (3 * x1 * x1 + CURVE_A) % P
        slope = (numerator * pow(denominator, -1, P)) % P

    else:
        denominator = (x2 - x1) % P

        if denominator == 0:
            return None

        numerator = (y2 - y1) % P
        slope = (numerator * pow(denominator, -1, P)) % P

    x3 = (slope * slope - x1 - x2) % P
    y3 = (slope * (x1 - x3) - y1) % P

    return x3, y3


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


def is_valid_curve_point(point):
    if point is None:
        return True

    x, y = point

    left = (y * y) % P
    right = (x * x * x + CURVE_A * x + CURVE_B) % P

    return left == right


def build_point_table():
    point_table = {}

    for scalar in range(1, GROUP_ORDER):
        point = scalar_multiply(scalar, GENERATOR)

        if point is None:
            raise ValueError(
                "Получен нейтральный элемент до достижения порядка группы."
            )

        if not is_valid_curve_point(point):
            raise ValueError(
                "Обнаружена точка, не принадлежащая эллиптической кривой."
            )

        point_table[scalar] = point

    if scalar_multiply(GROUP_ORDER, GENERATOR) is not None:
        raise ValueError(
            "Проверка порядка генератора не пройдена."
        )

    unique_points = set(point_table.values())

    if len(unique_points) != GROUP_ORDER - 1:
        raise ValueError(
            "Обнаружены повторяющиеся точки в таблице кратных генератора."
        )

    return point_table


# ============================================================
# ГЕОМЕТРИЧЕСКИЕ ПРИЗНАКИ
# ============================================================

BASE_FEATURES = [
    "xa",
    "ya",
    "xb",
    "yb",
]

EUCLIDEAN_FEATURES = [
    "dx",
    "dy",
    "abs_dx",
    "abs_dy",
    "distance",
    "mid_x",
    "mid_y",
    "dot",
    "cross",
    "radius_a",
    "radius_b",
]

FINITE_FIELD_FEATURES = [
    "slope_difference",
    "slope_direct",
    "same_x",
    "same_y",
]


FEATURE_SETS = {
    "xy_control": BASE_FEATURES,

    "euclidean_geometry": (
        BASE_FEATURES + EUCLIDEAN_FEATURES
    ),

    "finite_field_geometry": (
        BASE_FEATURES + FINITE_FIELD_FEATURES
    ),

    "combined_geometry": (
        BASE_FEATURES
        + EUCLIDEAN_FEATURES
        + FINITE_FIELD_FEATURES
    ),
}


def safe_modular_slope(numerator, denominator):
    denominator %= P

    if denominator == 0:
        return 0.0

    value = (
        (numerator % P)
        * pow(denominator, -1, P)
    ) % P

    return value / P


def calculate_pair_features(point_a, point_b):
    xa, ya = point_a
    xb, yb = point_b

    xa_n = xa / P
    ya_n = ya / P
    xb_n = xb / P
    yb_n = yb / P

    dx = xb_n - xa_n
    dy = yb_n - ya_n

    distance = math.sqrt(dx * dx + dy * dy)

    mid_x = (xa_n + xb_n) / 2.0
    mid_y = (ya_n + yb_n) / 2.0

    dot = (
        xa * xb + ya * yb
    ) / (P * P)

    cross = (
        xa * yb - ya * xb
    ) / (P * P)

    radius_a = math.sqrt(
        xa_n * xa_n + ya_n * ya_n
    )

    radius_b = math.sqrt(
        xb_n * xb_n + yb_n * yb_n
    )

    denominator = (xb - xa) % P

    slope_difference = safe_modular_slope(
        yb + ya,
        denominator
    )

    slope_direct = safe_modular_slope(
        yb - ya,
        denominator
    )

    same_x = 1.0 if xa == xb else 0.0
    same_y = 1.0 if ya == yb else 0.0

    return {
        "xa": xa_n,
        "ya": ya_n,
        "xb": xb_n,
        "yb": yb_n,

        "dx": dx,
        "dy": dy,
        "abs_dx": abs(dx),
        "abs_dy": abs(dy),
        "distance": distance,
        "mid_x": mid_x,
        "mid_y": mid_y,
        "dot": dot,
        "cross": cross,
        "radius_a": radius_a,
        "radius_b": radius_b,

        "slope_difference": slope_difference,
        "slope_direct": slope_direct,
        "same_x": same_x,
        "same_y": same_y,
    }


# ============================================================
# ФОРМИРОВАНИЕ ОБУЧАЮЩЕГО НАБОРА
# ============================================================

def build_pair_dataset(point_table):
    records = []

    scalars = list(range(1, GROUP_ORDER))

    for scalar_a in scalars:
        point_a = point_table[scalar_a]

        for scalar_b in scalars:
            if scalar_a == scalar_b:
                continue

            point_b = point_table[scalar_b]

            relative_scalar = (
                scalar_b - scalar_a
            ) % GROUP_ORDER

            # Истинная разность используется только
            # для формирования обучающего ответа.
            difference_point = point_add(
                point_b,
                point_negate(point_a)
            )

            expected_point = point_table[relative_scalar]

            if difference_point != expected_point:
                raise ValueError(
                    "Нарушена проверка групповой разности."
                )

            features = calculate_pair_features(
                point_a,
                point_b
            )

            record = {
                "pair_id": f"{scalar_a}_{scalar_b}",

                "scalar_a": scalar_a,
                "scalar_b": scalar_b,
                "relative_scalar": relative_scalar,

                "point_a_x": point_a[0],
                "point_a_y": point_a[1],

                "point_b_x": point_b[0],
                "point_b_y": point_b[1],

                "target_r_x": difference_point[0],
                "target_r_y": difference_point[1],
            }

            record.update(features)

            records.append(record)

    return pd.DataFrame(records)


# ============================================================
# РАЗБИЕНИЕ ПО УЗЛАМ
# ============================================================

def build_node_folds():
    rng = np.random.default_rng(RANDOM_SEED)

    nodes = np.arange(1, GROUP_ORDER)

    rng.shuffle(nodes)

    if len(nodes) != NUMBER_OF_FOLDS * NODES_PER_FOLD:
        raise ValueError(
            "Количество узлов не соответствует настройкам разбиения."
        )

    chunks = np.array_split(nodes, NUMBER_OF_FOLDS)

    folds = []

    for fold_index, test_nodes in enumerate(chunks):
        test_set = set(int(value) for value in test_nodes)

        train_nodes = [
            int(value)
            for value in nodes
            if int(value) not in test_set
        ]

        folds.append({
            "fold": fold_index + 1,
            "train_nodes": sorted(train_nodes),
            "test_nodes": sorted(test_set),
        })

    return folds


def select_fold_pairs(dataset, node_list):
    node_set = set(node_list)

    mask = (
        dataset["scalar_a"].isin(node_set)
        & dataset["scalar_b"].isin(node_set)
    )

    return dataset.loc[mask].copy()


# ============================================================
# МОДЕЛИ
# ============================================================

def create_regression_model(model_name, seed):
    if model_name == "knn_15":
        return KNeighborsRegressor(
            n_neighbors=15,
            weights="distance",
            n_jobs=-1
        )

    if model_name == "extra_trees_160":
        return ExtraTreesRegressor(
            n_estimators=160,
            min_samples_leaf=2,
            max_features=1.0,
            n_jobs=-1,
            random_state=seed
        )

    if model_name == "random_forest_140":
        return RandomForestRegressor(
            n_estimators=140,
            min_samples_leaf=2,
            max_features=1.0,
            n_jobs=-1,
            random_state=seed
        )

    raise ValueError(
        f"Неизвестная регрессионная модель: {model_name}"
    )


def create_classification_model(model_name, seed):
    if model_name == "knn_classifier_15":
        return KNeighborsClassifier(
            n_neighbors=15,
            weights="distance",
            n_jobs=-1
        )

    if model_name == "extra_trees_classifier_180":
        return ExtraTreesClassifier(
            n_estimators=180,
            min_samples_leaf=2,
            max_features=1.0,
            n_jobs=-1,
            random_state=seed
        )

    raise ValueError(
        f"Неизвестная классификационная модель: {model_name}"
    )


# ============================================================
# ДЕКОДЕР КООРДИНАТ В СКАЛЯР
# ============================================================

def build_decoder(train_sample):
    decoder = {}

    for _, row in train_sample.iterrows():
        scalar = int(row["relative_scalar"])

        point = (
            int(row["target_r_x"]),
            int(row["target_r_y"])
        )

        if scalar in decoder:
            if decoder[scalar] != point:
                raise ValueError(
                    "Один скаляр соответствует разным точкам."
                )
        else:
            decoder[scalar] = point

    decoder_scalars = sorted(decoder.keys())

    decoder_points = np.array(
        [
            [
                decoder[scalar][0] / P,
                decoder[scalar][1] / P
            ]
            for scalar in decoder_scalars
        ],
        dtype=float
    )

    decoder_labels = np.array(
        decoder_scalars,
        dtype=int
    )

    return decoder_points, decoder_labels


def decode_nearest_points(predicted_points, decoder_points, decoder_labels):
    distances = np.sum(
        (
            predicted_points[:, None, :]
            - decoder_points[None, :, :]
        ) ** 2,
        axis=2
    )

    nearest_indices = np.argmin(distances, axis=1)

    predicted_scalars = decoder_labels[nearest_indices]

    nearest_distances = np.sqrt(
        distances[
            np.arange(len(predicted_points)),
            nearest_indices
        ]
    )

    return predicted_scalars, nearest_distances


# ============================================================
# МЕТРИКИ
# ============================================================

def circular_distance(predicted, actual):
    difference = abs(
        int(predicted) - int(actual)
    )

    return min(
        difference,
        GROUP_ORDER - difference
    )


def calculate_scalar_metrics(predictions, actual_values):
    distances = [
        circular_distance(predicted, actual)
        for predicted, actual in zip(
            predictions,
            actual_values
        )
    ]

    accuracy = np.mean(
        np.array(predictions) == np.array(actual_values)
    )

    return {
        "scalar_mae": float(np.mean(distances)),
        "scalar_accuracy": float(accuracy),
    }


def calculate_coordinate_metrics(
    predicted_points,
    actual_points
):
    differences = predicted_points - actual_points

    x_errors = np.abs(differences[:, 0])
    y_errors = np.abs(differences[:, 1])

    euclidean_errors = np.sqrt(
        np.sum(differences ** 2, axis=1)
    )

    return {
        "r_x_mae": float(np.mean(x_errors) * P),
        "r_y_mae": float(np.mean(y_errors) * P),

        "r_coordinate_mae": float(
            np.mean((x_errors + y_errors) / 2.0) * P
        ),

        "r_euclidean_mae": float(
            np.mean(euclidean_errors) * P
        ),
    }


# ============================================================
# СОХРАНЕНИЕ ИНФОРМАЦИИ О РАЗБИЕНИЯХ
# ============================================================

def save_fold_information(folds, output_dir):
    records = []

    for fold in folds:
        records.append({
            "fold": fold["fold"],
            "train_nodes": ",".join(
                str(value)
                for value in fold["train_nodes"]
            ),
            "test_nodes": ",".join(
                str(value)
                for value in fold["test_nodes"]
            ),
            "train_node_count": len(fold["train_nodes"]),
            "test_node_count": len(fold["test_nodes"]),
        })

    pd.DataFrame(records).to_csv(
        output_dir / "folds.csv",
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# ОСНОВНОЙ ЭКСПЕРИМЕНТ
# ============================================================

def run_experiment(output_dir):
    print("Проверка параметров эллиптической кривой...")

    point_table = build_point_table()

    print("Генератор и порядок группы проверены.")
    print("Формирование полного набора ориентированных пар...")

    dataset = build_pair_dataset(point_table)

    expected_pair_count = (
        (GROUP_ORDER - 1)
        * (GROUP_ORDER - 2)
    )

    if len(dataset) != expected_pair_count:
        raise ValueError(
            "Количество сформированных пар не совпадает с ожидаемым."
        )

    dataset.to_csv(
        output_dir / "dataset_pairs.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"Сформировано пар: {len(dataset)}"
    )

    folds = build_node_folds()

    save_fold_information(folds, output_dir)

    rng = np.random.default_rng(RANDOM_SEED)

    fold_metrics = []
    prediction_records = []
    decoder_coverage_records = []

    total_regression_runs = (
        NUMBER_OF_FOLDS
        * len(FEATURE_SETS)
        * len(REGRESSION_MODELS)
    )

    total_classification_runs = (
        NUMBER_OF_FOLDS
        * len(FEATURE_SETS)
        * len(CLASSIFICATION_MODELS)
    )

    total_runs = (
        total_regression_runs
        + total_classification_runs
    )

    current_run = 0

    for fold in folds:
        fold_number = fold["fold"]

        print("")
        print(
            f"Обработка разбиения {fold_number} "
            f"из {NUMBER_OF_FOLDS}..."
        )

        train_pairs = select_fold_pairs(
            dataset,
            fold["train_nodes"]
        )

        test_pairs = select_fold_pairs(
            dataset,
            fold["test_nodes"]
        )

        if len(test_pairs) == 0:
            raise ValueError(
                "Тестовая выборка оказалась пустой."
            )

        sample_size = min(
            MAX_TRAIN_PAIRS,
            len(train_pairs)
        )

        sample_indices = rng.choice(
            len(train_pairs),
            size=sample_size,
            replace=False
        )

        train_sample = train_pairs.iloc[
            sample_indices
        ].copy()

        decoder_points, decoder_labels = build_decoder(
            train_sample
        )

        decoder_coverage = len(decoder_labels)

        decoder_coverage_records.append({
            "fold": fold_number,
            "train_pair_count": len(train_pairs),
            "train_sample_count": len(train_sample),
            "decoder_unique_scalars": decoder_coverage,
            "decoder_total_possible": GROUP_ORDER - 1,
            "decoder_coverage_ratio": (
                decoder_coverage / (GROUP_ORDER - 1)
            ),
        })

        actual_scalars = test_pairs[
            "relative_scalar"
        ].to_numpy(dtype=int)

        actual_r_points = test_pairs[
            ["target_r_x", "target_r_y"]
        ].to_numpy(dtype=float) / P

        true_r_scalars, oracle_distances = decode_nearest_points(
            actual_r_points,
            decoder_points,
            decoder_labels
        )

        oracle_metrics = calculate_scalar_metrics(
            true_r_scalars,
            actual_scalars
        )

        for feature_name, feature_columns in FEATURE_SETS.items():
            x_train = train_sample[
                feature_columns
            ].to_numpy(dtype=float)

            x_test = test_pairs[
                feature_columns
            ].to_numpy(dtype=float)

            y_train_r = train_sample[
                ["target_r_x", "target_r_y"]
            ].to_numpy(dtype=float) / P

            y_train_scalar = train_sample[
                "relative_scalar"
            ].to_numpy(dtype=int)

            # ------------------------------------------------
            # ПЕРВАЯ СТАДИЯ:
            # ГЕОМЕТРИЧЕСКОЕ ПРЕДСКАЗАНИЕ КООРДИНАТ R
            # ------------------------------------------------

            for model_name in REGRESSION_MODELS:
                current_run += 1

                print(
                    f"Модель {current_run} из {total_runs}: "
                    f"{feature_name}, {model_name}"
                )

                model = create_regression_model(
                    model_name,
                    RANDOM_SEED + fold_number
                )

                model.fit(
                    x_train,
                    y_train_r
                )

                predicted_r = np.asarray(
                    model.predict(x_test),
                    dtype=float
                )

                if predicted_r.ndim == 1:
                    predicted_r = predicted_r.reshape(-1, 2)

                predicted_d, decoder_distances = decode_nearest_points(
                    predicted_r,
                    decoder_points,
                    decoder_labels
                )

                coordinate_metrics = calculate_coordinate_metrics(
                    predicted_r,
                    actual_r_points
                )

                scalar_metrics = calculate_scalar_metrics(
                    predicted_d,
                    actual_scalars
                )

                row_metrics = {
                    "fold": fold_number,
                    "pipeline_type": "Геометрическая цепочка",
                    "feature_set": feature_name,
                    "model": model_name,

                    "train_pair_count": len(train_pairs),
                    "train_sample_count": len(train_sample),
                    "test_pair_count": len(test_pairs),

                    "decoder_coverage": decoder_coverage,

                    **coordinate_metrics,

                    "stage2_scalar_mae": scalar_metrics["scalar_mae"],
                    "stage2_scalar_accuracy": scalar_metrics["scalar_accuracy"],

                    "oracle_decoder_mae": oracle_metrics["scalar_mae"],
                    "oracle_decoder_accuracy": oracle_metrics["scalar_accuracy"],

                    "mean_decoder_distance": float(
                        np.mean(decoder_distances)
                    ),

                    "oracle_mean_decoder_distance": float(
                        np.mean(oracle_distances)
                    ),
                }

                fold_metrics.append(row_metrics)

                for index, (_, test_row) in enumerate(
                    test_pairs.iterrows()
                ):
                    prediction_records.append({
                        "fold": fold_number,
                        "pipeline_type": "Геометрическая цепочка",
                        "feature_set": feature_name,
                        "model": model_name,

                        "pair_id": test_row["pair_id"],

                        "scalar_a": int(test_row["scalar_a"]),
                        "scalar_b": int(test_row["scalar_b"]),
                        "true_relative_scalar": int(
                            test_row["relative_scalar"]
                        ),

                        "true_r_x": int(test_row["target_r_x"]),
                        "true_r_y": int(test_row["target_r_y"]),

                        "predicted_r_x": float(
                            predicted_r[index, 0] * P
                        ),
                        "predicted_r_y": float(
                            predicted_r[index, 1] * P
                        ),

                        "predicted_relative_scalar": int(
                            predicted_d[index]
                        ),

                        "decoder_distance": float(
                            decoder_distances[index]
                        ),

                        "oracle_relative_scalar": int(
                            true_r_scalars[index]
                        ),

                        "oracle_decoder_distance": float(
                            oracle_distances[index]
                        ),
                    })

            # ------------------------------------------------
            # КОНТРОЛЬ:
            # ПРЯМОЕ ПРЕДСКАЗАНИЕ СКАЛЯРА ИЗ ПАРЫ
            # ------------------------------------------------

            for model_name in CLASSIFICATION_MODELS:
                current_run += 1

                print(
                    f"Модель {current_run} из {total_runs}: "
                    f"{feature_name}, {model_name}"
                )

                model = create_classification_model(
                    model_name,
                    RANDOM_SEED + fold_number
                )

                model.fit(
                    x_train,
                    y_train_scalar
                )

                predicted_scalar = model.predict(
                    x_test
                ).astype(int)

                scalar_metrics = calculate_scalar_metrics(
                    predicted_scalar,
                    actual_scalars
                )

                fold_metrics.append({
                    "fold": fold_number,
                    "pipeline_type": "Прямое предсказание скаляра",
                    "feature_set": feature_name,
                    "model": model_name,

                    "train_pair_count": len(train_pairs),
                    "train_sample_count": len(train_sample),
                    "test_pair_count": len(test_pairs),

                    "decoder_coverage": decoder_coverage,

                    "direct_scalar_mae": scalar_metrics["scalar_mae"],
                    "direct_scalar_accuracy": scalar_metrics["scalar_accuracy"],
                })

                for index, (_, test_row) in enumerate(
                    test_pairs.iterrows()
                ):
                    prediction_records.append({
                        "fold": fold_number,
                        "pipeline_type": "Прямое предсказание скаляра",
                        "feature_set": feature_name,
                        "model": model_name,

                        "pair_id": test_row["pair_id"],

                        "scalar_a": int(test_row["scalar_a"]),
                        "scalar_b": int(test_row["scalar_b"]),
                        "true_relative_scalar": int(
                            test_row["relative_scalar"]
                        ),

                        "predicted_relative_scalar": int(
                            predicted_scalar[index]
                        ),
                    })

    # ========================================================
    # СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
    # ========================================================

    metrics_df = pd.DataFrame(fold_metrics)

    predictions_df = pd.DataFrame(prediction_records)

    coverage_df = pd.DataFrame(
        decoder_coverage_records
    )

    metrics_df.to_csv(
        output_dir / "fold_metrics.csv",
        index=False,
        encoding="utf-8-sig"
    )

    predictions_df.to_csv(
        output_dir / "predictions.csv",
        index=False,
        encoding="utf-8-sig"
    )

    coverage_df.to_csv(
        output_dir / "decoder_coverage.csv",
        index=False,
        encoding="utf-8-sig"
    )

    numeric_columns = [
        column
        for column in metrics_df.select_dtypes(
            include=[np.number]
        ).columns
        if column != "fold"
    ]

    summary_df = (
        metrics_df
        .groupby(
            [
                "pipeline_type",
                "feature_set",
                "model"
            ],
            as_index=False
        )[numeric_columns]
        .mean()
    )

    summary_df.to_csv(
        output_dir / "summary_metrics.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # ИТОГОВАЯ ИНФОРМАЦИЯ
    # ========================================================

    best_geometric = summary_df[
        summary_df["pipeline_type"] == "Геометрическая цепочка"
    ].sort_values(
        by="stage2_scalar_mae",
        ascending=True
    )

    best_direct = summary_df[
        summary_df["pipeline_type"] == "Прямое предсказание скаляра"
    ].sort_values(
        by="direct_scalar_mae",
        ascending=True
    )

    result_summary = {
        "experiment": "C16",
        "timestamp": datetime.now().isoformat(),

        "curve": {
            "p": P,
            "a": CURVE_A,
            "b": CURVE_B,
            "generator": list(GENERATOR),
            "group_order": GROUP_ORDER,
        },

        "dataset": {
            "total_ordered_pairs": len(dataset),
            "fold_count": NUMBER_OF_FOLDS,
            "nodes_per_fold": NODES_PER_FOLD,
            "max_train_pairs": MAX_TRAIN_PAIRS,
        },

        "best_geometric_pipeline": (
            best_geometric.iloc[0].to_dict()
            if len(best_geometric) > 0
            else None
        ),

        "best_direct_pipeline": (
            best_direct.iloc[0].to_dict()
            if len(best_direct) > 0
            else None
        ),

        "interpretation_warning": (
            "Координаты истинной групповой разности используются "
            "только как обучающие ответы и для диагностического "
            "сравнения. Декодер обучается на разностях обучающей "
            "выборки. Его точность не является самостоятельным "
            "решением задачи дискретного логарифмирования."
        ),
    }

    with open(
        output_dir / "summary.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            result_summary,
            file,
            ensure_ascii=False,
            indent=4,
            default=str
        )

    # ========================================================
    # ТЕКСТОВЫЙ ОТЧЁТ
    # ========================================================

    report_lines = []

    report_lines.append("ЭКСПЕРИМЕНТ C16")
    report_lines.append("=" * 70)
    report_lines.append("")

    report_lines.append("ЦЕЛЬ")
    report_lines.append(
        "Проверка геометрической аппроксимации групповой разности "
        "двух точек и последующего восстановления относительного скаляра."
    )

    report_lines.append("")
    report_lines.append("ПАРАМЕТРЫ КРИВОЙ")
    report_lines.append(f"p = {P}")
    report_lines.append(f"a = {CURVE_A}")
    report_lines.append(f"b = {CURVE_B}")
    report_lines.append(f"G = {GENERATOR}")
    report_lines.append(f"Порядок группы = {GROUP_ORDER}")

    report_lines.append("")
    report_lines.append("РАЗМЕР НАБОРА")
    report_lines.append(
        f"Количество ориентированных пар: {len(dataset)}"
    )
    report_lines.append(
        f"Количество разбиений: {NUMBER_OF_FOLDS}"
    )
    report_lines.append(
        f"Максимум обучающих пар: {MAX_TRAIN_PAIRS}"
    )

    report_lines.append("")
    report_lines.append("ЛУЧШАЯ ГЕОМЕТРИЧЕСКАЯ ЦЕПОЧКА")

    if len(best_geometric) > 0:
        best_row = best_geometric.iloc[0]

        report_lines.append(
            f"Признаки: {best_row['feature_set']}"
        )
        report_lines.append(
            f"Модель: {best_row['model']}"
        )
        report_lines.append(
            f"Средняя ошибка скаляра: "
            f"{best_row['stage2_scalar_mae']:.6f}"
        )
        report_lines.append(
            f"Точность скаляра: "
            f"{best_row['stage2_scalar_accuracy']:.6f}"
        )
        report_lines.append(
            f"Средняя ошибка координат R: "
            f"{best_row['r_coordinate_mae']:.6f}"
        )

    report_lines.append("")
    report_lines.append("ЛУЧШАЯ ПРЯМАЯ МОДЕЛЬ")

    if len(best_direct) > 0:
        best_row = best_direct.iloc[0]

        report_lines.append(
            f"Признаки: {best_row['feature_set']}"
        )
        report_lines.append(
            f"Модель: {best_row['model']}"
        )
        report_lines.append(
            f"Средняя ошибка скаляра: "
            f"{best_row['direct_scalar_mae']:.6f}"
        )
        report_lines.append(
            f"Точность скаляра: "
            f"{best_row['direct_scalar_accuracy']:.6f}"
        )

    report_lines.append("")
    report_lines.append("МЕТОДОЛОГИЧЕСКИЕ ОГРАНИЧЕНИЯ")
    report_lines.append(
        "1. Истинная групповая разность используется для формирования "
        "обучающих ответов."
    )
    report_lines.append(
        "2. Групповая операция не входит в признаки и предсказание "
        "геометрической модели."
    )
    report_lines.append(
        "3. Декодер использует только соответствия, полученные "
        "из обучающей выборки."
    )
    report_lines.append(
        "4. Полнота декодера отдельно измеряется в каждом разбиении."
    )
    report_lines.append(
        "5. Точность декодера нельзя интерпретировать как решение ECDLP."
    )
    report_lines.append(
        "6. Результаты на игрушечной кривой не переносятся автоматически "
        "на secp256k1."
    )

    with open(
        output_dir / "report.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write(
            "\n".join(report_lines)
        )

    print("")
    print("Экспериментальные таблицы сохранены.")
    print("Подготовка итогового отчёта завершена.")

    return result_summary


# ============================================================
# СОЗДАНИЕ ZIP-АРХИВА
# ============================================================

def create_archive(output_dir, archive_path):
    print("Создание полного ZIP-архива...")

    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6
    ) as archive:

        for file_path in output_dir.rglob("*"):
            if file_path.is_file():
                archive_name = (
                    Path(output_dir.name)
                    / file_path.relative_to(output_dir)
                )

                archive.write(
                    file_path,
                    arcname=str(archive_name)
                )

    print("ZIP-архив создан.")


# ============================================================
# ТОЧКА ВХОДА
# ============================================================

def main():
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_dir = Path(
        f"{OUTPUT_PREFIX}_results_{timestamp}"
    )

    archive_path = Path(
        f"{OUTPUT_PREFIX}_{timestamp}.zip"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("ЗАПУСК ЭКСПЕРИМЕНТА C16")
    print("=" * 70)

    print(f"Папка результатов: {output_dir}")

    try:
        source_path = Path(__file__).resolve()

        shutil.copy2(
            source_path,
            output_dir / "experimentC16.py"
        )

        run_experiment(output_dir)

        readme = [
            "ЭКСПЕРИМЕНТ C16",
            "",
            "Основная цель:",
            "Геометрическая аппроксимация координат групповой разности",
            "с последующей оценкой восстановления относительного скаляра.",
            "",
            "Основные файлы:",
            "dataset_pairs.csv - полный набор ориентированных пар.",
            "folds.csv - состав обучающих и тестовых узлов.",
            "fold_metrics.csv - результаты каждого разбиения.",
            "summary_metrics.csv - средние метрики моделей.",
            "predictions.csv - индивидуальные прогнозы.",
            "decoder_coverage.csv - полнота декодера по разбиениям.",
            "summary.json - итоговая машинно-читаемая сводка.",
            "report.txt - текстовый отчёт.",
            "experimentC16.py - копия исполняемого файла.",
            "",
            "Важно:",
            "Декодер использует обучающие соответствия точек и скаляров.",
            "Его точность не является доказательством решения ECDLP.",
            "Эксперимент не выполняет прямой перебор скаляра при предсказании.",
        ]

        with open(
            output_dir / "README.txt",
            "w",
            encoding="utf-8"
        ) as file:
            file.write(
                "\n".join(readme)
            )

        create_archive(
            output_dir,
            archive_path
        )

        print("")
        print("=" * 70)
        print("ЭКСПЕРИМЕНТ C16 УСПЕШНО ЗАВЕРШЁН")
        print("=" * 70)

        print(f"Папка результатов: {output_dir.resolve()}")
        print(f"ZIP-архив: {archive_path.resolve()}")

    except Exception as error:
        error_text = (
            "ОШИБКА ВЫПОЛНЕНИЯ ЭКСПЕРИМЕНТА C16\n\n"
            + str(error)
            + "\n\n"
            + traceback.format_exc()
        )

        with open(
            output_dir / "error_log.txt",
            "w",
            encoding="utf-8"
        ) as file:
            file.write(error_text)

        print("")
        print("Во время эксперимента произошла ошибка.")
        print(f"Подробный журнал: {output_dir / 'error_log.txt'}")

        try:
            create_archive(
                output_dir,
                archive_path
            )
        except Exception:
            print("Не удалось создать архив аварийных результатов.")

        raise


if __name__ == "__main__":
    main()