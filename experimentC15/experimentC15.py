import os
import sys
import json
import time
import zipfile
import traceback
import platform
from datetime import datetime
from contextlib import redirect_stdout

try:
    import numpy as np
    import pandas as pd
    import sklearn

    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

except ImportError as error:
    print("ОШИБКА: не удалось загрузить необходимые библиотеки.")
    print("Установите зависимости командой:")
    print("python -m pip install numpy pandas scikit-learn")
    print("Подробности:", str(error))
    raise SystemExit(1)


# ============================================================
# НАСТРОЙКИ ЭКСПЕРИМЕНТА
# ============================================================

FIELD_P = 499
CURVE_A = 0
CURVE_B = 7

GENERATOR = (3, 212)
GROUP_ORDER = 73

RANDOM_SEED = 15015

FOLD_COUNT = 6
NODES_PER_FOLD = 12

MAX_TRAIN_PAIRS = 900

PERMUTATION_COUNT = 39

RESULT_PREFIX = "experimentC15"


# ============================================================
# ОПИСАНИЯ МОДЕЛЕЙ
# ============================================================

MODEL_SPECS = [
    {
        "name": "knn_5",
        "type": "knn",
        "neighbors": 5,
        "weights": "distance"
    },
    {
        "name": "knn_15",
        "type": "knn",
        "neighbors": 15,
        "weights": "distance"
    },
    {
        "name": "knn_31",
        "type": "knn",
        "neighbors": 31,
        "weights": "distance"
    },
    {
        "name": "extra_trees_120",
        "type": "extra_trees",
        "estimators": 120,
        "max_depth": 14,
        "min_samples_leaf": 3
    },
    {
        "name": "extra_trees_160",
        "type": "extra_trees",
        "estimators": 160,
        "max_depth": None,
        "min_samples_leaf": 5
    }
]


PRIMARY_BLOCKS = [
    "xy_control",
    "euclidean_geometry",
    "finite_field_chord",
    "combined_geometry"
]

DIAGNOSTIC_BLOCKS = [
    "group_difference_control"
]

ALL_BLOCKS = PRIMARY_BLOCKS + DIAGNOSTIC_BLOCKS


# ============================================================
# ВСПОМОГАТЕЛЬНЫЙ ВЫВОД
# ============================================================

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            stream.write(message)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def print_section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ============================================================
# АРИФМЕТИКА ЭЛЛИПТИЧЕСКОЙ КРИВОЙ
# ============================================================

def is_on_curve(point):
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


def point_add_with_slope(point1, point2):
    if point1 is None:
        return point2, None, "нейтральный_элемент"

    if point2 is None:
        return point1, None, "нейтральный_элемент"

    x1, y1 = point1
    x2, y2 = point2

    if x1 == x2 and (y1 + y2) % FIELD_P == 0:
        return None, None, "вертикальная_прямая"

    if point1 == point2:
        denominator = (2 * y1) % FIELD_P

        if denominator == 0:
            return None, None, "вертикальная_касательная"

        numerator = (
            3 * x1 * x1 + CURVE_A
        ) % FIELD_P

        slope = (
            numerator * pow(denominator, -1, FIELD_P)
        ) % FIELD_P

        kind = "касательная"

    else:
        denominator = (x2 - x1) % FIELD_P

        if denominator == 0:
            return None, None, "вертикальная_прямая"

        numerator = (y2 - y1) % FIELD_P

        slope = (
            numerator * pow(denominator, -1, FIELD_P)
        ) % FIELD_P

        kind = "секущая"

    x3 = (
        slope * slope - x1 - x2
    ) % FIELD_P

    y3 = (
        slope * (x1 - x3) - y1
    ) % FIELD_P

    result = (x3, y3)

    if not is_on_curve(result):
        raise ValueError(
            "Ошибка арифметики: результат сложения не лежит на кривой."
        )

    return result, slope, kind


def point_add(point1, point2):
    result, _, _ = point_add_with_slope(point1, point2)
    return result


def build_group():
    points = {}

    current = None

    for scalar in range(1, GROUP_ORDER + 1):
        current = point_add(current, GENERATOR)

        if scalar < GROUP_ORDER:
            if current is None:
                raise ValueError(
                    "Генератор имеет порядок меньше ожидаемого."
                )

            if current in points.values():
                raise ValueError(
                    "Обнаружено повторение точки до достижения порядка группы."
                )

            if not is_on_curve(current):
                raise ValueError(
                    "Обнаружена точка вне эллиптической кривой."
                )

            points[scalar] = current

    if current is not None:
        raise ValueError(
            "После 73 умножений генератор не вернулся в бесконечность."
        )

    if len(set(points.values())) != GROUP_ORDER - 1:
        raise ValueError(
            "Количество уникальных ненулевых точек не совпадает с ожидаемым."
        )

    return points


# ============================================================
# ГЕОМЕТРИЧЕСКИЕ ПРИЗНАКИ ПАРЫ ТОЧЕК
# ============================================================

def build_pair_features(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    dx = x2 - x1
    dy = y2 - y1

    abs_dx = abs(dx)
    abs_dy = abs(dy)

    distance_squared = dx * dx + dy * dy
    distance = float(np.sqrt(distance_squared))

    midpoint_x = (x1 + x2) / 2.0
    midpoint_y = (y1 + y2) / 2.0

    radius1 = float(np.sqrt(x1 * x1 + y1 * y1))
    radius2 = float(np.sqrt(x2 * x2 + y2 * y2))

    angle = float(np.arctan2(dy, dx))

    slope = dy / dx if dx != 0 else 0.0
    vertical_flag = 1.0 if dx == 0 else 0.0

    dot_product = x1 * x2 + y1 * y2
    cross_product = x1 * y2 - y1 * x2

    denominator = radius1 * radius2

    if denominator == 0:
        cosine_between = 0.0
        sine_between = 0.0
    else:
        cosine_between = dot_product / denominator
        sine_between = cross_product / denominator

    angle1 = float(np.arctan2(y1, x1))
    angle2 = float(np.arctan2(y2, x2))

    euclidean_features = [
        x1 / FIELD_P,
        y1 / FIELD_P,
        x2 / FIELD_P,
        y2 / FIELD_P,

        dx / FIELD_P,
        dy / FIELD_P,

        abs_dx / FIELD_P,
        abs_dy / FIELD_P,

        distance / FIELD_P,
        distance_squared / (FIELD_P * FIELD_P),

        midpoint_x / FIELD_P,
        midpoint_y / FIELD_P,

        slope / FIELD_P,
        vertical_flag,

        np.sin(angle),
        np.cos(angle),

        radius1 / FIELD_P,
        radius2 / FIELD_P,

        (radius2 - radius1) / FIELD_P,

        dot_product / (FIELD_P * FIELD_P),
        cross_product / (FIELD_P * FIELD_P),

        cosine_between,
        sine_between,

        np.sin(angle1),
        np.cos(angle1),

        np.sin(angle2),
        np.cos(angle2)
    ]

    xy_features = [
        x1 / FIELD_P,
        y1 / FIELD_P,
        x2 / FIELD_P,
        y2 / FIELD_P
    ]

    # Для разности Qb - Qa складываем Qb с противоположной точкой -Qa.
    negative_first = point_neg(point1)

    difference_point, slope_mod, slope_kind = (
        point_add_with_slope(negative_first, point2)
    )

    if difference_point is None:
        raise ValueError(
            "Разность двух различных точек неожиданно оказалась бесконечностью."
        )

    x_delta_mod = (x2 - x1) % FIELD_P
    y_sum_mod = (y2 + y1) % FIELD_P

    if slope_mod is None:
        raise ValueError(
            "Не удалось вычислить наклон конечнополевой секущей."
        )

    finite_field_features = [
        x_delta_mod / FIELD_P,
        y_sum_mod / FIELD_P,
        slope_mod / FIELD_P,
        ((slope_mod * slope_mod) % FIELD_P) / FIELD_P,
        1.0 if slope_kind == "касательная" else 0.0
    ]

    group_difference_features = [
        difference_point[0] / FIELD_P,
        difference_point[1] / FIELD_P
    ]

    combined_features = (
        euclidean_features + finite_field_features
    )

    return {
        "xy_control": xy_features,
        "euclidean_geometry": euclidean_features,
        "finite_field_chord": finite_field_features,
        "combined_geometry": combined_features,
        "group_difference_control": group_difference_features
    }


# ============================================================
# ПОСТРОЕНИЕ НАБОРА ПАР
# ============================================================

def build_dataset(points):
    pair_a = []
    pair_b = []

    feature_rows = {
        block_name: []
        for block_name in ALL_BLOCKS
    }

    for scalar_a in range(1, GROUP_ORDER):
        for scalar_b in range(1, GROUP_ORDER):

            if scalar_a == scalar_b:
                continue

            point_a = points[scalar_a]
            point_b = points[scalar_b]

            features = build_pair_features(
                point_a,
                point_b
            )

            pair_a.append(scalar_a)
            pair_b.append(scalar_b)

            for block_name in ALL_BLOCKS:
                feature_rows[block_name].append(
                    features[block_name]
                )

    pair_a = np.asarray(pair_a, dtype=int)
    pair_b = np.asarray(pair_b, dtype=int)

    feature_arrays = {}

    for block_name in ALL_BLOCKS:
        feature_arrays[block_name] = np.asarray(
            feature_rows[block_name],
            dtype=float
        )

    expected_pairs = (
        (GROUP_ORDER - 1) * (GROUP_ORDER - 2)
    )

    if len(pair_a) != expected_pairs:
        raise ValueError(
            "Количество пар не совпадает с ожидаемым."
        )

    return pair_a, pair_b, feature_arrays


# ============================================================
# РАЗБИЕНИЕ НА НЕПЕРЕСЕКАЮЩИЕСЯ ГРУППЫ ТОЧЕК
# ============================================================

def build_folds(pair_a, pair_b):
    rng = np.random.default_rng(RANDOM_SEED)

    nodes = np.arange(1, GROUP_ORDER)

    shuffled_nodes = rng.permutation(nodes)

    node_groups = np.array_split(
        shuffled_nodes,
        FOLD_COUNT
    )

    folds = []

    for fold_id, test_nodes_array in enumerate(node_groups):
        test_nodes = set(
            int(value)
            for value in test_nodes_array
        )

        train_mask = np.array([
            int(a) not in test_nodes
            and int(b) not in test_nodes
            for a, b in zip(pair_a, pair_b)
        ])

        test_mask = np.array([
            int(a) in test_nodes
            and int(b) in test_nodes
            for a, b in zip(pair_a, pair_b)
        ])

        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]

        sample_rng = np.random.default_rng(
            RANDOM_SEED + 1000 + fold_id
        )

        sample_size = min(
            MAX_TRAIN_PAIRS,
            len(train_indices)
        )

        sampled_train_indices = sample_rng.choice(
            train_indices,
            size=sample_size,
            replace=False
        )

        sampled_train_indices.sort()

        train_nodes = set(
            int(value)
            for value in pair_a[sampled_train_indices]
        )

        train_nodes.update(
            int(value)
            for value in pair_b[sampled_train_indices]
        )

        test_endpoint_nodes = set(test_nodes)

        if train_nodes.intersection(test_endpoint_nodes):
            raise ValueError(
                "Обнаружена утечка точек между обучением и тестом."
            )

        folds.append({
            "fold_id": fold_id + 1,
            "test_nodes": sorted(test_nodes),
            "train_indices": sampled_train_indices,
            "test_indices": test_indices
        })

    return folds


# ============================================================
# ЦИКЛИЧЕСКОЕ ПРЕДСТАВЛЕНИЕ ЦЕЛЕВОЙ МЕТКИ
# ============================================================

def encode_targets(targets):
    angles = (
        2.0 * np.pi * targets.astype(float) / GROUP_ORDER
    )

    return np.column_stack([
        np.sin(angles),
        np.cos(angles)
    ])


def decode_targets(predictions):
    angles = np.arctan2(
        predictions[:, 0],
        predictions[:, 1]
    )

    normalized = np.mod(
        angles,
        2.0 * np.pi
    )

    scalar_values = (
        normalized * GROUP_ORDER / (2.0 * np.pi)
    )

    return (
        np.rint(scalar_values).astype(int)
        % GROUP_ORDER
    )


def circular_errors(predicted, actual):
    forward = (
        predicted - actual
    ) % GROUP_ORDER

    backward = (
        actual - predicted
    ) % GROUP_ORDER

    return np.minimum(forward, backward)


# ============================================================
# СОЗДАНИЕ МОДЕЛИ
# ============================================================

def build_model(spec, fold_id):
    if spec["type"] == "knn":
        return Pipeline([
            ("scaler", StandardScaler()),
            (
                "model",
                KNeighborsRegressor(
                    n_neighbors=spec["neighbors"],
                    weights=spec["weights"],
                    metric="minkowski",
                    p=2,
                    n_jobs=-1
                )
            )
        ])

    if spec["type"] == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=spec["estimators"],
            max_depth=spec["max_depth"],
            min_samples_leaf=spec["min_samples_leaf"],
            max_features=1.0,
            random_state=RANDOM_SEED + fold_id,
            n_jobs=-1
        )

    raise ValueError(
        "Неизвестный тип модели."
    )


# ============================================================
# ОЦЕНКА ВСЕХ МОДЕЛЕЙ
# ============================================================

def evaluate_labels(
    pair_a,
    pair_b,
    feature_arrays,
    folds,
    label_map,
    collect_predictions=False
):
    targets = np.array([
        (
            label_map[int(b)] - label_map[int(a)]
        ) % GROUP_ORDER
        for a, b in zip(pair_a, pair_b)
    ], dtype=int)

    if np.any(targets == 0):
        raise ValueError(
            "В наборе обнаружена нулевая разность скаляров."
        )

    encoded_targets = encode_targets(targets)

    metric_records = []
    prediction_records = []

    for block_name in ALL_BLOCKS:
        feature_matrix = feature_arrays[block_name]

        for spec in MODEL_SPECS:
            config_name = (
                block_name + "__" + spec["name"]
            )

            all_actual = []
            all_predicted = []

            for fold in folds:
                fold_id = fold["fold_id"]

                train_indices = fold["train_indices"]
                test_indices = fold["test_indices"]

                model = build_model(
                    spec,
                    fold_id
                )

                model.fit(
                    feature_matrix[train_indices],
                    encoded_targets[train_indices]
                )

                raw_predictions = model.predict(
                    feature_matrix[test_indices]
                )

                predicted_values = decode_targets(
                    raw_predictions
                )

                actual_values = targets[test_indices]

                errors = circular_errors(
                    predicted_values,
                    actual_values
                )

                all_actual.extend(
                    actual_values.tolist()
                )

                all_predicted.extend(
                    predicted_values.tolist()
                )

                if collect_predictions:
                    for position, pair_index in enumerate(test_indices):
                        prediction_records.append({
                            "configuration": config_name,
                            "feature_block": block_name,
                            "model": spec["name"],
                            "fold": fold_id,

                            "point_a_id": int(pair_a[pair_index]),
                            "point_b_id": int(pair_b[pair_index]),

                            "target_difference": int(
                                actual_values[position]
                            ),

                            "predicted_difference": int(
                                predicted_values[position]
                            ),

                            "circular_error": int(
                                errors[position]
                            ),

                            "exact_match": int(
                                predicted_values[position]
                                == actual_values[position]
                            )
                        })

            all_actual = np.asarray(
                all_actual,
                dtype=int
            )

            all_predicted = np.asarray(
                all_predicted,
                dtype=int
            )

            errors = circular_errors(
                all_predicted,
                all_actual
            )

            metric_records.append({
                "configuration": config_name,
                "feature_block": block_name,
                "model": spec["name"],

                "mae": float(np.mean(errors)),
                "median_error": float(np.median(errors)),
                "max_error": int(np.max(errors)),

                "accuracy": float(
                    np.mean(all_predicted == all_actual)
                ),

                "sample_count": int(len(all_actual))
            })

    metrics = pd.DataFrame(metric_records)

    metrics = metrics.sort_values(
        ["mae", "accuracy"],
        ascending=[True, False]
    ).reset_index(drop=True)

    predictions = pd.DataFrame(
        prediction_records
    )

    return metrics, predictions


# ============================================================
# ПЕРЕСТАНОВКИ СКАЛЯРНЫХ МЕТОК
# ============================================================

def build_random_label_map(rng):
    nodes = np.arange(1, GROUP_ORDER)

    shuffled_labels = rng.permutation(
        np.arange(1, GROUP_ORDER)
    )

    return {
        int(node): int(label)
        for node, label in zip(nodes, shuffled_labels)
    }


def run_permutation_test(
    pair_a,
    pair_b,
    feature_arrays,
    folds,
    observed_primary_mae,
    observed_primary_accuracy,
    observed_diagnostic_mae,
    observed_diagnostic_accuracy
):
    rng = np.random.default_rng(
        RANDOM_SEED + 50000
    )

    records = []

    print_section("ПЕРЕСТАНОВОЧНЫЙ ТЕСТ")

    print(
        "Количество перестановок:",
        PERMUTATION_COUNT
    )

    for permutation_id in range(1, PERMUTATION_COUNT + 1):
        label_map = build_random_label_map(rng)

        metrics, _ = evaluate_labels(
            pair_a,
            pair_b,
            feature_arrays,
            folds,
            label_map,
            collect_predictions=False
        )

        primary_metrics = metrics[
            metrics["feature_block"].isin(
                PRIMARY_BLOCKS
            )
        ]

        diagnostic_metrics = metrics[
            metrics["feature_block"].isin(
                DIAGNOSTIC_BLOCKS
            )
        ]

        best_primary = primary_metrics.iloc[0]

        best_diagnostic = diagnostic_metrics.iloc[0]

        records.append({
            "permutation": permutation_id,

            "best_primary_mae": float(
                best_primary["mae"]
            ),

            "best_primary_accuracy": float(
                primary_metrics["accuracy"].max()
            ),

            "best_primary_configuration": (
                best_primary["configuration"]
            ),

            "best_diagnostic_mae": float(
                best_diagnostic["mae"]
            ),

            "best_diagnostic_accuracy": float(
                diagnostic_metrics["accuracy"].max()
            ),

            "best_diagnostic_configuration": (
                best_diagnostic["configuration"]
            )
        })

        if (
            permutation_id % 5 == 0
            or permutation_id == PERMUTATION_COUNT
        ):
            print(
                "Выполнено перестановок:",
                permutation_id,
                "из",
                PERMUTATION_COUNT
            )

    permutation_frame = pd.DataFrame(records)

    primary_null = permutation_frame[
        "best_primary_mae"
    ].to_numpy()

    primary_accuracy_null = permutation_frame[
        "best_primary_accuracy"
    ].to_numpy()

    diagnostic_null = permutation_frame[
        "best_diagnostic_mae"
    ].to_numpy()

    diagnostic_accuracy_null = permutation_frame[
        "best_diagnostic_accuracy"
    ].to_numpy()

    primary_p_value = (
        1 + np.sum(
            primary_null <= observed_primary_mae
        )
    ) / (
        PERMUTATION_COUNT + 1
    )

    primary_accuracy_p_value = (
        1 + np.sum(
            primary_accuracy_null >= observed_primary_accuracy
        )
    ) / (
        PERMUTATION_COUNT + 1
    )

    diagnostic_p_value = (
        1 + np.sum(
            diagnostic_null <= observed_diagnostic_mae
        )
    ) / (
        PERMUTATION_COUNT + 1
    )

    diagnostic_accuracy_p_value = (
        1 + np.sum(
            diagnostic_accuracy_null >= observed_diagnostic_accuracy
        )
    ) / (
        PERMUTATION_COUNT + 1
    )

    statistics = {
        "primary_null_mean_mae": float(
            np.mean(primary_null)
        ),

        "primary_null_min_mae": float(
            np.min(primary_null)
        ),

        "primary_null_max_mae": float(
            np.max(primary_null)
        ),

        "primary_p_value": float(
            primary_p_value
        ),

        "primary_accuracy_null_max": float(
            np.max(primary_accuracy_null)
        ),

        "primary_accuracy_p_value": float(
            primary_accuracy_p_value
        ),

        "diagnostic_null_mean_mae": float(
            np.mean(diagnostic_null)
        ),

        "diagnostic_null_min_mae": float(
            np.min(diagnostic_null)
        ),

        "diagnostic_p_value": float(
            diagnostic_p_value
        ),

        "diagnostic_accuracy_p_value": float(
            diagnostic_accuracy_p_value
        )
    }

    return permutation_frame, statistics


# ============================================================
# СОХРАНЕНИЕ НАБОРА ДАННЫХ
# ============================================================

def save_dataset(
    output_dir,
    pair_a,
    pair_b,
    points,
    feature_arrays
):
    records = []

    for index, (scalar_a, scalar_b) in enumerate(
        zip(pair_a, pair_b)
    ):
        point_a = points[int(scalar_a)]
        point_b = points[int(scalar_b)]

        record = {
            "point_a_id": int(scalar_a),
            "point_b_id": int(scalar_b),

            "x_a": point_a[0],
            "y_a": point_a[1],

            "x_b": point_b[0],
            "y_b": point_b[1],

            "target_difference": (
                int(scalar_b) - int(scalar_a)
            ) % GROUP_ORDER
        }

        for block_name in ALL_BLOCKS:
            values = feature_arrays[block_name][index]

            for feature_id, value in enumerate(values):
                record[
                    block_name + "_feature_" + str(feature_id + 1)
                ] = float(value)

        records.append(record)

    frame = pd.DataFrame(records)

    path = os.path.join(
        output_dir,
        "pair_dataset.csv"
    )

    frame.to_csv(
        path,
        index=False,
        encoding="utf-8-sig"
    )

    return path


def save_feature_definitions(output_dir, feature_arrays):
    records = []

    descriptions = {
        "xy_control": (
            "Координаты двух концов ориентированной пары."
        ),

        "euclidean_geometry": (
            "Координаты, расстояния, направления, "
            "середина хорды, радиусы, скалярное и векторное произведения."
        ),

        "finite_field_chord": (
            "Конечнополевые характеристики секущей "
            "для групповой разности Qb - Qa."
        ),

        "combined_geometry": (
            "Евклидова геометрия вместе с характеристиками "
            "конечнополевой секущей."
        ),

        "group_difference_control": (
            "Координаты точки Qb - Qa. "
            "Диагностический контроль сведения пары к одной точке."
        )
    }

    for block_name in ALL_BLOCKS:
        for feature_id in range(
            feature_arrays[block_name].shape[1]
        ):
            records.append({
                "feature_block": block_name,
                "feature_id": feature_id + 1,
                "description": descriptions[block_name]
            })

    frame = pd.DataFrame(records)

    path = os.path.join(
        output_dir,
        "feature_definitions.csv"
    )

    frame.to_csv(
        path,
        index=False,
        encoding="utf-8-sig"
    )

    return path


# ============================================================
# СОХРАНЕНИЕ КОНФИГУРАЦИИ
# ============================================================

def save_configuration(output_dir):
    configuration = {
        "experiment": RESULT_PREFIX,

        "curve": {
            "p": FIELD_P,
            "a": CURVE_A,
            "b": CURVE_B,
            "generator": list(GENERATOR),
            "group_order": GROUP_ORDER
        },

        "random_seed": RANDOM_SEED,

        "fold_count": FOLD_COUNT,
        "nodes_per_fold": NODES_PER_FOLD,

        "maximum_training_pairs": MAX_TRAIN_PAIRS,

        "permutation_count": PERMUTATION_COUNT,

        "primary_feature_blocks": PRIMARY_BLOCKS,
        "diagnostic_feature_blocks": DIAGNOSTIC_BLOCKS,

        "models": MODEL_SPECS,

        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__
    }

    path = os.path.join(
        output_dir,
        "configuration.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            configuration,
            file,
            ensure_ascii=False,
            indent=4
        )

    return path


# ============================================================
# ТЕКСТОВЫЙ ОТЧЁТ
# ============================================================

def save_summary(
    output_dir,
    points,
    pair_a,
    folds,
    observed_metrics,
    permutation_statistics,
    elapsed_seconds
):
    primary_metrics = observed_metrics[
        observed_metrics["feature_block"].isin(
            PRIMARY_BLOCKS
        )
    ]

    diagnostic_metrics = observed_metrics[
        observed_metrics["feature_block"].isin(
            DIAGNOSTIC_BLOCKS
        )
    ]

    best_primary = primary_metrics.iloc[0]
    best_diagnostic = diagnostic_metrics.iloc[0]

    lines = []

    lines.append("ЭКСПЕРИМЕНТ C15")
    lines.append("Относительное скалярное расстояние через геометрию пары точек")
    lines.append("")

    lines.append("ПАРАМЕТРЫ КРИВОЙ")
    lines.append("Поле: p = " + str(FIELD_P))
    lines.append("Кривая: y^2 = x^3 + 7 mod 499")
    lines.append("Генератор: " + str(GENERATOR))
    lines.append("Порядок группы: " + str(GROUP_ORDER))
    lines.append("Ненулевых точек: " + str(len(points)))
    lines.append("Количество ориентированных пар: " + str(len(pair_a)))
    lines.append("")

    lines.append("ПРОВЕРКА")
    lines.append("Количество блоков: " + str(len(folds)))
    lines.append("Точки обучения и теста не пересекаются внутри каждого блока.")
    lines.append("Пары между разными блоками не используются для оценки.")
    lines.append("")

    lines.append("ЛУЧШАЯ ОСНОВНАЯ МОДЕЛЬ")
    lines.append(
        "Конфигурация: " + str(best_primary["configuration"])
    )
    lines.append(
        "Блок признаков: " + str(best_primary["feature_block"])
    )
    lines.append(
        "Модель: " + str(best_primary["model"])
    )
    lines.append(
        "MAE: " + str(best_primary["mae"])
    )
    lines.append(
        "Медианная ошибка: " + str(best_primary["median_error"])
    )
    lines.append(
        "Максимальная ошибка: " + str(best_primary["max_error"])
    )
    lines.append(
        "Точность: " + str(best_primary["accuracy"])
    )
    lines.append("")

    lines.append("ЛУЧШАЯ МОДЕЛЬ ПО БЛОКАМ ПРИЗНАКОВ")

    for block_name in ALL_BLOCKS:
        subset = observed_metrics[
            observed_metrics["feature_block"] == block_name
        ]

        best = subset.iloc[0]

        lines.append(
            block_name
            + ": MAE = "
            + str(best["mae"])
            + ", точность = "
            + str(best["accuracy"])
            + ", модель = "
            + str(best["model"])
        )

    lines.append("")

    lines.append("ДИАГНОСТИЧЕСКАЯ МОДЕЛЬ РАЗНОСТИ ТОЧЕК")
    lines.append(
        "Конфигурация: "
        + str(best_diagnostic["configuration"])
    )
    lines.append(
        "MAE: " + str(best_diagnostic["mae"])
    )
    lines.append(
        "Точность: " + str(best_diagnostic["accuracy"])
    )
    lines.append(
        "Эта модель использует координаты Qb - Qa "
        "и не считается самостоятельным обходом ECDLP."
    )

    lines.append("")

    lines.append("ПЕРЕСТАНОВОЧНЫЙ ТЕСТ ОСНОВНЫХ МОДЕЛЕЙ")
    lines.append(
        "Средняя лучшая MAE нулевой модели: "
        + str(permutation_statistics["primary_null_mean_mae"])
    )
    lines.append(
        "Минимальная лучшая MAE нулевой модели: "
        + str(permutation_statistics["primary_null_min_mae"])
    )
    lines.append(
        "Наблюдаемая лучшая MAE: "
        + str(best_primary["mae"])
    )
    lines.append(
        "p-value MAE: "
        + str(permutation_statistics["primary_p_value"])
    )
    lines.append(
        "p-value точности: "
        + str(permutation_statistics["primary_accuracy_p_value"])
    )

    lines.append("")

    lines.append("ПЕРЕСТАНОВОЧНЫЙ ТЕСТ ДИАГНОСТИЧЕСКОЙ МОДЕЛИ")
    lines.append(
        "Средняя лучшая MAE нулевой модели: "
        + str(permutation_statistics["diagnostic_null_mean_mae"])
    )
    lines.append(
        "Минимальная лучшая MAE нулевой модели: "
        + str(permutation_statistics["diagnostic_null_min_mae"])
    )
    lines.append(
        "p-value MAE: "
        + str(permutation_statistics["diagnostic_p_value"])
    )
    lines.append(
        "p-value точности: "
        + str(permutation_statistics["diagnostic_accuracy_p_value"])
    )

    lines.append("")

    lines.append("ВРЕМЯ ВЫПОЛНЕНИЯ")
    lines.append(
        "Секунд: " + str(round(elapsed_seconds, 3))
    )

    lines.append("")
    lines.append("МЕТОДОЛОГИЧЕСКОЕ ПРИМЕЧАНИЕ")
    lines.append(
        "Целевая метка не включается в признаки."
    )
    lines.append(
        "Идентификаторы точек используются только "
        "для построения пар, разбиения и аудита."
    )
    lines.append(
        "Перестановки выполняются на уровне точек, "
        "а не отдельных пар."
    )
    lines.append(
        "Лучший результат выбирается среди всех конфигураций "
        "до сравнения с нулевым распределением."
    )
    lines.append(
        "Результат на группе порядка 73 не является "
        "доказательством эффективности на secp256k1."
    )

    path = os.path.join(
        output_dir,
        "summary.txt"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:
        file.write("\n".join(lines))

    return path


# ============================================================
# ОСНОВНОЙ ЗАПУСК
# ============================================================

def run_experiment(output_dir):
    start_time = time.perf_counter()

    print_section("ЭКСПЕРИМЕНТ C15")
    print("Исследование относительного скалярного расстояния.")
    print("Геометрическое представление пары точек.")
    print("")

    print("Параметры кривой:")
    print("Поле:", FIELD_P)
    print("Кривая: y^2 = x^3 + 7 mod 499")
    print("Генератор:", GENERATOR)
    print("Порядок группы:", GROUP_ORDER)

    print_section("ПОСТРОЕНИЕ ГРУППЫ")

    points = build_group()

    print("Количество ненулевых точек:", len(points))
    print("Проверка принадлежности кривой пройдена.")
    print("Проверка порядка генератора пройдена.")

    print_section("ПОСТРОЕНИЕ ПАР")

    pair_a, pair_b, feature_arrays = build_dataset(
        points
    )

    print("Количество ориентированных пар:", len(pair_a))

    for block_name in ALL_BLOCKS:
        print(
            "Признаков в блоке",
            block_name,
            ":",
            feature_arrays[block_name].shape[1]
        )

    print_section("РАЗБИЕНИЕ ДАННЫХ")

    folds = build_folds(
        pair_a,
        pair_b
    )

    for fold in folds:
        print(
            "Блок",
            fold["fold_id"],
            ": точек теста =",
            len(fold["test_nodes"]),
            ", пар обучения =",
            len(fold["train_indices"]),
            ", пар теста =",
            len(fold["test_indices"])
        )

    print_section("НАБЛЮДАЕМЫЕ МОДЕЛИ")

    observed_label_map = {
        scalar: scalar
        for scalar in range(1, GROUP_ORDER)
    }

    observed_metrics, observed_predictions = evaluate_labels(
        pair_a,
        pair_b,
        feature_arrays,
        folds,
        observed_label_map,
        collect_predictions=True
    )

    primary_metrics = observed_metrics[
        observed_metrics["feature_block"].isin(
            PRIMARY_BLOCKS
        )
    ]

    diagnostic_metrics = observed_metrics[
        observed_metrics["feature_block"].isin(
            DIAGNOSTIC_BLOCKS
        )
    ]

    best_primary = primary_metrics.iloc[0]
    best_diagnostic = diagnostic_metrics.iloc[0]

    print("Лучшие основные конфигурации:")

    for block_name in PRIMARY_BLOCKS:
        subset = observed_metrics[
            observed_metrics["feature_block"] == block_name
        ]

        best = subset.iloc[0]

        print(
            block_name,
            ": MAE =",
            round(best["mae"], 6),
            ", точность =",
            round(best["accuracy"], 6),
            ", модель =",
            best["model"]
        )

    print("")
    print("Лучшая основная конфигурация:")
    print(best_primary["configuration"])
    print("MAE:", best_primary["mae"])
    print("Точность:", best_primary["accuracy"])

    print("")
    print("Диагностическая модель разности:")
    print(best_diagnostic["configuration"])
    print("MAE:", best_diagnostic["mae"])
    print("Точность:", best_diagnostic["accuracy"])

    print_section("ПЕРЕСТАНОВОЧНЫЙ ТЕСТ")

    permutation_frame, permutation_statistics = (
        run_permutation_test(
            pair_a,
            pair_b,
            feature_arrays,
            folds,

            float(best_primary["mae"]),
            float(primary_metrics["accuracy"].max()),

            float(best_diagnostic["mae"]),
            float(diagnostic_metrics["accuracy"].max())
        )
    )

    print_section("РЕЗУЛЬТАТЫ СТАТИСТИЧЕСКОГО КОНТРОЛЯ")

    for key, value in permutation_statistics.items():
        print(key, ":", value)

    elapsed_seconds = (
        time.perf_counter() - start_time
    )

    print_section("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")

    save_dataset(
        output_dir,
        pair_a,
        pair_b,
        points,
        feature_arrays
    )

    save_feature_definitions(
        output_dir,
        feature_arrays
    )

    save_configuration(
        output_dir
    )

    observed_metrics.to_csv(
        os.path.join(
            output_dir,
            "cv_results.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    observed_predictions.to_csv(
        os.path.join(
            output_dir,
            "predictions_observed.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    permutation_frame.to_csv(
        os.path.join(
            output_dir,
            "permutation_results.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    with open(
        os.path.join(
            output_dir,
            "permutation_statistics.json"
        ),
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            permutation_statistics,
            file,
            ensure_ascii=False,
            indent=4
        )

    save_summary(
        output_dir,
        points,
        pair_a,
        folds,
        observed_metrics,
        permutation_statistics,
        elapsed_seconds
    )

    print("Сохранение таблиц завершено.")
    print("Время выполнения:", round(elapsed_seconds, 3), "секунд")


# ============================================================
# СОЗДАНИЕ ПОЛНОГО ZIP-АРХИВА
# ============================================================

def create_archive(output_dir, archive_path):
    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6
    ) as archive:

        for root, directories, files in os.walk(output_dir):
            directories.sort()
            files.sort()

            for file_name in files:
                full_path = os.path.join(
                    root,
                    file_name
                )

                relative_path = os.path.relpath(
                    full_path,
                    output_dir
                )

                archive.write(
                    full_path,
                    arcname=os.path.join(
                        os.path.basename(output_dir),
                        relative_path
                    )
                )


def main():
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    base_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    output_name = (
        RESULT_PREFIX
        + "_results_"
        + timestamp
    )

    output_dir = os.path.join(
        base_dir,
        output_name
    )

    archive_path = os.path.join(
        base_dir,
        RESULT_PREFIX
        + "_"
        + timestamp
        + ".zip"
    )

    os.makedirs(
        output_dir,
        exist_ok=False
    )

    log_path = os.path.join(
        output_dir,
        "run.log"
    )

    try:
        with open(
            log_path,
            "w",
            encoding="utf-8"
        ) as log_file:

            tee = Tee(
                sys.stdout,
                log_file
            )

            with redirect_stdout(tee):
                run_experiment(output_dir)

        create_archive(
            output_dir,
            archive_path
        )

        archive_message = (
            "Полный ZIP-архив создан: "
            + archive_path
        )

        print("")
        print(archive_message)

        with open(
            log_path,
            "a",
            encoding="utf-8"
        ) as log_file:
            log_file.write("\n")
            log_file.write(archive_message)
            log_file.write("\n")

        create_archive(
            output_dir,
            archive_path
        )

        print("")
        print("ЭКСПЕРИМЕНТ C15 ЗАВЕРШЁН.")
        print("Каталог результатов:", output_dir)
        print("Архив:", archive_path)

    except Exception:
        print("")
        print("ОШИБКА ВО ВРЕМЯ ВЫПОЛНЕНИЯ ЭКСПЕРИМЕНТА.")
        traceback.print_exc()

        error_path = os.path.join(
            output_dir,
            "error.txt"
        )

        with open(
            error_path,
            "w",
            encoding="utf-8"
        ) as file:
            file.write(
                traceback.format_exc()
            )

        try:
            create_archive(
                output_dir,
                archive_path
            )

            print("Архив частичных результатов создан:")
            print(archive_path)

        except Exception:
            print("Не удалось создать архив частичных результатов.")

        raise


if __name__ == "__main__":
    main()