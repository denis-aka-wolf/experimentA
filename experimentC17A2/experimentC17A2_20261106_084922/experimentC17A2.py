
# -*- coding: utf-8 -*-

import csv
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

from sklearn.ensemble import ExtraTreesRegressor


# ============================================================
# НАСТРОЙКИ ЭКСПЕРИМЕНТА
# ============================================================

FIELD_P = 499
CURVE_A = 0
CURVE_B = 7

GENERATOR = (3, 212)
GROUP_ORDER = 73

RANDOM_SEED = 1702

FOLD_COUNT = 6
REPEAT_COUNT = 10

PERMUTATION_COUNT = 999

MAIN_TREES = 80
PERMUTATION_TREES = 32

OUTPUT_PREFIX = "experimentC17A2"


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
        numerator = (3 * x1 * x1 + CURVE_A) % FIELD_P
        denominator = (2 * y1) % FIELD_P
    else:
        numerator = (y2 - y1) % FIELD_P
        denominator = (x2 - x1) % FIELD_P

    if denominator == 0:
        return None

    inverse = pow(denominator, -1, FIELD_P)
    slope = (numerator * inverse) % FIELD_P

    x3 = (slope * slope - x1 - x2) % FIELD_P
    y3 = (slope * (x1 - x3) - y1) % FIELD_P

    return x3, y3


def scalar_multiply(scalar, point):
    result = None
    current = point
    value = scalar

    while value > 0:
        if value & 1:
            result = point_add(result, current)

        current = point_add(current, current)
        value >>= 1

    return result


def is_on_curve(point):
    if point is None:
        return True

    x, y = point

    left = (y * y) % FIELD_P
    right = (x * x * x + CURVE_B) % FIELD_P

    return left == right


# ============================================================
# ЦЕЛОЧИСЛЕННЫЙ ПОДЪЁМ КООРДИНАТ
# ============================================================

def calculate_integer_lift(point):
    x, y = point

    numerator = x ** 3 + CURVE_B - y ** 2

    if numerator % FIELD_P != 0:
        raise ValueError(
            "Ошибка вычисления целочисленного подъёма."
        )

    return numerator // FIELD_P


# ============================================================
# ПОСТРОЕНИЕ ТОЧЕК
# ============================================================

def build_points():
    points = []
    current = None

    for scalar in range(1, GROUP_ORDER):
        current = point_add(current, GENERATOR)

        if current is None:
            raise ValueError(
                "Генератор достиг бесконечно удалённой точки."
            )

        if not is_on_curve(current):
            raise ValueError(
                "Точка не принадлежит заданной кривой."
            )

        points.append({
            "point_id": scalar - 1,
            "scalar": scalar,
            "x": current[0],
            "y": current[1],
            "lift": calculate_integer_lift(current)
        })

    if point_add(current, GENERATOR) is not None:
        raise ValueError(
            "Порядок генератора не равен заданному порядку группы."
        )

    if len(set((p["x"], p["y"]) for p in points)) != GROUP_ORDER - 1:
        raise ValueError(
            "Обнаружены повторяющиеся точки."
        )

    return points


# ============================================================
# ГЕОМЕТРИЧЕСКИЕ ПРИЗНАКИ
# ============================================================

def build_feature_sets(point_a, point_b):
    x1 = point_a["x"]
    y1 = point_a["y"]

    x2 = point_b["x"]
    y2 = point_b["y"]

    k1 = point_a["lift"]
    k2 = point_b["lift"]

    dx = x2 - x1
    dy = y2 - y1

    distance_squared = dx * dx + dy * dy
    distance = math.sqrt(distance_squared)

    angle = math.atan2(dy, dx)

    midpoint_x = (x1 + x2) / 2.0
    midpoint_y = (y1 + y2) / 2.0

    xy_control = [
        x1,
        y1,
        x2,
        y2
    ]

    euclidean_geometry = [
        dx,
        dy,
        abs(dx),
        abs(dy),
        distance,
        distance_squared,
        midpoint_x,
        midpoint_y,
        math.cos(angle),
        math.sin(angle)
    ]

    lift_geometry = [
        k1,
        k2,
        k2 - k1,
        k1 + k2,
        abs(k2 - k1)
    ]

    combined_geometry = (
        xy_control
        + euclidean_geometry
        + lift_geometry
    )

    return {
        "xy_control": xy_control,
        "euclidean_geometry": euclidean_geometry,
        "lift_geometry": lift_geometry,
        "combined_geometry": combined_geometry
    }


# ============================================================
# ПОСТРОЕНИЕ ОРИЕНТИРОВАННЫХ ПАР
# ============================================================

def build_pairs(points):
    pairs = []

    for index_a in range(len(points)):
        for index_b in range(len(points)):
            if index_a == index_b:
                continue

            features = build_feature_sets(
                points[index_a],
                points[index_b]
            )

            pairs.append({
                "index_a": index_a,
                "index_b": index_b,
                "point_a": points[index_a]["point_id"],
                "point_b": points[index_b]["point_id"],
                "features": features
            })

    return pairs


# ============================================================
# ЦЕЛЕВЫЕ СКАЛЯРЫ
# ============================================================

def build_targets(pairs, scalar_labels):
    targets = []

    for pair in pairs:
        scalar_a = scalar_labels[pair["index_a"]]
        scalar_b = scalar_labels[pair["index_b"]]

        difference = (
            scalar_b - scalar_a
        ) % GROUP_ORDER

        targets.append(difference)

    return np.asarray(targets, dtype=int)


# ============================================================
# СЛУЧАЙНОЕ РАЗДЕЛЕНИЕ ТОЧЕК
# ============================================================

def build_folds(points, seed):
    rng = random.Random(seed)

    indices = list(range(len(points)))
    rng.shuffle(indices)

    folds = [
        []
        for _ in range(FOLD_COUNT)
    ]

    for index, point_index in enumerate(indices):
        folds[index % FOLD_COUNT].append(point_index)

    return folds


def build_fold_data(pairs, folds):
    result = []

    for fold_index, test_points in enumerate(folds):
        test_set = set(test_points)

        train_pairs = []
        test_pairs = []

        for pair_index, pair in enumerate(pairs):
            index_a = pair["index_a"]
            index_b = pair["index_b"]

            a_is_test = index_a in test_set
            b_is_test = index_b in test_set

            if a_is_test and b_is_test:
                test_pairs.append(pair_index)

            elif not a_is_test and not b_is_test:
                train_pairs.append(pair_index)

        result.append({
            "fold": fold_index + 1,
            "test_points": sorted(test_points),
            "train_pairs": train_pairs,
            "test_pairs": test_pairs
        })

    return result


# ============================================================
# МЕТРИКИ
# ============================================================

def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=int)
    predicted = np.asarray(predicted, dtype=float)

    rounded = np.rint(predicted).astype(int) % GROUP_ORDER

    linear_errors = np.abs(rounded - actual)

    circular_errors = np.minimum(
        linear_errors,
        GROUP_ORDER - linear_errors
    )

    exact = rounded == actual

    return {
        "mae_linear": float(np.mean(linear_errors)),
        "mae_circular": float(np.mean(circular_errors)),
        "accuracy": float(np.mean(exact)),
        "exact_count": int(np.sum(exact)),
        "zero_predictions": int(np.sum(rounded == 0)),
        "sample_count": int(len(actual))
    }


# ============================================================
# МОДЕЛЬ
# ============================================================

def create_model(tree_count, seed):
    return ExtraTreesRegressor(
        n_estimators=tree_count,
        max_features=1.0,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1
    )


# ============================================================
# ПОДГОТОВКА МАТРИЦ ПРИЗНАКОВ
# ============================================================

def prepare_feature_matrices(pairs):
    feature_names = [
        "xy_control",
        "euclidean_geometry",
        "lift_geometry",
        "combined_geometry"
    ]

    matrices = {}

    for feature_name in feature_names:
        matrices[feature_name] = np.asarray(
            [
                pair["features"][feature_name]
                for pair in pairs
            ],
            dtype=float
        )

    return matrices


# ============================================================
# ОБУЧЕНИЕ И ОЦЕНКА НА ОДНОМ РАЗБИЕНИИ
# ============================================================

def evaluate_feature(
    matrix,
    targets,
    train_indices,
    test_indices,
    tree_count,
    seed
):
    x_train = matrix[train_indices]
    x_test = matrix[test_indices]

    y_train = targets[train_indices]
    y_test = targets[test_indices]

    model = create_model(tree_count, seed)

    model.fit(x_train, y_train)

    predicted = model.predict(x_test)

    metrics = calculate_metrics(
        y_test,
        predicted
    )

    return metrics, predicted, y_test


# ============================================================
# ПОВТОРНАЯ ПРОВЕРКА УСТОЙЧИВОСТИ
# ============================================================

def run_repeated_validation(
    points,
    pairs,
    matrices,
    original_labels
):
    targets = build_targets(
        pairs,
        original_labels
    )

    fold_rows = []
    repeat_rows = []
    prediction_rows = []

    feature_names = [
        "xy_control",
        "euclidean_geometry",
        "lift_geometry",
        "combined_geometry"
    ]

    print()
    print("Начинается проверка устойчивости разбиений.")
    print("Количество повторений:", REPEAT_COUNT)

    for repeat_index in range(REPEAT_COUNT):
        seed = RANDOM_SEED + 100 + repeat_index

        folds = build_folds(
            points,
            seed
        )

        fold_data = build_fold_data(
            pairs,
            folds
        )

        repeat_predictions = {
            name: []
            for name in feature_names
        }

        repeat_actual = {
            name: []
            for name in feature_names
        }

        for fold in fold_data:
            train_indices = fold["train_pairs"]
            test_indices = fold["test_pairs"]

            for feature_index, feature_name in enumerate(feature_names):
                metrics, predicted, actual = evaluate_feature(
                    matrices[feature_name],
                    targets,
                    train_indices,
                    test_indices,
                    MAIN_TREES,
                    seed + fold["fold"] * 100 + feature_index
                )

                fold_rows.append({
                    "repeat": repeat_index + 1,
                    "fold": fold["fold"],
                    "feature": feature_name,
                    **metrics
                })

                repeat_predictions[feature_name].extend(
                    predicted.tolist()
                )

                repeat_actual[feature_name].extend(
                    actual.tolist()
                )

                rounded = (
                    np.rint(predicted).astype(int) % GROUP_ORDER
                )

                for local_index, pair_index in enumerate(test_indices):
                    pair = pairs[pair_index]

                    prediction_rows.append({
                        "repeat": repeat_index + 1,
                        "fold": fold["fold"],
                        "feature": feature_name,
                        "pair_index": pair_index,
                        "point_a": pair["point_a"],
                        "point_b": pair["point_b"],
                        "actual": int(actual[local_index]),
                        "predicted_float": float(predicted[local_index]),
                        "predicted_integer": int(rounded[local_index]),
                        "correct": int(
                            rounded[local_index] == actual[local_index]
                        )
                    })

        for feature_name in feature_names:
            metrics = calculate_metrics(
                repeat_actual[feature_name],
                repeat_predictions[feature_name]
            )

            repeat_rows.append({
                "repeat": repeat_index + 1,
                "feature": feature_name,
                **metrics
            })

        print(
            "Повторение",
            repeat_index + 1,
            "из",
            REPEAT_COUNT,
            "завершено."
        )

    return fold_rows, repeat_rows, prediction_rows


# ============================================================
# ФИКСИРОВАННОЕ РАЗБИЕНИЕ ДЛЯ ПЕРЕСТАНОВОЧНОГО ТЕСТА
# ============================================================

def build_fixed_holdout(points, pairs):
    folds = build_folds(
        points,
        RANDOM_SEED + 777
    )

    fold_data = build_fold_data(
        pairs,
        folds
    )

    # Используется первая тестовая группа.
    # Её состав фиксирован для всех перестановок.

    selected = fold_data[0]

    return selected


# ============================================================
# ПЕРЕСТАНОВОЧНЫЙ ТЕСТ
# ============================================================

def run_permutation_test(
    points,
    pairs,
    matrices,
    original_labels,
    fixed_holdout
):
    rng = random.Random(RANDOM_SEED + 50000)

    train_indices = fixed_holdout["train_pairs"]
    test_indices = fixed_holdout["test_pairs"]

    original_targets = build_targets(
        pairs,
        original_labels
    )

    observed_control, _, _ = evaluate_feature(
        matrices["xy_control"],
        original_targets,
        train_indices,
        test_indices,
        PERMUTATION_TREES,
        RANDOM_SEED + 60000
    )

    observed_euclidean, observed_predictions, observed_actual = evaluate_feature(
        matrices["euclidean_geometry"],
        original_targets,
        train_indices,
        test_indices,
        PERMUTATION_TREES,
        RANDOM_SEED + 60000
    )

    observed_improvement = (
        observed_control["mae_circular"]
        - observed_euclidean["mae_circular"]
    )

    permutation_rows = []

    null_euclidean_errors = []
    null_improvements = []

    print()
    print("Начинается перестановочный тест.")
    print("Количество перестановок:", PERMUTATION_COUNT)
    print("Размер фиксированной тестовой группы:", len(fixed_holdout["test_points"]))
    print("Количество тестовых пар:", len(test_indices))

    for permutation_index in range(PERMUTATION_COUNT):
        shuffled_labels = original_labels.copy()
        rng.shuffle(shuffled_labels)

        permuted_targets = build_targets(
            pairs,
            shuffled_labels
        )

        control_metrics, _, _ = evaluate_feature(
            matrices["xy_control"],
            permuted_targets,
            train_indices,
            test_indices,
            PERMUTATION_TREES,
            RANDOM_SEED + 60000
        )

        euclidean_metrics, _, _ = evaluate_feature(
            matrices["euclidean_geometry"],
            permuted_targets,
            train_indices,
            test_indices,
            PERMUTATION_TREES,
            RANDOM_SEED + 60000
        )

        improvement = (
            control_metrics["mae_circular"]
            - euclidean_metrics["mae_circular"]
        )

        null_euclidean_errors.append(
            euclidean_metrics["mae_circular"]
        )

        null_improvements.append(improvement)

        permutation_rows.append({
            "permutation": permutation_index + 1,
            "control_mae_circular": control_metrics["mae_circular"],
            "euclidean_mae_circular": euclidean_metrics["mae_circular"],
            "control_accuracy": control_metrics["accuracy"],
            "euclidean_accuracy": euclidean_metrics["accuracy"],
            "improvement": improvement
        })

        if (
            (permutation_index + 1) % 25 == 0
            or permutation_index + 1 == PERMUTATION_COUNT
        ):
            print(
                "Выполнено перестановок:",
                permutation_index + 1,
                "из",
                PERMUTATION_COUNT
            )

    p_value_error = (
        1 + sum(
            value <= observed_euclidean["mae_circular"]
            for value in null_euclidean_errors
        )
    ) / (PERMUTATION_COUNT + 1)

    p_value_improvement = (
        1 + sum(
            value >= observed_improvement
            for value in null_improvements
        )
    ) / (PERMUTATION_COUNT + 1)

    summary = {
        "fixed_test_fold": fixed_holdout["fold"],
        "test_point_count": len(fixed_holdout["test_points"]),
        "test_pair_count": len(test_indices),
        "permutation_count": PERMUTATION_COUNT,
        "observed_control_mae_circular": observed_control["mae_circular"],
        "observed_euclidean_mae_circular": observed_euclidean["mae_circular"],
        "observed_control_accuracy": observed_control["accuracy"],
        "observed_euclidean_accuracy": observed_euclidean["accuracy"],
        "observed_improvement": observed_improvement,
        "null_mean_euclidean_mae": float(np.mean(null_euclidean_errors)),
        "null_std_euclidean_mae": float(np.std(null_euclidean_errors)),
        "null_min_euclidean_mae": float(np.min(null_euclidean_errors)),
        "null_max_euclidean_mae": float(np.max(null_euclidean_errors)),
        "null_mean_improvement": float(np.mean(null_improvements)),
        "null_std_improvement": float(np.std(null_improvements)),
        "p_value_euclidean_error": float(p_value_error),
        "p_value_improvement_over_control": float(p_value_improvement)
    }

    holdout_predictions = []

    rounded = (
        np.rint(observed_predictions).astype(int) % GROUP_ORDER
    )

    for local_index, pair_index in enumerate(test_indices):
        pair = pairs[pair_index]

        holdout_predictions.append({
            "pair_index": pair_index,
            "point_a": pair["point_a"],
            "point_b": pair["point_b"],
            "actual": int(observed_actual[local_index]),
            "predicted_float": float(observed_predictions[local_index]),
            "predicted_integer": int(rounded[local_index]),
            "correct": int(
                rounded[local_index] == observed_actual[local_index]
            )
        })

    return (
        permutation_rows,
        summary,
        holdout_predictions
    )


# ============================================================
# СОХРАНЕНИЕ CSV
# ============================================================

def save_dict_rows(rows, output_path):
    if not rows:
        return

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)


def save_points_csv(points, output_path):
    save_dict_rows(
        points,
        output_path
    )


def save_pairs_csv(pairs, targets, output_path):
    rows = []

    for index, pair in enumerate(pairs):
        features = pair["features"]

        values = (
            features["xy_control"]
            + features["euclidean_geometry"]
            + features["lift_geometry"]
        )

        rows.append({
            "pair_index": index,
            "point_a": pair["point_a"],
            "point_b": pair["point_b"],
            "scalar_difference": int(targets[index]),
            "features": json.dumps(
                values,
                ensure_ascii=False
            )
        })

    save_dict_rows(
        rows,
        output_path
    )


# ============================================================
# СОХРАНЕНИЕ JSON
# ============================================================

def save_json(data, output_path):
    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=4
        )


# ============================================================
# ТЕКСТОВЫЙ ОТЧЁТ
# ============================================================

def save_report(
    output_path,
    points,
    pairs,
    repeat_rows,
    permutation_summary
):
    lines = []

    lines.append("ЭКСПЕРИМЕНТ C17A2")
    lines.append("=" * 70)
    lines.append("Геометрическая аппроксимация относительного скаляра")
    lines.append("")

    lines.append("ПАРАМЕТРЫ КРИВОЙ")
    lines.append(f"p = {FIELD_P}")
    lines.append(f"Кривая: y^2 = x^3 + {CURVE_B} mod {FIELD_P}")
    lines.append(f"Генератор: {GENERATOR}")
    lines.append(f"Порядок группы: {GROUP_ORDER}")
    lines.append(f"Количество точек: {len(points)}")
    lines.append(f"Количество пар: {len(pairs)}")
    lines.append("")

    lines.append("ПОВТОРНАЯ ПРОВЕРКА")
    lines.append("-" * 70)

    feature_names = [
        "xy_control",
        "euclidean_geometry",
        "lift_geometry",
        "combined_geometry"
    ]

    for feature_name in feature_names:
        selected = [
            row for row in repeat_rows
            if row["feature"] == feature_name
        ]

        mae_values = [
            row["mae_circular"]
            for row in selected
        ]

        accuracy_values = [
            row["accuracy"]
            for row in selected
        ]

        lines.append(feature_name)

        lines.append(
            f"  Средняя циклическая MAE: "
            f"{np.mean(mae_values):.6f}"
        )

        lines.append(
            f"  Стандартное отклонение MAE: "
            f"{np.std(mae_values):.6f}"
        )

        lines.append(
            f"  Лучшая MAE: {np.min(mae_values):.6f}"
        )

        lines.append(
            f"  Худшая MAE: {np.max(mae_values):.6f}"
        )

        lines.append(
            f"  Средняя точность: "
            f"{np.mean(accuracy_values):.6f}"
        )

        lines.append("")

    lines.append("ПЕРЕСТАНОВОЧНЫЙ ТЕСТ")
    lines.append("-" * 70)

    for key, value in permutation_summary.items():
        lines.append(f"{key}: {value}")

    lines.append("")
    lines.append("МЕТОДОЛОГИЯ")
    lines.append("-" * 70)

    lines.append(
        "1. Все пары ориентированные."
    )

    lines.append(
        "2. Обучающие и тестовые пары не имеют общих точек "
        "внутри одного разбиения."
    )

    lines.append(
        "3. Разбиение точек не использует скалярные метки."
    )

    lines.append(
        "4. Геометрические признаки строятся только "
        "по координатам и целочисленным подъёмам."
    )

    lines.append(
        "5. Групповые операции не используются при предсказании."
    )

    lines.append(
        "6. Перестановочный тест использует одно фиксированное "
        "разбиение для наблюдаемого результата и всех перестановок."
    )

    lines.append(
        "7. Перестановки перемешивают соответствие точек "
        "и скалярных меток."
    )

    lines.append(
        "8. Результаты игрушечной кривой не доказывают "
        "применимость метода к secp256k1."
    )

    lines.append("")
    lines.append("ИНТЕРПРЕТАЦИЯ")
    lines.append("-" * 70)

    lines.append(
        "Положительный результат требует устойчивого преимущества "
        "евклидовой геометрии над контролем по координатам."
    )

    lines.append(
        "Перестановочный тест должен показать, что наблюдаемое "
        "преимущество редко возникает при случайном соответствии "
        "точек и скалярных меток."
    )

    lines.append(
        "Отдельно необходимо проверить, улучшает ли целочисленный "
        "подъём результат относительно одной геометрии."
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:
        file.write("\n".join(lines))


# ============================================================
# ZIP-АРХИВ
# ============================================================

def create_archive(output_directory, archive_path):
    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED
    ) as archive:

        for file_path in output_directory.rglob("*"):
            if (
                file_path.is_file()
                and file_path != archive_path
            ):
                archive.write(
                    file_path,
                    arcname=file_path.relative_to(output_directory)
                )


# ============================================================
# ОСНОВНОЙ ЭКСПЕРИМЕНТ
# ============================================================

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_directory = Path(
        f"{OUTPUT_PREFIX}_{timestamp}"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("ЭКСПЕРИМЕНТ C17A2")
    print("Проверка устойчивости геометрической аппроксимации")
    print("=" * 70)

    print("Построение точек кривой...")

    points = build_points()

    print("Количество точек:", len(points))

    pairs = build_pairs(points)

    print("Количество ориентированных пар:", len(pairs))

    original_labels = [
        point["scalar"]
        for point in points
    ]

    targets = build_targets(
        pairs,
        original_labels
    )

    matrices = prepare_feature_matrices(
        pairs
    )

    save_points_csv(
        points,
        output_directory / "points.csv"
    )

    save_pairs_csv(
        pairs,
        targets,
        output_directory / "all_pairs.csv"
    )

    save_json(
        {
            "curve": {
                "p": FIELD_P,
                "a": CURVE_A,
                "b": CURVE_B,
                "generator": GENERATOR,
                "group_order": GROUP_ORDER
            },
            "repeat_count": REPEAT_COUNT,
            "fold_count": FOLD_COUNT,
            "permutation_count": PERMUTATION_COUNT,
            "main_trees": MAIN_TREES,
            "permutation_trees": PERMUTATION_TREES,
            "random_seed": RANDOM_SEED,
            "feature_dimensions": {
                "xy_control": 4,
                "euclidean_geometry": 10,
                "lift_geometry": 5,
                "combined_geometry": 19
            }
        },
        output_directory / "experiment_config.json"
    )

    print()
    print("Запуск повторной проверки...")

    fold_rows, repeat_rows, prediction_rows = run_repeated_validation(
        points,
        pairs,
        matrices,
        original_labels
    )

    save_dict_rows(
        fold_rows,
        output_directory / "fold_metrics.csv"
    )

    save_dict_rows(
        repeat_rows,
        output_directory / "repeat_metrics.csv"
    )

    save_dict_rows(
        prediction_rows,
        output_directory / "repeated_predictions.csv"
    )

    print()
    print("Повторная проверка завершена.")

    fixed_holdout = build_fixed_holdout(
        points,
        pairs
    )

    save_json(
        {
            "fold": fixed_holdout["fold"],
            "test_points": fixed_holdout["test_points"],
            "train_pair_count": len(fixed_holdout["train_pairs"]),
            "test_pair_count": len(fixed_holdout["test_pairs"])
        },
        output_directory / "fixed_holdout.json"
    )

    print()
    print("Запуск перестановочного теста...")

    (
        permutation_rows,
        permutation_summary,
        holdout_predictions
    ) = run_permutation_test(
        points,
        pairs,
        matrices,
        original_labels,
        fixed_holdout
    )

    save_dict_rows(
        permutation_rows,
        output_directory / "permutation_results.csv"
    )

    save_dict_rows(
        holdout_predictions,
        output_directory / "holdout_predictions.csv"
    )

    save_json(
        permutation_summary,
        output_directory / "permutation_summary.json"
    )

    save_report(
        output_directory / "report.txt",
        points,
        pairs,
        repeat_rows,
        permutation_summary
    )

    script_path = Path(__file__)

    if script_path.exists():
        shutil.copy2(
            script_path,
            output_directory / script_path.name
        )

    archive_path = output_directory / (
        f"{OUTPUT_PREFIX}_{timestamp}.zip"
    )

    create_archive(
        output_directory,
        archive_path
    )

    print()
    print("=" * 70)
    print("ЭКСПЕРИМЕНТ ЗАВЕРШЁН")
    print("=" * 70)

    print("Результаты:", output_directory.resolve())
    print("ZIP-архив:", archive_path.resolve())

    print()
    print("СРЕДНИЕ РЕЗУЛЬТАТЫ ПОВТОРНОЙ ПРОВЕРКИ")

    for feature_name in [
        "xy_control",
        "euclidean_geometry",
        "lift_geometry",
        "combined_geometry"
    ]:
        selected = [
            row for row in repeat_rows
            if row["feature"] == feature_name
        ]

        mean_mae = np.mean([
            row["mae_circular"]
            for row in selected
        ])

        mean_accuracy = np.mean([
            row["accuracy"]
            for row in selected
        ])

        print(
            feature_name,
            "| средняя циклическая MAE:",
            round(float(mean_mae), 6),
            "| средняя точность:",
            round(float(mean_accuracy), 6)
        )

    print()
    print("РЕЗУЛЬТАТЫ ПЕРЕСТАНОВОЧНОГО ТЕСТА")

    for key, value in permutation_summary.items():
        print(key, ":", value)

    print()
    print("Полный ZIP-архив создан.")


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print()
        print("ОШИБКА ВЫПОЛНЕНИЯ ЭКСПЕРИМЕНТА")
        print(str(error))
        traceback.print_exc()
        sys.exit(1)
