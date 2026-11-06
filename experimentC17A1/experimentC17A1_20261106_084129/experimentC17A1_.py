
# -*- coding: utf-8 -*-

import csv
import json
import math
import os
import random
import shutil
import sys
import traceback
import zipfile

from datetime import datetime
from pathlib import Path

import numpy as np

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# НАСТРОЙКИ ЭКСПЕРИМЕНТА
# ============================================================

FIELD_P = 499
CURVE_A = 0
CURVE_B = 7

GENERATOR = (3, 212)
GROUP_ORDER = 73

RANDOM_SEED = 1701
FOLD_COUNT = 6

PERMUTATION_COUNT = 39

EXTRA_TREES_COUNT = 100
KNN_NEIGHBORS = 7

OUTPUT_PREFIX = "experimentC17A1"


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
            "Ошибка: числитель подъёма не делится на модуль поля."
        )

    return numerator // FIELD_P


# ============================================================
# ПОСТРОЕНИЕ ТОЧЕК ГРУППЫ
# ============================================================

def build_points():
    points = []
    current = None

    for scalar in range(1, GROUP_ORDER):
        current = point_add(current, GENERATOR)

        if current is None:
            raise ValueError(
                "Генератор достиг бесконечно удалённой точки слишком рано."
            )

        if not is_on_curve(current):
            raise ValueError(
                "Обнаружена точка, не принадлежащая эллиптической кривой."
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
            "Порядок генератора не совпадает с GROUP_ORDER."
        )

    if len(set((item["x"], item["y"]) for item in points)) != len(points):
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

    abs_dx = abs(dx)
    abs_dy = abs(dy)

    distance_squared = dx * dx + dy * dy
    distance = math.sqrt(distance_squared)

    midpoint_x = (x1 + x2) / 2.0
    midpoint_y = (y1 + y2) / 2.0

    angle = math.atan2(dy, dx)

    # Контроль: только исходные координаты двух точек.
    xy_control = [
        x1,
        y1,
        x2,
        y2
    ]

    # Геометрия пары без целочисленного подъёма.
    euclidean_geometry = [
        dx,
        dy,
        abs_dx,
        abs_dy,
        distance,
        distance_squared,
        midpoint_x,
        midpoint_y,
        math.cos(angle),
        math.sin(angle)
    ]

    # Целочисленный подъём координат обеих точек.
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
# ПОСТРОЕНИЕ ВСЕХ ОРИЕНТИРОВАННЫХ ПАР
# ============================================================

def build_pairs(points):
    pairs = []

    for index_a in range(len(points)):
        for index_b in range(len(points)):
            if index_a == index_b:
                continue

            point_a = points[index_a]
            point_b = points[index_b]

            features = build_feature_sets(point_a, point_b)

            pairs.append({
                "index_a": index_a,
                "index_b": index_b,
                "point_a": point_a["point_id"],
                "point_b": point_b["point_id"],
                "features": features
            })

    return pairs


# ============================================================
# РАЗДЕЛЕНИЕ ТОЧЕК НА НЕЗАВИСИМЫЕ ГРУППЫ
# ============================================================

def build_folds(points):
    # Разделение строится только по координатам.
    # Истинные скаляры при построении групп не используются.

    sorted_indices = sorted(
        range(len(points)),
        key=lambda index: (
            points[index]["x"],
            points[index]["y"]
        )
    )

    folds = [[] for _ in range(FOLD_COUNT)]

    for rank, point_index in enumerate(sorted_indices):
        folds[rank % FOLD_COUNT].append(point_index)

    return folds


def build_fold_pairs(pairs, folds):
    fold_data = []

    for fold_index, test_indices in enumerate(folds):
        test_set = set(test_indices)

        train_pair_indices = []
        test_pair_indices = []

        for pair_index, pair in enumerate(pairs):
            index_a = pair["index_a"]
            index_b = pair["index_b"]

            a_is_test = index_a in test_set
            b_is_test = index_b in test_set

            if a_is_test and b_is_test:
                test_pair_indices.append(pair_index)

            elif not a_is_test and not b_is_test:
                train_pair_indices.append(pair_index)

        fold_data.append({
            "fold": fold_index + 1,
            "test_points": sorted(test_indices),
            "train_pairs": train_pair_indices,
            "test_pairs": test_pair_indices
        })

    return fold_data


# ============================================================
# ЦЕЛЕВЫЕ ЗНАЧЕНИЯ
# ============================================================

def build_targets(pairs, scalar_labels):
    targets = []

    for pair in pairs:
        scalar_a = scalar_labels[pair["index_a"]]
        scalar_b = scalar_labels[pair["index_b"]]

        difference = (scalar_b - scalar_a) % GROUP_ORDER

        targets.append(difference)

    return np.asarray(targets, dtype=int)


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
        "sample_count": int(len(actual))
    }


# ============================================================
# МОДЕЛИ
# ============================================================

def create_model(model_name, random_seed):
    if model_name == "ExtraTrees":
        return ExtraTreesRegressor(
            n_estimators=EXTRA_TREES_COUNT,
            max_features=1.0,
            min_samples_leaf=2,
            random_state=random_seed,
            n_jobs=-1
        )

    if model_name == "KNN":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor(
                n_neighbors=KNN_NEIGHBORS,
                weights="distance",
                p=2
            ))
        ])

    raise ValueError("Неизвестная модель.")


# ============================================================
# ОДИН ПРОХОД ОБУЧЕНИЯ И ПРОВЕРКИ
# ============================================================

def evaluate_model(
    pairs,
    targets,
    fold_data,
    feature_name,
    model_name,
    random_seed,
    save_predictions=False
):
    all_actual = []
    all_predicted = []

    fold_results = []
    prediction_rows = []

    for fold in fold_data:
        train_indices = fold["train_pairs"]
        test_indices = fold["test_pairs"]

        if not train_indices or not test_indices:
            continue

        x_train = np.asarray([
            pairs[index]["features"][feature_name]
            for index in train_indices
        ], dtype=float)

        x_test = np.asarray([
            pairs[index]["features"][feature_name]
            for index in test_indices
        ], dtype=float)

        y_train = targets[train_indices]
        y_test = targets[test_indices]

        model = create_model(
            model_name,
            random_seed + fold["fold"]
        )

        model.fit(x_train, y_train)

        predicted = model.predict(x_test)

        metrics = calculate_metrics(y_test, predicted)

        fold_results.append({
            "fold": fold["fold"],
            "feature": feature_name,
            "model": model_name,
            **metrics
        })

        all_actual.extend(y_test.tolist())
        all_predicted.extend(predicted.tolist())

        if save_predictions:
            rounded = np.rint(predicted).astype(int) % GROUP_ORDER

            for local_index, pair_index in enumerate(test_indices):
                pair = pairs[pair_index]

                prediction_rows.append({
                    "fold": fold["fold"],
                    "pair_index": pair_index,
                    "point_a": pair["point_a"],
                    "point_b": pair["point_b"],
                    "actual": int(y_test[local_index]),
                    "predicted_float": float(predicted[local_index]),
                    "predicted_integer": int(rounded[local_index]),
                    "correct": int(
                        rounded[local_index] == y_test[local_index]
                    )
                })

    aggregate = calculate_metrics(
        all_actual,
        all_predicted
    )

    return aggregate, fold_results, prediction_rows


# ============================================================
# CSV: ТОЧКИ
# ============================================================

def save_points_csv(points, output_path):
    with open(output_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "point_id",
                "scalar",
                "x",
                "y",
                "lift"
            ]
        )

        writer.writeheader()

        for point in points:
            writer.writerow(point)


# ============================================================
# CSV: ВСЕ ПАРЫ И ПРИЗНАКИ
# ============================================================

def save_pairs_csv(pairs, targets, output_path):
    feature_names = [
        "x1", "y1", "x2", "y2",
        "dx", "dy", "abs_dx", "abs_dy",
        "distance", "distance_squared",
        "midpoint_x", "midpoint_y",
        "angle_cos", "angle_sin",
        "lift_a", "lift_b",
        "lift_difference", "lift_sum",
        "lift_absolute_difference"
    ]

    with open(output_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)

        writer.writerow([
            "pair_index",
            "point_a",
            "point_b",
            "scalar_difference",
            *feature_names
        ])

        for index, pair in enumerate(pairs):
            features = pair["features"]

            values = (
                features["xy_control"]
                + features["euclidean_geometry"]
                + features["lift_geometry"]
            )

            writer.writerow([
                index,
                pair["point_a"],
                pair["point_b"],
                int(targets[index]),
                *values
            ])


# ============================================================
# CSV: РЕЗУЛЬТАТЫ
# ============================================================

def save_dict_rows(rows, output_path):
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# ПЕРЕСТАНОВОЧНЫЙ ТЕСТ
# ============================================================

def run_permutation_test(
    points,
    pairs,
    fold_data,
    observed_results
):
    rng = random.Random(RANDOM_SEED + 9000)

    permutation_rows = []

    observed_combined = observed_results[
        ("combined_geometry", "ExtraTrees")
    ]["mae_circular"]

    observed_control = observed_results[
        ("xy_control", "ExtraTrees")
    ]["mae_circular"]

    observed_improvement = observed_control - observed_combined

    original_labels = [
        point["scalar"]
        for point in points
    ]

    print()
    print("Начинается перестановочный тест.")
    print("Количество перестановок:", PERMUTATION_COUNT)

    for permutation_index in range(PERMUTATION_COUNT):
        shuffled_labels = original_labels.copy()
        rng.shuffle(shuffled_labels)

        targets = build_targets(
            pairs,
            shuffled_labels
        )

        combined_metrics, _, _ = evaluate_model(
            pairs,
            targets,
            fold_data,
            "combined_geometry",
            "ExtraTrees",
            RANDOM_SEED + 20000 + permutation_index
        )

        control_metrics, _, _ = evaluate_model(
            pairs,
            targets,
            fold_data,
            "xy_control",
            "ExtraTrees",
            RANDOM_SEED + 30000 + permutation_index
        )

        improvement = (
            control_metrics["mae_circular"]
            - combined_metrics["mae_circular"]
        )

        row = {
            "permutation": permutation_index + 1,
            "combined_mae_circular": combined_metrics["mae_circular"],
            "combined_accuracy": combined_metrics["accuracy"],
            "control_mae_circular": control_metrics["mae_circular"],
            "control_accuracy": control_metrics["accuracy"],
            "improvement": improvement
        }

        permutation_rows.append(row)

        print(
            "Перестановка",
            permutation_index + 1,
            "из",
            PERMUTATION_COUNT,
            "| ошибка:",
            round(combined_metrics["mae_circular"], 4)
        )

    null_combined = [
        row["combined_mae_circular"]
        for row in permutation_rows
    ]

    null_improvements = [
        row["improvement"]
        for row in permutation_rows
    ]

    p_value_error = (
        1 + sum(
            value <= observed_combined
            for value in null_combined
        )
    ) / (PERMUTATION_COUNT + 1)

    p_value_improvement = (
        1 + sum(
            value >= observed_improvement
            for value in null_improvements
        )
    ) / (PERMUTATION_COUNT + 1)

    summary = {
        "observed_combined_mae_circular": observed_combined,
        "observed_control_mae_circular": observed_control,
        "observed_improvement": observed_improvement,
        "null_mean_combined_mae": float(np.mean(null_combined)),
        "null_min_combined_mae": float(np.min(null_combined)),
        "null_max_combined_mae": float(np.max(null_combined)),
        "null_mean_improvement": float(np.mean(null_improvements)),
        "p_value_combined_error": float(p_value_error),
        "p_value_improvement": float(p_value_improvement),
        "permutation_count": PERMUTATION_COUNT
    }

    return permutation_rows, summary


# ============================================================
# ТЕКСТОВЫЙ ОТЧЁТ
# ============================================================

def save_report(
    output_path,
    points,
    pairs,
    model_results,
    permutation_summary
):
    lines = []

    lines.append("ЭКСПЕРИМЕНТ C17A1")
    lines.append("=" * 65)
    lines.append("Геометрическое представление и аппроксимация скаляра")
    lines.append("")

    lines.append("ПАРАМЕТРЫ КРИВОЙ")
    lines.append(f"Поле: p = {FIELD_P}")
    lines.append(f"Кривая: y^2 = x^3 + {CURVE_B} mod {FIELD_P}")
    lines.append(f"Генератор: {GENERATOR}")
    lines.append(f"Порядок группы: {GROUP_ORDER}")
    lines.append(f"Количество точек: {len(points)}")
    lines.append(f"Количество ориентированных пар: {len(pairs)}")
    lines.append("")

    lines.append("ЦЕЛОЧИСЛЕННЫЙ ПОДЪЁМ")
    lines.append(
        "K(x,y) = (x^3 + 7 - y^2) / 499"
    )
    lines.append(
        "Подъём вычисляется по каноническим координатам точек."
    )
    lines.append(
        "Он не является скаляром и не интерпретируется как скаляр."
    )
    lines.append("")

    lines.append("РЕЗУЛЬТАТЫ МОДЕЛЕЙ")
    lines.append("-" * 65)

    for key, metrics in model_results.items():
        feature_name, model_name = key

        lines.append(
            f"{feature_name} | {model_name}"
        )

        lines.append(
            f"  Циклическая MAE: {metrics['mae_circular']:.6f}"
        )

        lines.append(
            f"  Линейная MAE: {metrics['mae_linear']:.6f}"
        )

        lines.append(
            f"  Точная точность: {metrics['accuracy']:.6f}"
        )

        lines.append(
            f"  Точных ответов: {metrics['exact_count']} "
            f"из {metrics['sample_count']}"
        )

        lines.append("")

    lines.append("ПЕРЕСТАНОВОЧНЫЙ ТЕСТ")
    lines.append("-" * 65)

    for key, value in permutation_summary.items():
        lines.append(f"{key}: {value}")

    lines.append("")
    lines.append("МЕТОДОЛОГИЧЕСКИЕ ОГРАНИЧЕНИЯ")
    lines.append("-" * 65)

    lines.append(
        "1. Обучающие и тестовые пары не имеют общих точек."
    )

    lines.append(
        "2. Разбиение точек выполнено по координатам, "
        "без использования скаляров."
    )

    lines.append(
        "3. Скаляры и результаты групповых операций "
        "не входят в признаки моделей."
    )

    lines.append(
        "4. Групповая арифметика используется только "
        "для формирования правильных ответов."
    )

    lines.append(
        "5. Перестановочный тест проверяет зависимость "
        "результата от соответствия координат и скаляров."
    )

    lines.append(
        "6. Эксперимент на игрушечной кривой не доказывает "
        "наличие аналогичной закономерности в secp256k1."
    )

    lines.append("")
    lines.append("ИНТЕРПРЕТАЦИЯ")
    lines.append("-" * 65)

    lines.append(
        "Положительный результат требует не только низкой ошибки, "
        "но и превосходства над контрольной моделью и "
        "перестановочным распределением."
    )

    lines.append(
        "Если наблюдаемая ошибка сопоставима с перестановочным "
        "контролем, оснований утверждать о выявлении закономерности нет."
    )

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


# ============================================================
# СОХРАНЕНИЕ JSON
# ============================================================

def save_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=4
        )


# ============================================================
# СОЗДАНИЕ ZIP-АРХИВА
# ============================================================

def create_archive(output_directory, archive_path):
    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED
    ) as archive:

        for file_path in output_directory.rglob("*"):
            if file_path.is_file() and file_path != archive_path:
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_directory = Path(
        f"{OUTPUT_PREFIX}_{timestamp}"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 65)
    print("ЭКСПЕРИМЕНТ C17A1")
    print("Геометрическая аппроксимация относительного скаляра")
    print("=" * 65)

    print("Проверка генератора и построение точек...")

    points = build_points()

    print("Количество точек:", len(points))

    pairs = build_pairs(points)

    print("Количество ориентированных пар:", len(pairs))

    folds = build_folds(points)
    fold_data = build_fold_pairs(pairs, folds)

    print("Построено независимых групп:", len(fold_data))

    # Истинные скаляры используются только как целевые значения.
    original_labels = [
        point["scalar"]
        for point in points
    ]

    targets = build_targets(
        pairs,
        original_labels
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
            "folds": [
                {
                    "fold": item["fold"],
                    "test_points": item["test_points"],
                    "train_pair_count": len(item["train_pairs"]),
                    "test_pair_count": len(item["test_pairs"])
                }
                for item in fold_data
            ],
            "feature_sets": {
                "xy_control": 4,
                "euclidean_geometry": 10,
                "lift_geometry": 5,
                "combined_geometry": 19
            },
            "permutation_count": PERMUTATION_COUNT,
            "random_seed": RANDOM_SEED
        },
        output_directory / "experiment_config.json"
    )

    feature_names = [
        "xy_control",
        "euclidean_geometry",
        "lift_geometry",
        "combined_geometry"
    ]

    model_names = [
        "ExtraTrees",
        "KNN"
    ]

    model_results = {}
    fold_results = []
    prediction_rows = []

    print()
    print("Начинается обучение моделей...")

    for feature_name in feature_names:
        for model_name in model_names:
            print(
                "Признаки:",
                feature_name,
                "| Модель:",
                model_name
            )

            metrics, current_folds, current_predictions = evaluate_model(
                pairs,
                targets,
                fold_data,
                feature_name,
                model_name,
                RANDOM_SEED,
                save_predictions=True
            )

            model_results[
                (feature_name, model_name)
            ] = metrics

            fold_results.extend(current_folds)
            prediction_rows.extend(current_predictions)

    save_dict_rows(
        fold_results,
        output_directory / "fold_metrics.csv"
    )

    save_dict_rows(
        prediction_rows,
        output_directory / "predictions.csv"
    )

    serializable_results = {}

    for key, value in model_results.items():
        serializable_results[
            key[0] + "__" + key[1]
        ] = value

    save_json(
        serializable_results,
        output_directory / "model_metrics.json"
    )

    print()
    print("Основные модели проверены.")
    print("Запускается перестановочный тест...")

    permutation_rows, permutation_summary = run_permutation_test(
        points,
        pairs,
        fold_data,
        model_results
    )

    save_dict_rows(
        permutation_rows,
        output_directory / "permutation_results.csv"
    )

    save_json(
        permutation_summary,
        output_directory / "permutation_summary.json"
    )

    save_report(
        output_directory / "report.txt",
        points,
        pairs,
        model_results,
        permutation_summary
    )

    # Сохраняем копию исполняемого файла в результатах.
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
    print("=" * 65)
    print("ЭКСПЕРИМЕНТ ЗАВЕРШЁН")
    print("=" * 65)

    print("Результаты сохранены в:", output_directory.resolve())
    print("ZIP-архив:", archive_path.resolve())

    print()
    print("СВОДКА ОСНОВНЫХ РЕЗУЛЬТАТОВ")

    for key, metrics in model_results.items():
        print(
            key[0],
            "|",
            key[1],
            "| циклическая MAE:",
            round(metrics["mae_circular"], 6),
            "| точность:",
            round(metrics["accuracy"], 6)
        )

    print()
    print("ПЕРЕСТАНОВОЧНЫЙ ТЕСТ")

    for key, value in permutation_summary.items():
        print(key, ":", value)

    print()
    print("Полный архив эксперимента создан.")


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print()
        print("ОШИБКА ВЫПОЛНЕНИЯ ЭКСПЕРИМЕНТА")
        print(str(error))
        traceback.print_exc()
        sys.exit(1)
