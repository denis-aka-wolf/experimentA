
# ============================================================
# EXPERIMENT C10
# ИСПРАВЛЕННАЯ ГЕОМЕТРИЧЕСКАЯ ТРИАНГУЛЯЦИЯ
#
# Исследовательская цель:
# Геометрическое представление точки Q=nG и аппроксимация
# скаляра n без прямого модульного перебора кандидатов.
#
# Кривая:
# y^2 = x^3 + 7 (mod 499)
#
# Генератор:
# G = (3, 212)
#
# Порядок:
# 73
#
# Особенности:
# 1. Общая выборка для всех моделей.
# 2. Ориентиры выбираются независимо от их скаляров.
# 3. Blocked CV с фиксированными исходными позициями.
# 4. Purge zone.
# 5. Корректный перестановочный контроль.
# 6. Учет выбора лучшей модели в permutation test.
# 7. CSV + ZIP.
# ============================================================

import math
import csv
import zipfile
import numpy as np

from pathlib import Path


# ============================================================
# 1. НАСТРОЙКИ
# ============================================================

P = 499
A = 0
B = 7

G = (3, 212)
ORDER = 73

FOLDS = 6
PURGE_RADIUS = 2

PERMUTATIONS = 99
RANDOM_SEED = 20261102

GAMMAS = [0.3, 1.0, 3.0]
ALPHAS = [0.01, 0.1]

BASE_DIR = Path(__file__).resolve().parent

RESULTS_DIR = BASE_DIR / "experimentC10_results"
RESULTS_DIR.mkdir(exist_ok=True)

ZIP_PATH = BASE_DIR / "experimentC10.zip"


# ============================================================
# 2. АРИФМЕТИКА ЭЛЛИПТИЧЕСКОЙ КРИВОЙ
# ============================================================

def mod_inverse(value, modulus):

    value %= modulus

    if value == 0:
        raise ZeroDivisionError("Обратного элемента не существует")

    return pow(value, -1, modulus)


def point_add(P1, P2):

    if P1 is None:
        return P2

    if P2 is None:
        return P1

    x1, y1 = P1
    x2, y2 = P2

    # Противоположные точки.
    if x1 == x2 and (y1 + y2) % P == 0:
        return None

    # Удвоение.
    if P1 == P2:

        if y1 == 0:
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

    return x3, y3


def scalar_multiply(k, point):

    result = None
    current = point

    while k > 0:

        if k & 1:
            result = point_add(result, current)

        current = point_add(current, current)

        k >>= 1

    return result


def generate_group():

    points = {}

    current = None

    for n in range(ORDER):

        points[n] = current

        current = point_add(current, G)

    if current is not None:
        raise RuntimeError("Ошибка проверки порядка генератора")

    if points[1] != G:
        raise RuntimeError("Ошибка генератора")

    if len(set(points.values())) != ORDER:
        raise RuntimeError("Обнаружены повторяющиеся точки")

    for n in range(1, ORDER):

        x, y = points[n]

        if (y * y - x * x * x - B) % P != 0:
            raise RuntimeError(
                f"Точка {n}G не лежит на кривой"
            )

    return points


# ============================================================
# 3. ГЕОМЕТРИЧЕСКИЕ ФУНКЦИИ
# ============================================================

def signed_modular_difference(a, b):

    """
    Кратчайшее знаковое смещение a-b по модулю P.

    Для P=499 диапазон:
    [-249, 249]
    """

    return ((a - b + P // 2) % P) - P // 2


def circular_distance(a, b):

    difference = abs(a - b)

    return min(difference, ORDER - difference)


def torus_distance(point, anchor):

    x, y = point
    ax, ay = anchor

    dx = signed_modular_difference(x, ax)
    dy = signed_modular_difference(y, ay)

    distance = math.sqrt(dx * dx + dy * dy)

    normalization = math.sqrt(
        2 * (P // 2) ** 2
    )

    return distance / normalization


def torus_vector(point, anchor):

    x, y = point
    ax, ay = anchor

    dx = signed_modular_difference(x, ax)
    dy = signed_modular_difference(y, ay)

    scale = float(P // 2)

    return dx / scale, dy / scale


# ============================================================
# 4. ВЫБОР ОПОРНЫХ ТОЧЕК
# ============================================================

def select_anchors(points):

    """
    Важное условие:

    Ориентиры выбираются исключительно по координатам.

    Скаляр n не используется при сортировке или выборе.

    После выбора мы знаем скаляры ориентиров только для
    технического исключения этих точек из выборки.
    """

    records = []

    for n in range(1, ORDER):

        records.append({
            "n": n,
            "point": points[n]
        })

    # Канонический порядок по координатам.
    records.sort(
        key=lambda item: (
            item["point"][0],
            item["point"][1]
        )
    )

    count = len(records)

    selected_indices = []

    for i in range(8):

        index = round(
            (i + 0.5) * count / 8
        )

        selected_indices.append(index)

    if len(set(selected_indices)) != 8:
        raise RuntimeError("Повторяющиеся ориентиры")

    anchors8 = [
        records[i] for i in selected_indices
    ]

    # Четыре ориентира берутся через один
    # из восьми, сохраняя координатный порядок.
    anchors4 = anchors8[::2]

    return anchors4, anchors8


# ============================================================
# 5. ГЕОМЕТРИЧЕСКИЕ ПРИЗНАКИ
# ============================================================

def build_features(point, anchors):

    distances = []

    vectors = []

    for anchor_record in anchors:

        anchor = anchor_record["point"]

        distances.append(
            torus_distance(point, anchor)
        )

        dx, dy = torus_vector(point, anchor)

        vectors.extend([dx, dy])

    distances = np.asarray(
        distances,
        dtype=float
    )

    vectors = np.asarray(
        vectors,
        dtype=float
    )

    return distances, vectors


def build_relational_features(distances):

    """
    Относительная геометрия системы расстояний.

    Признаки:

    1. Сами расстояния.
    2. Попарные разности расстояний.
    3. Положение каждого расстояния в ранжированном списке.
    """

    result = []

    # Исходные расстояния.
    result.extend(distances.tolist())

    # Попарные разности.
    for i in range(len(distances)):

        for j in range(i + 1, len(distances)):

            result.append(
                distances[i] - distances[j]
            )

    # Ранги расстояний.
    order = np.argsort(
        distances,
        kind="mergesort"
    )

    ranks = np.empty(
        len(distances),
        dtype=float
    )

    ranks[order] = (
        np.arange(len(distances))
        / max(1, len(distances) - 1)
    )

    result.extend(ranks.tolist())

    return np.asarray(result, dtype=float)


# ============================================================
# 6. ПОСТРОЕНИЕ ОБЩЕЙ ВЫБОРКИ
# ============================================================

def build_datasets(points, anchors4, anchors8):

    anchor_scalars = {
        item["n"] for item in anchors8
    }

    sample_scalars = [
        n
        for n in range(1, ORDER)
        if n not in anchor_scalars
    ]

    datasets = {
        "xy_control": [],
        "anchors4_distances": [],
        "anchors4_vectors": [],
        "anchors8_distances": [],
        "anchors8_vectors": [],
        "anchors8_relational": []
    }

    sample_points = []

    for n in sample_scalars:

        point = points[n]

        x, y = point

        sample_points.append(point)

        # Контроль XY.
        datasets["xy_control"].append([
            x / P,
            y / P
        ])

        # Геометрия относительно четырех ориентиров.
        d4, v4 = build_features(
            point,
            anchors4
        )

        datasets["anchors4_distances"].append(
            d4
        )

        datasets["anchors4_vectors"].append(
            v4
        )

        # Геометрия относительно восьми ориентиров.
        d8, v8 = build_features(
            point,
            anchors8
        )

        datasets["anchors8_distances"].append(
            d8
        )

        datasets["anchors8_vectors"].append(
            v8
        )

        datasets["anchors8_relational"].append(
            build_relational_features(d8)
        )

    for name in datasets:

        datasets[name] = np.asarray(
            datasets[name],
            dtype=float
        )

    return (
        datasets,
        sample_scalars,
        sample_points,
        anchor_scalars
    )


# ============================================================
# 7. FIXED BLOCKED CROSS-VALIDATION
# ============================================================

def build_folds(sample_scalars):

    """
    Разбиение строится по ИСХОДНЫМ позициям точек.

    Оно никогда не перестраивается при перестановках
    целевых скаляров.
    """

    sample_scalars = np.asarray(
        sample_scalars,
        dtype=int
    )

    block_size = math.ceil(
        ORDER / FOLDS
    )

    folds = []

    for fold_number in range(FOLDS):

        start = fold_number * block_size

        end = min(
            start + block_size,
            ORDER
        )

        test_indices = np.asarray([
            i
            for i, n in enumerate(sample_scalars)
            if start <= n < end
        ], dtype=int)

        if len(test_indices) == 0:
            continue

        test_positions = sample_scalars[
            test_indices
        ]

        train_indices = []

        for i, n in enumerate(sample_scalars):

            if i in test_indices:
                continue

            nearest_distance = min(
                circular_distance(n, t)
                for t in test_positions
            )

            if nearest_distance > PURGE_RADIUS:

                train_indices.append(i)

        train_indices = np.asarray(
            train_indices,
            dtype=int
        )

        folds.append({
            "fold": fold_number,
            "start": start,
            "end": end,
            "train": train_indices,
            "test": test_indices
        })

    # Проверка полного покрытия.
    assigned = np.zeros(
        len(sample_scalars),
        dtype=bool
    )

    for fold in folds:

        assigned[fold["test"]] = True

    if not np.all(assigned):
        raise RuntimeError(
            "Не все точки получили тестовый прогноз"
        )

    return folds


# ============================================================
# 8. КРУГОВОЕ ПРЕДСТАВЛЕНИЕ СКАЛЯРА
# ============================================================

def encode_labels(labels):

    labels = np.asarray(
        labels,
        dtype=float
    )

    angles = (
        2.0 * math.pi * labels / ORDER
    )

    return np.column_stack([
        np.cos(angles),
        np.sin(angles)
    ])


def decode_labels(vectors):

    angles = np.arctan2(
        vectors[:, 1],
        vectors[:, 0]
    )

    angles = np.mod(
        angles,
        2.0 * math.pi
    )

    labels = np.rint(
        angles * ORDER / (2.0 * math.pi)
    ).astype(int)

    return labels % ORDER


# ============================================================
# 9. RBF KERNEL
# ============================================================

def rbf_kernel(X1, X2, gamma):

    norms1 = np.sum(
        X1 * X1,
        axis=1
    ).reshape(-1, 1)

    norms2 = np.sum(
        X2 * X2,
        axis=1
    ).reshape(1, -1)

    distances = (
        norms1 + norms2 - 2.0 * X1 @ X2.T
    )

    distances = np.maximum(
        distances,
        0.0
    )

    return np.exp(
        -gamma * distances
    )


# ============================================================
# 10. ПРЕДВАРИТЕЛЬНЫЙ РАСЧЕТ ЯДЕР
# ============================================================

def prepare_kernel_cache(datasets, folds):

    cache = {}

    for model_name, X in datasets.items():

        for gamma in GAMMAS:

            prepared_folds = []

            for fold in folds:

                train_idx = fold["train"]
                test_idx = fold["test"]

                X_train = X[train_idx]
                X_test = X[test_idx]

                # Стандартизация только по train.
                mean = X_train.mean(axis=0)

                std = X_train.std(axis=0)

                keep = std > 1e-12

                if not np.any(keep):
                    raise RuntimeError(
                        f"Все признаки постоянны: {model_name}"
                    )

                X_train = (
                    X_train[:, keep] - mean[keep]
                ) / std[keep]

                X_test = (
                    X_test[:, keep] - mean[keep]
                ) / std[keep]

                K_train = rbf_kernel(
                    X_train,
                    X_train,
                    gamma
                )

                K_test = rbf_kernel(
                    X_test,
                    X_train,
                    gamma
                )

                prepared_folds.append({
                    "train": train_idx,
                    "test": test_idx,
                    "K_train": K_train,
                    "K_test": K_test
                })

            cache[(model_name, gamma)] = (
                prepared_folds
            )

    return cache


# ============================================================
# 11. ОЦЕНКА ОДНОЙ КОНФИГУРАЦИИ
# ============================================================

def evaluate_configuration(
    prepared_folds,
    target_labels,
    alpha
):

    target_labels = np.asarray(
        target_labels,
        dtype=int
    )

    encoded = encode_labels(
        target_labels
    )

    predictions = np.full(
        len(target_labels),
        -1,
        dtype=int
    )

    identity = None

    for fold in prepared_folds:

        train_idx = fold["train"]
        test_idx = fold["test"]

        K_train = fold["K_train"]
        K_test = fold["K_test"]

        Y_train = encoded[train_idx]

        regularized = (
            K_train
            + alpha * np.eye(len(train_idx))
        )

        coefficients = np.linalg.solve(
            regularized,
            Y_train
        )

        predicted_vectors = (
            K_test @ coefficients
        )

        predicted_labels = decode_labels(
            predicted_vectors
        )

        predictions[test_idx] = predicted_labels

    if np.any(predictions < 0):
        raise RuntimeError(
            "Некоторые прогнозы отсутствуют"
        )

    differences = np.abs(
        predictions - target_labels
    )

    circular_errors = np.minimum(
        differences,
        ORDER - differences
    )

    circular_mae = float(
        np.mean(circular_errors)
    )

    accuracy = float(
        np.mean(predictions == target_labels)
    )

    return {
        "accuracy": accuracy,
        "mae": circular_mae,
        "predictions": predictions,
        "errors": circular_errors
    }


# ============================================================
# 12. ПОЛНЫЙ НАБОР КОНФИГУРАЦИЙ
# ============================================================

def evaluate_all(
    cache,
    target_labels,
    keep_predictions=False
):

    results = []

    for model_name in [
        "xy_control",
        "anchors4_distances",
        "anchors4_vectors",
        "anchors8_distances",
        "anchors8_vectors",
        "anchors8_relational"
    ]:

        for gamma in GAMMAS:

            prepared = cache[
                (model_name, gamma)
            ]

            for alpha in ALPHAS:

                result = evaluate_configuration(
                    prepared,
                    target_labels,
                    alpha
                )

                row = {
                    "model": model_name,
                    "gamma": gamma,
                    "alpha": alpha,
                    "accuracy": result["accuracy"],
                    "mae": result["mae"]
                }

                if keep_predictions:

                    row["predictions"] = (
                        result["predictions"]
                    )

                    row["errors"] = (
                        result["errors"]
                    )

                results.append(row)

    return results


# ============================================================
# 13. СОХРАНЕНИЕ CSV
# ============================================================

def save_csv(path, rows, fieldnames):

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
# 14. ОСНОВНОЙ ЭКСПЕРИМЕНТ
# ============================================================

def main():

    print("=" * 78)

    print("ЗАПУСК ЭКСПЕРИМЕНТА C10")

    print("Исправленная геометрическая триангуляция")

    print("=" * 78)

    print()

    # --------------------------------------------------------
    # Построение группы.
    # --------------------------------------------------------

    print("[1/8] Построение эллиптической группы...")

    points = generate_group()

    print(
        f"Кривая: y^2 = x^3 + 7 (mod {P})"
    )

    print(f"Генератор: {G}")

    print(f"Порядок: {ORDER}")

    print(
        f"Проверено точек: {ORDER}"
    )

    # --------------------------------------------------------
    # Ориентиры.
    # --------------------------------------------------------

    print()
    print("[2/8] Выбор геометрических ориентиров...")

    anchors4, anchors8 = select_anchors(
        points
    )

    print()
    print("ВОСЕМЬ ОРИЕНТИРОВ:")

    anchor_rows = []

    for i, item in enumerate(anchors8):

        n = item["n"]
        x, y = item["point"]

        print(
            f"A{i + 1}: P={item['point']}, "
            f"контрольный скаляр={n}"
        )

        anchor_rows.append({
            "anchor": i + 1,
            "scalar_for_audit": n,
            "x": x,
            "y": y
        })

    print()
    print("ЧЕТЫРЕ ОРИЕНТИРА:")

    for i, item in enumerate(anchors4):

        print(
            f"A{i + 1}: {item['point']}"
        )

    # --------------------------------------------------------
    # Выборка.
    # --------------------------------------------------------

    print()
    print("[3/8] Построение общей выборки...")

    (
        datasets,
        sample_scalars,
        sample_points,
        anchor_scalars
    ) = build_datasets(
        points,
        anchors4,
        anchors8
    )

    print(
        f"Ориентиры исключены: {len(anchor_scalars)}"
    )

    print(
        f"Образцов: {len(sample_scalars)}"
    )

    print()

    for name, X in datasets.items():

        print(
            f"{name:28s} "
            f"признаков={X.shape[1]:3d} "
            f"образцов={X.shape[0]:3d}"
        )

    # --------------------------------------------------------
    # Разбиение.
    # --------------------------------------------------------

    print()
    print("[4/8] Построение фиксированных CV-блоков...")

    folds = build_folds(
        sample_scalars
    )

    fold_rows = []

    for fold in folds:

        print(
            f"Fold {fold['fold'] + 1}: "
            f"позиции [{fold['start']}, {fold['end']}), "
            f"train={len(fold['train'])}, "
            f"test={len(fold['test'])}"
        )

        for index in fold["test"]:

            fold_rows.append({
                "sample_scalar_position": sample_scalars[index],
                "fold": fold["fold"] + 1
            })

    # --------------------------------------------------------
    # Предварительные ядра.
    # --------------------------------------------------------

    print()
    print("[5/8] Подготовка RBF-ядер...")

    cache = prepare_kernel_cache(
        datasets,
        folds
    )

    print("Ядра подготовлены.")

    # --------------------------------------------------------
    # Наблюдаемые результаты.
    # --------------------------------------------------------

    print()
    print("[6/8] BLOCKED CROSS-VALIDATION")

    y_true = np.asarray(
        sample_scalars,
        dtype=int
    )

    observed = evaluate_all(
        cache,
        y_true,
        keep_predictions=True
    )

    observed.sort(
        key=lambda item: (
            item["mae"],
            -item["accuracy"]
        )
    )

    print()

    print(
        f"{'Модель':28s} "
        f"{'Gamma':>7s} "
        f"{'Alpha':>7s} "
        f"{'Accuracy':>10s} "
        f"{'Circular MAE':>14s}"
    )

    print("-" * 78)

    for row in observed:

        print(
            f"{row['model']:28s} "
            f"{row['gamma']:7.3f} "
            f"{row['alpha']:7.3f} "
            f"{row['accuracy']:10.4f} "
            f"{row['mae']:14.4f}"
        )

    best_mae = observed[0]

    best_accuracy = max(
        observed,
        key=lambda item: item["accuracy"]
    )

    observed_min_mae = best_mae["mae"]

    observed_max_accuracy = (
        best_accuracy["accuracy"]
    )

    print()
    print("ЛУЧШАЯ КОНФИГУРАЦИЯ ПО MAE:")

    print(
        f"Модель: {best_mae['model']}"
    )

    print(
        f"Gamma: {best_mae['gamma']}"
    )

    print(
        f"Alpha: {best_mae['alpha']}"
    )

    print(
        f"Circular MAE: {best_mae['mae']:.4f}"
    )

    print(
        f"Accuracy: {best_mae['accuracy']:.4f}"
    )

    print()
    print("ЛУЧШАЯ ACCURACY СРЕДИ ВСЕХ КОНФИГУРАЦИЙ:")

    print(
        f"Модель: {best_accuracy['model']}"
    )

    print(
        f"Accuracy: {best_accuracy['accuracy']:.4f}"
    )

    # --------------------------------------------------------
    # Перестановочный контроль.
    # --------------------------------------------------------

    print()
    print("[7/8] ПЕРЕСТАНОВОЧНЫЙ КОНТРОЛЬ")

    print(
        f"Количество перестановок: {PERMUTATIONS}"
    )

    print(
        "Исходные CV-блоки остаются неизменными."
    )

    print(
        "Переставляются только целевые скаляры."
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    permutation_min_mae = []

    permutation_max_accuracy = []

    for iteration in range(PERMUTATIONS):

        shuffled_labels = rng.permutation(
            y_true
        )

        permutation_results = evaluate_all(
            cache,
            shuffled_labels,
            keep_predictions=False
        )

        min_mae = min(
            row["mae"]
            for row in permutation_results
        )

        max_accuracy = max(
            row["accuracy"]
            for row in permutation_results
        )

        permutation_min_mae.append(
            min_mae
        )

        permutation_max_accuracy.append(
            max_accuracy
        )

        if (
            (iteration + 1) % 10 == 0
            or iteration + 1 == PERMUTATIONS
        ):

            print(
                f"Перестановка "
                f"{iteration + 1}/{PERMUTATIONS}"
            )

    permutation_min_mae = np.asarray(
        permutation_min_mae
    )

    permutation_max_accuracy = np.asarray(
        permutation_max_accuracy
    )

    # --------------------------------------------------------
    # Статистические оценки.
    # --------------------------------------------------------

    p_mae = (
        1 + np.sum(
            permutation_min_mae <= observed_min_mae
        )
    ) / (PERMUTATIONS + 1)

    p_accuracy = (
        1 + np.sum(
            permutation_max_accuracy >= observed_max_accuracy
        )
    ) / (PERMUTATIONS + 1)

    random_mean_mae = float(
        np.mean(permutation_min_mae)
    )

    random_min_mae = float(
        np.min(permutation_min_mae)
    )

    random_max_accuracy = float(
        np.mean(permutation_max_accuracy)
    )

    # --------------------------------------------------------
    # Вывод.
    # --------------------------------------------------------

    print()
    print("=" * 78)

    print("СТАТИСТИЧЕСКИЕ РЕЗУЛЬТАТЫ C10")

    print("=" * 78)

    print(
        f"Наблюдаемая лучшая MAE: "
        f"{observed_min_mae:.4f}"
    )

    print(
        f"Средняя случайная лучшая MAE: "
        f"{random_mean_mae:.4f}"
    )

    print(
        f"Минимальная случайная лучшая MAE: "
        f"{random_min_mae:.4f}"
    )

    print(
        f"Permutation p-value MAE: "
        f"{p_mae:.4f}"
    )

    print()

    print(
        f"Наблюдаемая максимальная Accuracy: "
        f"{observed_max_accuracy:.4f}"
    )

    print(
        f"Средняя случайная максимальная Accuracy: "
        f"{random_max_accuracy:.4f}"
    )

    print(
        f"Permutation p-value Accuracy: "
        f"{p_accuracy:.4f}"
    )

    print()

    print(
        "Случайная точность при равномерном угадывании "
        f"среди 64 допустимых меток: "
        f"{1 / len(y_true):.4%}"
    )

    # --------------------------------------------------------
    # CSV: сводная таблица.
    # --------------------------------------------------------

    summary_rows = []

    for row in observed:

        summary_rows.append({
            "model": row["model"],
            "gamma": row["gamma"],
            "alpha": row["alpha"],
            "accuracy": row["accuracy"],
            "circular_mae": row["mae"]
        })

    save_csv(
        RESULTS_DIR / "C10_model_summary.csv",
        summary_rows,
        [
            "model",
            "gamma",
            "alpha",
            "accuracy",
            "circular_mae"
        ]
    )

    # --------------------------------------------------------
    # CSV: предсказания лучшей модели.
    # --------------------------------------------------------

    prediction_rows = []

    for i, n in enumerate(sample_scalars):

        x, y = sample_points[i]

        prediction = int(
            best_mae["predictions"][i]
        )

        error = int(
            best_mae["errors"][i]
        )

        prediction_rows.append({
            "true_scalar": n,
            "predicted_scalar": prediction,
            "circular_error": error,
            "x": x,
            "y": y,
            "model": best_mae["model"],
            "gamma": best_mae["gamma"],
            "alpha": best_mae["alpha"]
        })

    save_csv(
        RESULTS_DIR / "C10_predictions.csv",
        prediction_rows,
        [
            "true_scalar",
            "predicted_scalar",
            "circular_error",
            "x",
            "y",
            "model",
            "gamma",
            "alpha"
        ]
    )

    # --------------------------------------------------------
    # CSV: ориентиры.
    # --------------------------------------------------------

    save_csv(
        RESULTS_DIR / "C10_anchors.csv",
        anchor_rows,
        [
            "anchor",
            "scalar_for_audit",
            "x",
            "y"
        ]
    )

    # --------------------------------------------------------
    # CSV: CV-блоки.
    # --------------------------------------------------------

    save_csv(
        RESULTS_DIR / "C10_folds.csv",
        fold_rows,
        [
            "sample_scalar_position",
            "fold"
        ]
    )

    # --------------------------------------------------------
    # CSV: перестановочный контроль.
    # --------------------------------------------------------

    permutation_rows = []

    for i in range(PERMUTATIONS):

        permutation_rows.append({
            "permutation": i + 1,
            "minimum_circular_mae": float(
                permutation_min_mae[i]
            ),
            "maximum_accuracy": float(
                permutation_max_accuracy[i]
            )
        })

    save_csv(
        RESULTS_DIR / "C10_permutations.csv",
        permutation_rows,
        [
            "permutation",
            "minimum_circular_mae",
            "maximum_accuracy"
        ]
    )

    # --------------------------------------------------------
    # Текстовый отчет.
    # --------------------------------------------------------

    report = []

    report.append("EXPERIMENT C10")
    report.append("Corrected geometric triangulation")
    report.append("")
    report.append(f"p = {P}")
    report.append(f"curve: y^2 = x^3 + {B}")
    report.append(f"G = {G}")
    report.append(f"order = {ORDER}")
    report.append("")
    report.append(f"sample_count = {len(y_true)}")
    report.append(f"anchor_count = {len(anchor_scalars)}")
    report.append(f"folds = {FOLDS}")
    report.append(f"purge_radius = {PURGE_RADIUS}")
    report.append(f"permutations = {PERMUTATIONS}")
    report.append("")
    report.append("BEST MAE CONFIGURATION")
    report.append(f"model = {best_mae['model']}")
    report.append(f"gamma = {best_mae['gamma']}")
    report.append(f"alpha = {best_mae['alpha']}")
    report.append(f"accuracy = {best_mae['accuracy']}")
    report.append(f"mae = {best_mae['mae']}")
    report.append("")
    report.append("STATISTICS")
    report.append(f"random_mean_mae = {random_mean_mae}")
    report.append(f"random_min_mae = {random_min_mae}")
    report.append(f"p_mae = {p_mae}")
    report.append(
        f"observed_max_accuracy = {observed_max_accuracy}"
    )
    report.append(
        f"random_mean_max_accuracy = {random_max_accuracy}"
    )
    report.append(f"p_accuracy = {p_accuracy}")
    report.append("")
    report.append("INTERPRETATION")
    report.append(
        "A result with p < 0.05 is evidence of a "
        "statistical association under this permutation null."
    )
    report.append(
        "It is not by itself evidence of a general "
        "ECDLP solution."
    )
    report.append(
        "The model uses supervised scalar labels "
        "from the remaining sample points."
    )

    with open(
        RESULTS_DIR / "C10_report.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "\n".join(report)
        )

    # --------------------------------------------------------
    # ZIP.
    # --------------------------------------------------------

    print()
    print("[8/8] Создание ZIP-архива...")

    with zipfile.ZipFile(
        ZIP_PATH,
        "w",
        compression=zipfile.ZIP_DEFLATED
    ) as archive:

        for file_path in sorted(
            RESULTS_DIR.iterdir()
        ):

            if file_path.is_file():

                archive.write(
                    file_path,
                    arcname=file_path.name
                )

    # --------------------------------------------------------
    # Завершение.
    # --------------------------------------------------------

    print()
    print("=" * 78)

    print("C10 ЗАВЕРШЁН")

    print("=" * 78)

    print()

    print("РЕЗУЛЬТАТЫ:")

    print(RESULTS_DIR)

    print()

    print("ZIP-АРХИВ:")

    print(ZIP_PATH)

    print()

    if p_mae < 0.05:

        print(
            "ВНИМАНИЕ: обнаружено статистически "
            "значимое улучшение Circular MAE."
        )

    else:

        print(
            "Статистически значимого улучшения "
            "Circular MAE не обнаружено."
        )

    if p_accuracy < 0.05:

        print(
            "Обнаружена статистически значимая "
            "точная предсказательная способность."
        )

    else:

        print(
            "Статистически значимое улучшение "
            "точной Accuracy не обнаружено."
        )

    print()
    print("Ожидаем анализа результатов C10.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
