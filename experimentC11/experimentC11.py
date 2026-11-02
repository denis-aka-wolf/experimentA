
# ============================================================
# EXPERIMENT C11
#
# МУЛЬТИПЛИКАТИВНАЯ ГЕОМЕТРИЧЕСКАЯ ОРБИТА
#
# Исследовательская цель:
# Найти геометрическое представление точки Q=nG,
# которое позволяет аппроксимировать скаляр n
# без прямого перебора возможных скаляров.
#
# Новая гипотеза:
# Геометрия точек [2]Q, [3]Q, ..., [13]Q
# может содержать признаки, полезные для аппроксимации n.
#
# Кривая:
# y^2 = x^3 + 7 (mod 499)
#
# G = (3, 212)
# ORDER = 73
#
# Все модели используют одинаковую выборку.
# CV-блоки фиксированы при перестановках.
#
# Результаты:
# CSV + TXT + ZIP
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

MULTIPLIERS = list(range(2, 14))

FOLDS = 6
PURGE_RADIUS = 2

PERMUTATIONS = 99

RANDOM_SEED = 20261103

GAMMAS = [0.3, 1.0, 3.0]
ALPHAS = [0.01, 0.1]

BASE_DIR = Path(__file__).resolve().parent

RESULTS_DIR = BASE_DIR / "experimentC11_results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ZIP_PATH = BASE_DIR / "experimentC11.zip"


# ============================================================
# 2. АРИФМЕТИКА ЭЛЛИПТИЧЕСКОЙ КРИВОЙ
# ============================================================

def mod_inverse(value, modulus):

    value %= modulus

    if value == 0:
        raise ZeroDivisionError(
            "Обратного элемента не существует"
        )

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

    # Удвоение точки.
    if P1 == P2:

        if y1 == 0:
            return None

        numerator = (
            3 * x1 * x1 + A
        ) % P

        denominator = (
            2 * y1
        ) % P

    else:

        numerator = (
            y2 - y1
        ) % P

        denominator = (
            x2 - x1
        ) % P

    slope = (
        numerator
        * mod_inverse(denominator, P)
    ) % P

    x3 = (
        slope * slope - x1 - x2
    ) % P

    y3 = (
        slope * (x1 - x3) - y1
    ) % P

    return x3, y3


def scalar_multiply(k, point):

    """
    Умножение точки на известный малый множитель.

    В C11 k принимает значения 2..13.

    Здесь нет перебора неизвестного скаляра n.
    """

    result = None
    current = point

    while k > 0:

        if k & 1:

            result = point_add(
                result,
                current
            )

        current = point_add(
            current,
            current
        )

        k >>= 1

    return result


def generate_group():

    points = {}

    current = None

    for n in range(ORDER):

        points[n] = current

        current = point_add(
            current,
            G
        )

    if current is not None:

        raise RuntimeError(
            "Порядок генератора не совпадает с ORDER"
        )

    if points[1] != G:

        raise RuntimeError(
            "Ошибка проверки генератора"
        )

    if len(set(points.values())) != ORDER:

        raise RuntimeError(
            "Обнаружены повторяющиеся точки"
        )

    # Проверка принадлежности кривой.
    for n in range(1, ORDER):

        x, y = points[n]

        if (
            y * y - x * x * x - B
        ) % P != 0:

            raise RuntimeError(
                f"Точка {n}G не лежит на кривой"
            )

    return points


# ============================================================
# 3. ПРОВЕРКА МУЛЬТИПЛИКАТИВНОЙ ОРБИТЫ
# ============================================================

def build_multiplication_orbit(Q):

    """
    Строим:

    Q, 2Q, 3Q, ..., 13Q.

    Каждый элемент вычисляется непосредственно
    из предыдущих операций группового умножения.

    Скаляр n неизвестен и не используется.
    """

    orbit = {
        1: Q
    }

    for j in MULTIPLIERS:

        orbit[j] = scalar_multiply(
            j,
            Q
        )

    return orbit


# ============================================================
# 4. МОДУЛЬНАЯ ГЕОМЕТРИЯ
# ============================================================

def signed_difference(a, b):

    return (
        (a - b + P // 2) % P
    ) - P // 2


def normalized_coordinates(point):

    if point is None:

        raise RuntimeError(
            "Точка на бесконечности не ожидается"
        )

    x, y = point

    return [
        x / P,
        y / P
    ]


def torus_distance(P1, P2):

    x1, y1 = P1
    x2, y2 = P2

    dx = signed_difference(
        x1,
        x2
    )

    dy = signed_difference(
        y1,
        y2
    )

    distance = math.sqrt(
        dx * dx + dy * dy
    )

    normalization = math.sqrt(
        2 * (P // 2) ** 2
    )

    return distance / normalization


# ============================================================
# 5. НАКЛОНЫ ХОРД
# ============================================================

def chord_slope(P1, P2):

    """
    Наклон прямой через две точки кривой.

    Для P1 == P2 используется касательная.
    """

    if P1 is None or P2 is None:

        raise RuntimeError(
            "Неожиданная точка на бесконечности"
        )

    x1, y1 = P1
    x2, y2 = P2

    if P1 == P2:

        if y1 == 0:

            return 0.0

        numerator = (
            3 * x1 * x1 + A
        ) % P

        denominator = (
            2 * y1
        ) % P

    else:

        if x1 == x2:

            # Вертикальная прямая.
            return 0.0

        numerator = (
            y2 - y1
        ) % P

        denominator = (
            x2 - x1
        ) % P

    slope = (
        numerator
        * mod_inverse(denominator, P)
    ) % P

    return slope / P


# ============================================================
# 6. ПОСТРОЕНИЕ ПРИЗНАКОВ
# ============================================================

def build_features(Q):

    orbit = build_multiplication_orbit(Q)

    features = {}

    # --------------------------------------------------------
    # Контрольная модель.
    # --------------------------------------------------------

    features["xy_control"] = (
        normalized_coordinates(Q)
    )

    # --------------------------------------------------------
    # Только координаты x точек 2Q..13Q.
    # --------------------------------------------------------

    x_features = []

    for j in MULTIPLIERS:

        point = orbit[j]

        x, y = point

        x_features.append(
            x / P
        )

    features["mul_x_2_13"] = x_features

    # --------------------------------------------------------
    # Координаты 2Q, 3Q, 5Q, 7Q.
    # --------------------------------------------------------

    short_features = []

    for j in [2, 3, 5, 7]:

        short_features.extend(
            normalized_coordinates(
                orbit[j]
            )
        )

    features["mul_xy_2_5"] = short_features

    # --------------------------------------------------------
    # Координаты всех точек 2Q..13Q.
    # --------------------------------------------------------

    xy_features = []

    for j in MULTIPLIERS:

        xy_features.extend(
            normalized_coordinates(
                orbit[j]
            )
        )

    features["mul_xy_2_13"] = xy_features

    # --------------------------------------------------------
    # Наклоны прямых Q -> jQ.
    # --------------------------------------------------------

    slope_features = []

    for j in MULTIPLIERS:

        slope_features.append(
            chord_slope(
                Q,
                orbit[j]
            )
        )

    features["mul_slope_2_13"] = slope_features

    # --------------------------------------------------------
    # Полная алгебраическая сигнатура.
    # --------------------------------------------------------

    full_features = (
        xy_features
        + slope_features
    )

    features["mul_full_2_13"] = full_features

    # --------------------------------------------------------
    # Попарные геометрические расстояния.
    #
    # Используются точки:
    # Q, 2Q, ..., 13Q.
    #
    # Всего 13 точек.
    # Число пар = 13*12/2 = 78.
    # --------------------------------------------------------

    orbit_points = [
        orbit[j]
        for j in range(1, 14)
    ]

    distance_features = []

    for i in range(len(orbit_points)):

        for j in range(i + 1, len(orbit_points)):

            distance_features.append(
                torus_distance(
                    orbit_points[i],
                    orbit_points[j]
                )
            )

    features["mul_pairdist_2_13"] = (
        distance_features
    )

    return features


# ============================================================
# 7. ПОСТРОЕНИЕ ОБЩЕЙ ВЫБОРКИ
# ============================================================

def build_datasets(points):

    dataset_names = [
        "xy_control",
        "mul_x_2_13",
        "mul_xy_2_5",
        "mul_xy_2_13",
        "mul_slope_2_13",
        "mul_full_2_13",
        "mul_pairdist_2_13"
    ]

    datasets = {
        name: []
        for name in dataset_names
    }

    sample_scalars = []
    sample_points = []

    for n in range(1, ORDER):

        Q = points[n]

        features = build_features(Q)

        sample_scalars.append(n)

        sample_points.append(Q)

        for name in dataset_names:

            datasets[name].append(
                features[name]
            )

    for name in dataset_names:

        datasets[name] = np.asarray(
            datasets[name],
            dtype=float
        )

        if not np.all(
            np.isfinite(datasets[name])
        ):

            raise RuntimeError(
                f"Некорректные признаки: {name}"
            )

    return (
        datasets,
        sample_scalars,
        sample_points
    )


# ============================================================
# 8. ФИКСИРОВАННЫЕ BLOCKED FOLDS
# ============================================================

def circular_distance(a, b):

    difference = abs(a - b)

    return min(
        difference,
        ORDER - difference
    )


def build_folds(sample_scalars):

    """
    CV-блоки зависят только от исходной позиции n.

    При перестановочном контроле они не меняются.
    """

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

        test_positions = [
            sample_scalars[i]
            for i in test_indices
        ]

        train_indices = []

        for i, n in enumerate(sample_scalars):

            if i in test_indices:
                continue

            nearest = min(
                circular_distance(n, t)
                for t in test_positions
            )

            if nearest > PURGE_RADIUS:

                train_indices.append(i)

        folds.append({
            "fold": fold_number,
            "start": start,
            "end": end,
            "train": np.asarray(
                train_indices,
                dtype=int
            ),
            "test": test_indices
        })

    assigned = np.zeros(
        len(sample_scalars),
        dtype=bool
    )

    for fold in folds:

        assigned[fold["test"]] = True

    if not np.all(assigned):

        raise RuntimeError(
            "Не все образцы попали в тестовые блоки"
        )

    return folds


# ============================================================
# 9. КРУГОВОЕ КОДИРОВАНИЕ ЦЕЛЕВОГО СКАЛЯРА
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
# 10. RBF KERNEL
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
        norms1
        + norms2
        - 2.0 * X1 @ X2.T
    )

    distances = np.maximum(
        distances,
        0.0
    )

    return np.exp(
        -gamma * distances
    )


# ============================================================
# 11. ПОДГОТОВКА ЯДЕР
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

                # Статистики вычисляются только на train.
                mean = X_train.mean(axis=0)

                std = X_train.std(axis=0)

                keep = std > 1e-12

                if not np.any(keep):

                    raise RuntimeError(
                        f"Нет изменяющихся признаков: {model_name}"
                    )

                X_train = (
                    X_train[:, keep]
                    - mean[keep]
                ) / std[keep]

                X_test = (
                    X_test[:, keep]
                    - mean[keep]
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
# 12. ОЦЕНКА ОДНОЙ КОНФИГУРАЦИИ
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

        predictions[test_idx] = (
            predicted_labels
        )

    if np.any(predictions < 0):

        raise RuntimeError(
            "Не все образцы получили прогноз"
        )

    differences = np.abs(
        predictions - target_labels
    )

    errors = np.minimum(
        differences,
        ORDER - differences
    )

    mae = float(
        np.mean(errors)
    )

    accuracy = float(
        np.mean(
            predictions == target_labels
        )
    )

    return {
        "accuracy": accuracy,
        "mae": mae,
        "predictions": predictions,
        "errors": errors
    }


# ============================================================
# 13. ВСЕ МОДЕЛИ И ПАРАМЕТРЫ
# ============================================================

def evaluate_all(
    cache,
    target_labels,
    keep_predictions=False
):

    results = []

    model_names = [
        "xy_control",
        "mul_x_2_13",
        "mul_xy_2_5",
        "mul_xy_2_13",
        "mul_slope_2_13",
        "mul_full_2_13",
        "mul_pairdist_2_13"
    ]

    for model_name in model_names:

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
# 14. CSV
# ============================================================

def save_csv(path, rows, fields):

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        writer.writeheader()

        writer.writerows(rows)


# ============================================================
# 15. ГЛАВНАЯ ФУНКЦИЯ
# ============================================================

def main():

    print("=" * 78)

    print("ЗАПУСК ЭКСПЕРИМЕНТА C11")

    print("Мультипликативная геометрическая орбита")

    print("=" * 78)

    # --------------------------------------------------------
    # Построение группы.
    # --------------------------------------------------------

    print()
    print("[1/8] Построение группы...")

    points = generate_group()

    print(
        f"Кривая: y^2 = x^3 + 7 (mod {P})"
    )

    print(f"Генератор: {G}")

    print(f"Порядок: {ORDER}")

    print(
        f"Проверено точек: {len(points)}"
    )

    # --------------------------------------------------------
    # Построение признаков.
    # --------------------------------------------------------

    print()
    print("[2/8] Построение мультипликативных орбит...")

    (
        datasets,
        sample_scalars,
        sample_points
    ) = build_datasets(points)

    print(
        f"Общая выборка: {len(sample_scalars)}"
    )

    print()

    for name, X in datasets.items():

        print(
            f"{name:25s} "
            f"признаков={X.shape[1]:3d} "
            f"образцов={X.shape[0]:3d}"
        )

    # --------------------------------------------------------
    # CV.
    # --------------------------------------------------------

    print()
    print("[3/8] Построение фиксированных CV-блоков...")

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
                "sample_position": sample_scalars[index],
                "fold": fold["fold"] + 1
            })

    # --------------------------------------------------------
    # Ядра.
    # --------------------------------------------------------

    print()
    print("[4/8] Подготовка RBF-ядер...")

    cache = prepare_kernel_cache(
        datasets,
        folds
    )

    print("Ядра подготовлены.")

    # --------------------------------------------------------
    # Наблюдаемые результаты.
    # --------------------------------------------------------

    print()
    print("[5/8] BLOCKED CROSS-VALIDATION")

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
        key=lambda row: (
            row["mae"],
            -row["accuracy"]
        )
    )

    print()

    print(
        f"{'Модель':25s} "
        f"{'Gamma':>7s} "
        f"{'Alpha':>7s} "
        f"{'Accuracy':>10s} "
        f"{'Circular MAE':>14s}"
    )

    print("-" * 78)

    for row in observed:

        print(
            f"{row['model']:25s} "
            f"{row['gamma']:7.3f} "
            f"{row['alpha']:7.3f} "
            f"{row['accuracy']:10.4f} "
            f"{row['mae']:14.4f}"
        )

    best_mae = observed[0]

    best_accuracy = max(
        observed,
        key=lambda row: row["accuracy"]
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

    # --------------------------------------------------------
    # Перестановочный контроль.
    # --------------------------------------------------------

    print()
    print("[6/8] ПЕРЕСТАНОВОЧНЫЙ КОНТРОЛЬ")

    print(
        f"Количество перестановок: {PERMUTATIONS}"
    )

    print(
        "Все модели используют одну и ту же "
        "переставленную последовательность меток."
    )

    print(
        "CV-блоки и purge-зоны остаются неизменными."
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
    # Статистика.
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

    random_mean_accuracy = float(
        np.mean(permutation_max_accuracy)
    )

    # --------------------------------------------------------
    # Вывод статистики.
    # --------------------------------------------------------

    print()
    print("=" * 78)

    print("СТАТИСТИЧЕСКИЕ РЕЗУЛЬТАТЫ C11")

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
        f"{random_mean_accuracy:.4f}"
    )

    print(
        f"Permutation p-value Accuracy: "
        f"{p_accuracy:.4f}"
    )

    print()

    print(
        f"Простой случайный ориентир Accuracy: "
        f"{1 / ORDER:.4%}"
    )

    # --------------------------------------------------------
    # CSV результатов моделей.
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
        RESULTS_DIR / "C11_model_summary.csv",
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
    # CSV предсказаний.
    # --------------------------------------------------------

    prediction_rows = []

    for i, n in enumerate(sample_scalars):

        x, y = sample_points[i]

        prediction_rows.append({
            "true_scalar": n,
            "predicted_scalar": int(
                best_mae["predictions"][i]
            ),
            "circular_error": int(
                best_mae["errors"][i]
            ),
            "x": x,
            "y": y,
            "model": best_mae["model"],
            "gamma": best_mae["gamma"],
            "alpha": best_mae["alpha"]
        })

    save_csv(
        RESULTS_DIR / "C11_predictions.csv",
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
    # CSV CV-блоков.
    # --------------------------------------------------------

    save_csv(
        RESULTS_DIR / "C11_folds.csv",
        fold_rows,
        [
            "sample_position",
            "fold"
        ]
    )

    # --------------------------------------------------------
    # CSV перестановок.
    # --------------------------------------------------------

    permutation_rows = []

    for i in range(PERMUTATIONS):

        permutation_rows.append({
            "permutation": i + 1,
            "minimum_mae": float(
                permutation_min_mae[i]
            ),
            "maximum_accuracy": float(
                permutation_max_accuracy[i]
            )
        })

    save_csv(
        RESULTS_DIR / "C11_permutations.csv",
        permutation_rows,
        [
            "permutation",
            "minimum_mae",
            "maximum_accuracy"
        ]
    )

    # --------------------------------------------------------
    # Текстовый отчет.
    # --------------------------------------------------------

    report = [
        "EXPERIMENT C11",
        "Multiplicative geometric orbit",
        "",
        f"P = {P}",
        f"G = {G}",
        f"ORDER = {ORDER}",
        f"MULTIPLIERS = {MULTIPLIERS}",
        "",
        f"SAMPLES = {len(y_true)}",
        f"FOLDS = {FOLDS}",
        f"PURGE_RADIUS = {PURGE_RADIUS}",
        f"PERMUTATIONS = {PERMUTATIONS}",
        "",
        "BEST MAE MODEL",
        f"model = {best_mae['model']}",
        f"gamma = {best_mae['gamma']}",
        f"alpha = {best_mae['alpha']}",
        f"accuracy = {best_mae['accuracy']}",
        f"mae = {best_mae['mae']}",
        "",
        "STATISTICS",
        f"random_mean_mae = {random_mean_mae}",
        f"random_min_mae = {random_min_mae}",
        f"p_mae = {p_mae}",
        f"observed_max_accuracy = {observed_max_accuracy}",
        f"random_mean_accuracy = {random_mean_accuracy}",
        f"p_accuracy = {p_accuracy}",
        "",
        "INTERPRETATION",
        "Features are computed from Q and fixed small multipliers.",
        "No candidate scalar enumeration is performed.",
        "All models use the same samples and fixed folds.",
        "Permutation changes targets but not fold positions.",
        "A significant result would indicate an association",
        "under the tested permutation null, not a general ECDLP solution."
    ]

    with open(
        RESULTS_DIR / "C11_report.txt",
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
    print("[7/8] Создание ZIP-архива...")

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
    # Итог.
    # --------------------------------------------------------

    print()
    print("[8/8] ЗАВЕРШЕНИЕ")

    print("=" * 78)

    print("C11 ЗАВЕРШЁН")

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
            "Обнаружено статистически значимое "
            "улучшение Circular MAE."
        )

    else:

        print(
            "Значимого улучшения Circular MAE "
            "не обнаружено."
        )

    if p_accuracy < 0.05:

        print(
            "Обнаружено статистически значимое "
            "улучшение точной Accuracy."
        )

    else:

        print(
            "Значимого улучшения точной Accuracy "
            "не обнаружено."
        )

    print()
    print("Ожидаем анализа результатов C11.")


# ============================================================
# ЗАПУСК
# ============================================================

if __name__ == "__main__":

    main()
