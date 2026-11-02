
import os
import csv
import math
import zipfile
import numpy as np

from pathlib import Path
from datetime import datetime


# ============================================================
# C7. ГЕОМЕТРИЧЕСКОЕ ПРЕДСТАВЛЕНИЕ ТРАЕКТОРИИ
#
# Цель:
# Проверить, содержат ли геометрические признаки локальной
# траектории информацию о скаляре n без прямого перебора.
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
# Важно:
# Никакая таблица соответствий (x,y) -> n не используется
# в качестве признаков модели.
# Перебор точек нужен только для создания экспериментальной
# последовательности и получения контрольных меток.
# ============================================================


# ----------------------------
# ПАРАМЕТРЫ
# ----------------------------

P = 499
A = 0
B = 7

G = (3, 212)
ORDER = 73

WINDOW = 2

PERMUTATIONS = 1000
RANDOM_SEED = 20261102

OUTPUT_DIR = Path("experimentC7_results")


# ----------------------------
# АРИФМЕТИКА КРИВОЙ
# ----------------------------

O = None


def inverse_mod(a, p):
    a %= p

    if a == 0:
        raise ZeroDivisionError("Обратного элемента не существует")

    return pow(a, -1, p)


def point_add(P1, P2):
    if P1 is None:
        return P2

    if P2 is None:
        return P1

    x1, y1 = P1
    x2, y2 = P2

    if x1 == x2 and (y1 + y2) % P == 0:
        return None

    if P1 == P2:
        if y1 % P == 0:
            return None

        numerator = (3 * x1 * x1 + A) % P
        denominator = (2 * y1) % P

    else:
        numerator = (y2 - y1) % P
        denominator = (x2 - x1) % P

    lam = (
        numerator * inverse_mod(denominator, P)
    ) % P

    x3 = (lam * lam - x1 - x2) % P
    y3 = (lam * (x1 - x3) - y1) % P

    return x3, y3


def scalar_mult(k, point):
    result = None
    addend = point

    while k > 0:
        if k & 1:
            result = point_add(result, addend)

        addend = point_add(addend, addend)
        k >>= 1

    return result


# ----------------------------
# ПОСТРОЕНИЕ ЦИКЛА
# ----------------------------

def build_cycle():

    cycle = []

    current = None

    for n in range(ORDER):

        cycle.append(current)
        current = point_add(current, G)

    if current is not None:
        raise RuntimeError("Порядок генератора указан неверно")

    return cycle


# ----------------------------
# ПРОВЕРКА КРИВОЙ
# ----------------------------

def verify_point(point):

    if point is None:
        return True

    x, y = point

    return (
        y * y - (x * x * x + B)
    ) % P == 0


# ----------------------------
# МОДУЛЬНОЕ РАССТОЯНИЕ
# ----------------------------

def circular_delta(a, b):

    """
    Кратчайшая знаковая разность на цикле Z/PZ.

    Результат лежит приблизительно в диапазоне
    [-P/2, P/2].
    """

    d = (a - b) % P

    if d > P / 2:
        d -= P

    return float(d)


def modular_distance(P1, P2):

    dx = circular_delta(P1[0], P2[0])
    dy = circular_delta(P1[1], P2[1])

    return math.sqrt(dx * dx + dy * dy)


# ----------------------------
# ОБЫЧНОЕ ЕВКЛИДОВО РАССТОЯНИЕ
# ----------------------------

def euclidean_distance(P1, P2):

    dx = P1[0] - P2[0]
    dy = P1[1] - P2[1]

    return math.sqrt(dx * dx + dy * dy)


# ----------------------------
# ГЕОМЕТРИЧЕСКИЕ ПРИЗНАКИ
# ----------------------------

def angle_between(v1, v2):

    norm1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    norm2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    dot = v1[0] * v2[0] + v1[1] * v2[1]

    cosine = dot / (norm1 * norm2)
    cosine = max(-1.0, min(1.0, cosine))

    return math.acos(cosine)


def triangle_area(a, b, c):

    return abs(
        a[0] * (b[1] - c[1]) +
        b[0] * (c[1] - a[1]) +
        c[0] * (a[1] - b[1])
    ) / 2.0


def make_features(window):

    """
    window:
        [P(n-2), P(n-1), P(n), P(n+1), P(n+2)]

    Все признаки строятся только по координатам точек.
    Значение n не передаётся в эту функцию.
    """

    center = window[2]

    points = [
        (float(q[0]), float(q[1]))
        for q in window
    ]

    features = {}

    # ----------------------------------------
    # M1. Исходные координаты центральной точки
    # ----------------------------------------

    features["xy"] = [
        center[0],
        center[1]
    ]

    # ----------------------------------------
    # M2. Разности соседних координат
    # ----------------------------------------

    diffs = []

    for i in range(4):

        dx = points[i + 1][0] - points[i][0]
        dy = points[i + 1][1] - points[i][1]

        diffs.extend([dx, dy])

    features["coordinate_differences"] = diffs

    # ----------------------------------------
    # M3. Модульные разности
    # ----------------------------------------

    modular_diffs = []

    for i in range(4):

        dx = circular_delta(
            window[i + 1][0],
            window[i][0]
        )

        dy = circular_delta(
            window[i + 1][1],
            window[i][1]
        )

        modular_diffs.extend([dx, dy])

    features["modular_differences"] = modular_diffs

    # ----------------------------------------
    # M4. Евклидовы расстояния от центра
    # ----------------------------------------

    distances = []

    for i in range(5):

        if i == 2:
            continue

        distances.append(
            euclidean_distance(
                points[2],
                points[i]
            )
        )

    features["euclidean_distances"] = distances

    # ----------------------------------------
    # M5. Модульные расстояния от центра
    # ----------------------------------------

    modular_distances = []

    for i in range(5):

        if i == 2:
            continue

        modular_distances.append(
            modular_distance(
                window[2],
                window[i]
            )
        )

    features["modular_distances"] = modular_distances

    # ----------------------------------------
    # M6. Векторы относительно центра
    # ----------------------------------------

    vectors = []

    for i in range(5):

        if i == 2:
            continue

        dx = points[i][0] - points[2][0]
        dy = points[i][1] - points[2][1]

        vectors.extend([dx, dy])

    features["centered_vectors"] = vectors

    # ----------------------------------------
    # M7. Углы между последовательными векторами
    # ----------------------------------------

    local_vectors = []

    for i in range(4):

        local_vectors.append((
            points[i + 1][0] - points[i][0],
            points[i + 1][1] - points[i][1]
        ))

    angles = []

    for i in range(3):

        angles.append(
            angle_between(
                local_vectors[i],
                local_vectors[i + 1]
            )
        )

    features["angles"] = angles

    # ----------------------------------------
    # M8. Площади треугольников
    # ----------------------------------------

    areas = []

    areas.append(
        triangle_area(
            points[0],
            points[2],
            points[4]
        )
    )

    areas.append(
        triangle_area(
            points[1],
            points[2],
            points[3]
        )
    )

    areas.append(
        triangle_area(
            points[0],
            points[1],
            points[2]
        )
    )

    areas.append(
        triangle_area(
            points[2],
            points[3],
            points[4]
        )
    )

    features["areas"] = areas

    # ----------------------------------------
    # M9. Отношения расстояний
    # ----------------------------------------

    ratios = []

    for i in range(4):

        d1 = euclidean_distance(
            points[2],
            points[i]
        )

        d2 = euclidean_distance(
            points[2],
            points[i + 1]
        )

        if d2 == 0:
            ratios.append(0.0)
        else:
            ratios.append(d1 / d2)

    features["distance_ratios"] = ratios

    # ----------------------------------------
    # M10. Нормированные геометрические признаки
    # ----------------------------------------

    scale = max(
        distances + [1.0]
    )

    normalized = []

    for value in distances:
        normalized.append(value / scale)

    for value in areas:
        normalized.append(value / (scale * scale))

    features["normalized_geometry"] = normalized

    return features


# ----------------------------
# ПОДГОТОВКА ВЫБОРКИ
# ----------------------------

def build_dataset(cycle):

    X = {
        "coordinates_xy": [],
        "coordinate_differences": [],
        "modular_differences": [],
        "euclidean_distances": [],
        "modular_distances": [],
        "centered_vectors": [],
        "angles": [],
        "areas": [],
        "distance_ratios": [],
        "normalized_geometry": [],
    }

    labels = []
    sample_indices = []

    # Используем только окна, которые не содержат O.
    # Одинаковый набор образцов применяется для всех моделей.

    for n in range(1, ORDER):

        indices = [
            (n - 2) % ORDER,
            (n - 1) % ORDER,
            n % ORDER,
            (n + 1) % ORDER,
            (n + 2) % ORDER
        ]

        window = [
            cycle[i]
            for i in indices
        ]

        if any(point is None for point in window):
            continue

        features = make_features(window)

        for key in X:
            X[key].append(features[
                "xy" if key == "coordinates_xy" else key
            ])

        labels.append(n)
        sample_indices.append(n)

    for key in X:
        X[key] = np.asarray(
            X[key],
            dtype=float
        )

    return (
        X,
        np.asarray(labels, dtype=int),
        np.asarray(sample_indices, dtype=int)
    )


# ----------------------------
# КРУГОВАЯ ЦЕЛЬ
# ----------------------------

def make_circular_target(labels):

    angles = 2 * np.pi * labels / ORDER

    return np.column_stack([
        np.cos(angles),
        np.sin(angles)
    ])


def circular_error(predicted, actual):

    predicted_angles = np.arctan2(
        predicted[:, 1],
        predicted[:, 0]
    )

    actual_angles = np.arctan2(
        actual[:, 1],
        actual[:, 0]
    )

    delta = predicted_angles - actual_angles

    delta = np.arctan2(
        np.sin(delta),
        np.cos(delta)
    )

    return np.abs(delta) * ORDER / (2 * np.pi)


# ----------------------------
# RIDGE
# ----------------------------

def ridge_predict(X_train, y_train, X_test, alpha=1.0):

    mean = X_train.mean(axis=0)

    std = X_train.std(axis=0)
    std[std < 1e-12] = 1.0

    Xtr = (X_train - mean) / std
    Xte = (X_test - mean) / std

    y_mean = y_train.mean(axis=0)
    ytr = y_train - y_mean

    identity = np.eye(Xtr.shape[1])

    weights = np.linalg.solve(
        Xtr.T @ Xtr + alpha * identity,
        Xtr.T @ ytr
    )

    return Xte @ weights + y_mean


def loo_evaluate(X, labels):

    target = make_circular_target(labels)

    predictions = []

    for i in range(len(X)):

        train_mask = np.ones(
            len(X),
            dtype=bool
        )

        train_mask[i] = False

        prediction = ridge_predict(
            X[train_mask],
            target[train_mask],
            X[i:i + 1]
        )

        predictions.append(prediction[0])

    predictions = np.asarray(predictions)

    predicted_angles = np.arctan2(
        predictions[:, 1],
        predictions[:, 0]
    )

    predicted_labels = (
        np.rint(
            predicted_angles * ORDER / (2 * np.pi)
        ).astype(int) % ORDER
    )

    accuracy = np.mean(
        predicted_labels == labels
    )

    errors = circular_error(
        predictions,
        target
    )

    return {
        "accuracy": float(accuracy),
        "circular_mae": float(np.mean(errors)),
        "predictions": predictions,
        "predicted_labels": predicted_labels
    }


# ----------------------------
# ПЕРЕСТАНОВОЧНЫЙ КОНТРОЛЬ
# ----------------------------

def permutation_test(X, labels, permutations, rng):

    observed = loo_evaluate(
        X,
        labels
    )

    observed_accuracy = observed["accuracy"]

    random_accuracies = []

    for _ in range(permutations):

        shuffled = rng.permutation(labels)

        result = loo_evaluate(
            X,
            shuffled
        )

        random_accuracies.append(
            result["accuracy"]
        )

    random_accuracies = np.asarray(
        random_accuracies
    )

    p_value = (
        1 + np.sum(
            random_accuracies >= observed_accuracy
        )
    ) / (permutations + 1)

    return (
        observed,
        random_accuracies,
        float(p_value)
    )


# ----------------------------
# СОХРАНЕНИЕ CSV
# ----------------------------

def save_csv(path, headers, rows):

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(headers)
        writer.writerows(rows)


# ----------------------------
# ГЛАВНАЯ ФУНКЦИЯ
# ----------------------------

def main():

    print("Запуск эксперимента C7...")
    print("Геометрические представления траектории")
    print("=" * 78)

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    cycle = build_cycle()

    for point in cycle:

        if not verify_point(point):
            raise RuntimeError(
                f"Обнаружена точка вне кривой: {point}"
            )

    print(f"Кривая: y^2 = x^3 + 7 (mod {P})")
    print(f"Генератор: {G}")
    print(f"Порядок: {ORDER}")

    X, labels, indices = build_dataset(cycle)

    print()
    print("Общая выборка:")
    print(f"Количество образцов: {len(labels)}")
    print(f"Центральные скаляры: {indices.tolist()}")

    models = {
        "coordinates_xy": [
            "coordinates_xy"
        ],

        "coordinate_differences": [
            "coordinate_differences"
        ],

        "modular_differences": [
            "modular_differences"
        ],

        "euclidean_distances": [
            "euclidean_distances"
        ],

        "modular_distances": [
            "modular_distances"
        ],

        "centered_vectors": [
            "centered_vectors"
        ],

        "angles": [
            "angles"
        ],

        "areas": [
            "areas"
        ],

        "distance_ratios": [
            "distance_ratios"
        ],

        "normalized_geometry": [
            "normalized_geometry"
        ],

        "all_features": list(X.keys())
    }

    results = []
    detailed_results = []

    print()
    print("ОСНОВНОЕ СРАВНЕНИЕ")
    print("-" * 78)

    print(
        f"{'Модель':30s}"
        f"{'Признаки':>10s}"
        f"{'N':>7s}"
        f"{'Accuracy':>12s}"
        f"{'Circular MAE':>16s}"
    )

    for model_name, feature_names in models.items():

        matrix = np.column_stack([
            X[name]
            for name in feature_names
        ])

        result = loo_evaluate(
            matrix,
            labels
        )

        results.append([
            model_name,
            matrix.shape[1],
            len(labels),
            result["accuracy"],
            result["circular_mae"]
        ])

        print(
            f"{model_name:30s}"
            f"{matrix.shape[1]:10d}"
            f"{len(labels):7d}"
            f"{result['accuracy']:12.4f}"
            f"{result['circular_mae']:16.3f}"
        )

        for i in range(len(labels)):

            detailed_results.append([
                model_name,
                int(labels[i]),
                int(result["predicted_labels"][i]),
                float(result["predicted_labels"][i] == labels[i])
            ])

    # ----------------------------------------
    # ЛУЧШАЯ МОДЕЛЬ
    # ----------------------------------------

    best = max(
        results,
        key=lambda row: (
            row[3],
            -row[4]
        )
    )

    best_name = best[0]

    best_features = models[best_name]

    best_matrix = np.column_stack([
        X[name]
        for name in best_features
    ])

    print()
    print("ПЕРЕСТАНОВОЧНЫЙ КОНТРОЛЬ")
    print("-" * 78)

    observed, random_acc, p_value = permutation_test(
        best_matrix,
        labels,
        PERMUTATIONS,
        rng
    )

    print(f"Лучшая модель: {best_name}")
    print(f"Наблюдаемая accuracy: {observed['accuracy']:.4f}")
    print(f"Средняя случайная accuracy: {random_acc.mean():.4f}")
    print(f"Максимальная случайная accuracy: {random_acc.max():.4f}")
    print(f"Permutation p-value: {p_value:.4f}")
    print(f"Circular MAE: {observed['circular_mae']:.3f}")

    # ----------------------------------------
    # СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
    # ----------------------------------------

    save_csv(
        OUTPUT_DIR / "C7_model_comparison.csv",
        [
            "model",
            "feature_count",
            "sample_count",
            "accuracy",
            "circular_mae"
        ],
        results
    )

    save_csv(
        OUTPUT_DIR / "C7_predictions.csv",
        [
            "model",
            "actual_scalar",
            "predicted_scalar",
            "correct"
        ],
        detailed_results
    )

    save_csv(
        OUTPUT_DIR / "C7_permutation.csv",
        [
            "permutation",
            "accuracy"
        ],
        [
            [i + 1, value]
            for i, value in enumerate(random_acc)
        ]
    )

    report = []

    report.append("ЭКСПЕРИМЕНТ C7")
    report.append("=" * 60)
    report.append(f"Дата: {datetime.now().isoformat()}")
    report.append(f"Кривая: y^2 = x^3 + 7 mod {P}")
    report.append(f"Генератор: {G}")
    report.append(f"Порядок: {ORDER}")
    report.append(f"Размер выборки: {len(labels)}")
    report.append("")
    report.append("СРАВНЕНИЕ МОДЕЛЕЙ")

    for row in results:

        report.append(
            f"{row[0]} | "
            f"features={row[1]} | "
            f"N={row[2]} | "
            f"accuracy={row[3]:.4f} | "
            f"MAE={row[4]:.4f}"
        )

    report.append("")
    report.append("ПЕРЕСТАНОВОЧНЫЙ КОНТРОЛЬ")
    report.append(f"Лучшая модель: {best_name}")
    report.append(f"Observed accuracy: {observed['accuracy']:.4f}")
    report.append(f"Random mean: {random_acc.mean():.4f}")
    report.append(f"Permutation p-value: {p_value:.4f}")
    report.append(f"Circular MAE: {observed['circular_mae']:.4f}")

    report.append("")
    report.append("МЕТОДОЛОГИЧЕСКИЕ ОГРАНИЧЕНИЯ")
    report.append(
        "1. Игрушечная группа содержит только 73 элемента."
    )
    report.append(
        "2. Координаты являются представителями классов по модулю P."
    )
    report.append(
        "3. Евклидовы признаки зависят от выбора числового представления."
    )
    report.append(
        "4. Высокая точность на этой группе не доказывает переносимость "
        "на криптографические кривые."
    )
    report.append(
        "5. Эксперимент не использует таблицу скалярных соответствий "
        "как входные признаки модели."
    )

    with open(
        OUTPUT_DIR / "C7_report.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write("\n".join(report))

    # ----------------------------------------
    # ZIP-АРХИВ
    # ----------------------------------------

    archive_path = Path("experimentC7.zip")

    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED
    ) as archive:

        for file_path in OUTPUT_DIR.iterdir():

            if file_path.is_file():

                archive.write(
                    file_path,
                    arcname=(
                        OUTPUT_DIR.name +
                        "/" +
                        file_path.name
                    )
                )

    print()
    print("РЕЗУЛЬТАТЫ СОХРАНЕНЫ:")
    print(OUTPUT_DIR.resolve())

    print()
    print("ZIP-АРХИВ:")
    print(archive_path.resolve())

    print()
    print("C7 ЗАВЕРШЁН.")


if __name__ == "__main__":
    main()
