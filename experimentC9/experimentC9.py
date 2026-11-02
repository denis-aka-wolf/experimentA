
import csv
import math
import zipfile
from pathlib import Path
from datetime import datetime

import numpy as np


# ============================================================
# C9. ГЕОМЕТРИЧЕСКАЯ ТРИАНГУЛЯЦИЯ ПО ОПОРНЫМ ТОЧКАМ
#
# Цель:
# Аппроксимировать скаляр n по геометрическим отношениям
# неизвестной точки Q=nG к небольшому числу опорных точек.
#
# Опорные точки:
# A_j = a_j * G
#
# Признаки:
# 1. Расстояния от Q до опорных точек.
# 2. Знаковые модульные смещения координат Q относительно
#    опорных точек.
#
# Модель:
# RBF Kernel Ridge Regression.
#
# Проверка:
# Blocked Cross-Validation с исключением целых участков
# цикла и защитной зоной вокруг тестового участка.
#
# Прямой перебор скаляра для предсказания не используется.
# ============================================================


# ----------------------------
# ПАРАМЕТРЫ
# ----------------------------

P = 499
A = 0
B = 7

G = (3, 212)
ORDER = 73

ANCHOR_COUNTS = [4, 8]

GAMMA_MULTIPLIERS = [
    0.3,
    1.0,
    3.0
]

ALPHA_VALUES = [
    0.01,
    0.1
]

FOLD_COUNT = 6
PURGE_RADIUS = 2

PERMUTATIONS = 100

RANDOM_SEED = 20261102

OUTPUT_DIR = Path("experimentC9_results")
ARCHIVE_PATH = Path("experimentC9.zip")


# ============================================================
# АРИФМЕТИКА ЭЛЛИПТИЧЕСКОЙ КРИВОЙ
# ============================================================

O = None


def inverse_mod(value, modulus):

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

    if x1 == x2 and (y1 + y2) % P == 0:
        return None

    if P1 == P2:

        if y1 == 0:
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

    y3 = (
        lam * (x1 - x3) - y1
    ) % P

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


def verify_point(point):

    if point is None:
        return True

    x, y = point

    return (
        y * y - (x * x * x + B)
    ) % P == 0


# ============================================================
# ПОСТРОЕНИЕ ОПОРНЫХ ТОЧЕК
# ============================================================

def build_anchor_indices(count):

    """
    Равномерно распределяем небольшое число ориентиров
    по циклу скаляров.

    Ориентиры не являются таблицей всех точек.
    Их количество задаётся параметром count.
    """

    result = []

    for i in range(count):

        value = round(
            (i + 0.5) * ORDER / count
        )

        value %= ORDER

        if value not in result:
            result.append(value)

    return sorted(result)


def build_anchor_points(count):

    indices = build_anchor_indices(count)

    anchors = []

    for scalar in indices:

        point = scalar_mult(
            scalar,
            G
        )

        if point is None:
            raise RuntimeError(
                "Опорная точка не должна быть O"
            )

        anchors.append({
            "scalar": scalar,
            "point": point
        })

    return anchors


# ============================================================
# МОДУЛЬНАЯ ГЕОМЕТРИЯ
# ============================================================

def circular_delta(a, b):

    """
    Кратчайшая знаковая разность двух координат
    в конечном поле.
    """

    d = (a - b) % P

    if d > P / 2:
        d -= P

    return float(d)


def toroidal_distance(Q, anchor):

    dx = circular_delta(
        Q[0],
        anchor[0]
    )

    dy = circular_delta(
        Q[1],
        anchor[1]
    )

    return math.sqrt(
        dx * dx + dy * dy
    )


# ============================================================
# ПОСТРОЕНИЕ ПРИЗНАКОВ
# ============================================================

def build_anchor_features(Q, anchors):

    distances = []
    vectors = []

    for item in anchors:

        anchor = item["point"]

        dx = circular_delta(
            Q[0],
            anchor[0]
        )

        dy = circular_delta(
            Q[1],
            anchor[1]
        )

        distance = math.sqrt(
            dx * dx + dy * dy
        )

        # Нормируем значения для сопоставимости.

        distances.append(
            distance / P
        )

        vectors.extend([
            dx / P,
            dy / P
        ])

    return {
        "anchor_distances": distances,
        "anchor_vectors": vectors
    }


# ============================================================
# СОЗДАНИЕ НАБОРОВ ДАННЫХ
# ============================================================

def build_dataset():

    # Все точки цикла создаются только для эксперимента.
    # При предсказании модель получает геометрические
    # признаки, а не индекс точки в этом цикле.

    cycle = []

    current = None

    for n in range(ORDER):

        cycle.append(current)

        current = point_add(
            current,
            G
        )

    if current is not None:
        raise RuntimeError(
            "Порядок генератора указан неверно"
        )

    for point in cycle:

        if not verify_point(point):
            raise RuntimeError(
                f"Точка не лежит на кривой: {point}"
            )

    datasets = {}

    # Контрольная модель: координаты точки.
    xy = []
    labels = []

    # Для каждой конфигурации ориентиров создаём
    # отдельные геометрические представления.

    for anchor_count in ANCHOR_COUNTS:

        anchors = build_anchor_points(
            anchor_count
        )

        anchor_indices = [
            item["scalar"]
            for item in anchors
        ]

        distance_data = []
        vector_data = []

        local_labels = []

        for n in range(1, ORDER):

            # Опорные точки исключаем из целевой выборки.
            # Иначе нулевое расстояние до ориентира
            # может создать тривиальную подсказку.

            if n in anchor_indices:
                continue

            Q = cycle[n]

            if Q is None:
                continue

            features = build_anchor_features(
                Q,
                anchors
            )

            distance_data.append(
                features["anchor_distances"]
            )

            vector_data.append(
                features["anchor_vectors"]
            )

            local_labels.append(n)

        datasets[
            f"anchors_{anchor_count}_distances"
        ] = {
            "X": np.asarray(
                distance_data,
                dtype=float
            ),
            "labels": np.asarray(
                local_labels,
                dtype=int
            ),
            "anchor_indices": anchor_indices
        }

        datasets[
            f"anchors_{anchor_count}_vectors"
        ] = {
            "X": np.asarray(
                vector_data,
                dtype=float
            ),
            "labels": np.asarray(
                local_labels,
                dtype=int
            ),
            "anchor_indices": anchor_indices
        }

    # Контрольная модель по исходным координатам.

    for n in range(1, ORDER):

        if n in build_anchor_indices(ANCHOR_COUNTS[0]):
            continue

        Q = cycle[n]

        xy.append([
            Q[0] / P,
            Q[1] / P
        ])

        labels.append(n)

    datasets["xy_control"] = {
        "X": np.asarray(xy, dtype=float),
        "labels": np.asarray(labels, dtype=int),
        "anchor_indices": []
    }

    return datasets


# ============================================================
# КРУГОВАЯ ЦЕЛЬ
# ============================================================

def make_target(labels):

    angles = (
        2.0 * np.pi * labels / ORDER
    )

    return np.column_stack([
        np.cos(angles),
        np.sin(angles)
    ])


def decode_prediction(prediction):

    angles = np.arctan2(
        prediction[:, 1],
        prediction[:, 0]
    )

    return (
        np.rint(
            angles * ORDER / (2.0 * np.pi)
        ).astype(int) % ORDER
    )


def evaluate_predictions(prediction, labels):

    target = make_target(labels)

    predicted_labels = decode_prediction(
        prediction
    )

    accuracy = np.mean(
        predicted_labels == labels
    )

    predicted_angles = np.arctan2(
        prediction[:, 1],
        prediction[:, 0]
    )

    actual_angles = np.arctan2(
        target[:, 1],
        target[:, 0]
    )

    delta = (
        predicted_angles - actual_angles
    )

    delta = np.arctan2(
        np.sin(delta),
        np.cos(delta)
    )

    errors = (
        np.abs(delta) *
        ORDER /
        (2.0 * np.pi)
    )

    return {
        "accuracy": float(accuracy),
        "mae": float(np.mean(errors)),
        "predicted_labels": predicted_labels,
        "errors": errors
    }


# ============================================================
# BLOCKED CROSS-VALIDATION
# ============================================================

def cyclic_scalar_distance(a, b):

    d = abs(a - b)

    return min(
        d,
        ORDER - d
    )


def get_fold_indices(labels, fold_number):

    """
    Разбиваем полный цикл на непрерывные блоки.

    Каждый тестовый блок содержит примерно ORDER/FOLD_COUNT
    последовательных скаляров.

    Обучающие образцы, расположенные ближе PURGE_RADIUS
    к тестовому участку, исключаются.

    Это снижает риск того, что модель просто
    интерполирует между соседними точками.
    """

    block_size = math.ceil(
        ORDER / FOLD_COUNT
    )

    start = fold_number * block_size
    end = min(
        start + block_size,
        ORDER
    )

    test_mask = np.zeros(
        len(labels),
        dtype=bool
    )

    train_mask = np.ones(
        len(labels),
        dtype=bool
    )

    test_scalars = list(
        range(start, end)
    )

    for i, n in enumerate(labels):

        if start <= n < end:

            test_mask[i] = True
            train_mask[i] = False

        else:

            for test_n in test_scalars:

                if cyclic_scalar_distance(
                    int(n),
                    int(test_n)
                ) <= PURGE_RADIUS:

                    train_mask[i] = False
                    break

    return (
        np.where(train_mask)[0],
        np.where(test_mask)[0]
    )


# ============================================================
# МАСШТАБИРОВАНИЕ ПРИЗНАКОВ
# ============================================================

def standardize_train_test(X_train, X_test):

    mean = X_train.mean(axis=0)

    std = X_train.std(axis=0)

    std[std < 1e-12] = 1.0

    return (
        (X_train - mean) / std,
        (X_test - mean) / std
    )


# ============================================================
# RBF KERNEL
# ============================================================

def squared_distances(A, B):

    norm_a = np.sum(
        A * A,
        axis=1
    )

    norm_b = np.sum(
        B * B,
        axis=1
    )

    result = (
        norm_a[:, None] +
        norm_b[None, :] -
        2.0 * (A @ B.T)
    )

    return np.maximum(
        result,
        0.0
    )


def fit_predict_rbf(
    X_train,
    y_train,
    X_test,
    gamma_multiplier,
    alpha
):

    Xtr, Xte = standardize_train_test(
        X_train,
        X_test
    )

    train_distances = squared_distances(
        Xtr,
        Xtr
    )

    test_distances = squared_distances(
        Xte,
        Xtr
    )

    upper = train_distances[
        np.triu_indices(
            len(Xtr),
            k=1
        )
    ]

    positive = upper[
        upper > 1e-12
    ]

    if len(positive) == 0:
        median_distance = 1.0

    else:
        median_distance = float(
            np.median(positive)
        )

    gamma = (
        gamma_multiplier /
        median_distance
    )

    K_train = np.exp(
        -gamma * train_distances
    )

    K_test = np.exp(
        -gamma * test_distances
    )

    matrix = (
        K_train +
        alpha * np.eye(len(X_train))
    )

    coefficients = np.linalg.solve(
        matrix,
        y_train
    )

    return K_test @ coefficients


# ============================================================
# BLOCKED CV ДЛЯ ОДНОЙ КОНФИГУРАЦИИ
# ============================================================

def evaluate_configuration(
    X,
    labels,
    gamma_multiplier,
    alpha
):

    target = make_target(labels)

    predictions = np.zeros_like(target)

    assigned = np.zeros(
        len(labels),
        dtype=bool
    )

    fold_rows = []

    for fold in range(FOLD_COUNT):

        train_indices, test_indices = get_fold_indices(
            labels,
            fold
        )

        if len(train_indices) < 10:
            raise RuntimeError(
                "Слишком мало обучающих образцов"
            )

        if len(test_indices) == 0:
            continue

        fold_prediction = fit_predict_rbf(
            X[train_indices],
            target[train_indices],
            X[test_indices],
            gamma_multiplier,
            alpha
        )

        predictions[test_indices] = fold_prediction

        assigned[test_indices] = True

        fold_rows.append({
            "fold": fold + 1,
            "train_count": len(train_indices),
            "test_count": len(test_indices)
        })

    if not np.all(assigned):
        raise RuntimeError(
            "Не все образцы получили предсказание"
        )

    result = evaluate_predictions(
        predictions,
        labels
    )

    result["predictions"] = predictions
    result["fold_rows"] = fold_rows

    return result


# ============================================================
# ПЕРЕБОР КОНФИГУРАЦИЙ
# ============================================================

def evaluate_all_configurations(datasets, label_override=None):

    results = []

    for dataset_name, dataset in datasets.items():

        X = dataset["X"]

        labels = dataset["labels"]

        if label_override is not None:

            labels = label_override[dataset_name]

        for gamma in GAMMA_MULTIPLIERS:

            for alpha in ALPHA_VALUES:

                result = evaluate_configuration(
                    X,
                    labels,
                    gamma,
                    alpha
                )

                results.append({
                    "dataset": dataset_name,
                    "gamma": gamma,
                    "alpha": alpha,
                    "accuracy": result["accuracy"],
                    "mae": result["mae"],
                    "predictions": result["predictions"],
                    "predicted_labels": result["predicted_labels"],
                    "errors": result["errors"]
                })

    results.sort(
        key=lambda item: (
            item["mae"],
            -item["accuracy"]
        )
    )

    return results


# ============================================================
# ПЕРЕСТАНОВОЧНЫЙ КОНТРОЛЬ
# ============================================================

def permutation_test(datasets, permutations, rng):

    observed_results = evaluate_all_configurations(
        datasets
    )

    observed_best = observed_results[0]

    observed_mae = observed_best["mae"]
    observed_accuracy = observed_best["accuracy"]

    random_mae = []
    random_accuracy = []

    for iteration in range(permutations):

        # Для каждого набора данных отдельно
        # переставляем скалярные метки.
        #
        # Геометрические признаки и разбиение на блоки
        # остаются неизменными.

        shuffled_labels = {}

        for name, dataset in datasets.items():

            shuffled_labels[name] = rng.permutation(
                dataset["labels"]
            )

        permuted_results = evaluate_all_configurations(
            datasets,
            shuffled_labels
        )

        best = permuted_results[0]

        random_mae.append(
            best["mae"]
        )

        random_accuracy.append(
            best["accuracy"]
        )

        if (iteration + 1) % 20 == 0:

            print(
                f"Перестановки: "
                f"{iteration + 1}/{permutations}"
            )

    random_mae = np.asarray(
        random_mae
    )

    random_accuracy = np.asarray(
        random_accuracy
    )

    p_mae = (
        1 + np.sum(
            random_mae <= observed_mae
        )
    ) / (permutations + 1)

    p_accuracy = (
        1 + np.sum(
            random_accuracy >= observed_accuracy
        )
    ) / (permutations + 1)

    return {
        "observed_best": observed_best,
        "all_results": observed_results,
        "random_mae": random_mae,
        "random_accuracy": random_accuracy,
        "p_mae": float(p_mae),
        "p_accuracy": float(p_accuracy)
    }


# ============================================================
# СОХРАНЕНИЕ CSV
# ============================================================

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


# ============================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================

def main():

    print("Запуск эксперимента C9...")
    print("Геометрическая триангуляция по опорным точкам")
    print("=" * 78)

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not verify_point(G):
        raise RuntimeError(
            "Генератор не лежит на кривой"
        )

    if scalar_mult(ORDER, G) is not None:
        raise RuntimeError(
            "Порядок генератора указан неверно"
        )

    print(
        f"Кривая: y^2 = x^3 + 7 (mod {P})"
    )

    print(
        f"Генератор: {G}"
    )

    print(
        f"Порядок: {ORDER}"
    )

    # ----------------------------------------
    # Построение данных
    # ----------------------------------------

    datasets = build_dataset()

    print()
    print("ГЕОМЕТРИЧЕСКИЕ ПРЕДСТАВЛЕНИЯ")
    print("-" * 78)

    for name, dataset in datasets.items():

        print(
            f"{name:30s}"
            f"Признаки: {dataset['X'].shape[1]:3d}"
            f" | Образцов: {len(dataset['labels']):3d}"
        )

        if dataset["anchor_indices"]:

            print(
                "Опорные скаляры:",
                dataset["anchor_indices"]
            )

    # ----------------------------------------
    # Основное сравнение
    # ----------------------------------------

    print()
    print("BLOCKED CROSS-VALIDATION")
    print("-" * 78)

    results = evaluate_all_configurations(
        datasets
    )

    print(
        f"{'Представление':30s}"
        f"{'Gamma':>10s}"
        f"{'Alpha':>10s}"
        f"{'Accuracy':>12s}"
        f"{'Circular MAE':>16s}"
    )

    for item in results:

        print(
            f"{item['dataset']:30s}"
            f"{item['gamma']:10.3f}"
            f"{item['alpha']:10.3f}"
            f"{item['accuracy']:12.4f}"
            f"{item['mae']:16.3f}"
        )

    # ----------------------------------------
    # Перестановочный контроль
    # ----------------------------------------

    print()
    print("ПЕРЕСТАНОВОЧНЫЙ КОНТРОЛЬ")
    print("-" * 78)

    permutation = permutation_test(
        datasets,
        PERMUTATIONS,
        rng
    )

    best = permutation["observed_best"]

    print()
    print("ЛУЧШАЯ КОНФИГУРАЦИЯ")
    print("-" * 78)

    print(
        f"Представление: {best['dataset']}"
    )

    print(
        f"Gamma: {best['gamma']}"
    )

    print(
        f"Alpha: {best['alpha']}"
    )

    print(
        f"Accuracy: {best['accuracy']:.4f}"
    )

    print(
        f"Circular MAE: {best['mae']:.4f}"
    )

    print()
    print("СТАТИСТИЧЕСКАЯ ОЦЕНКА")
    print("-" * 78)

    print(
        f"Средняя случайная MAE: "
        f"{permutation['random_mae'].mean():.4f}"
    )

    print(
        f"Минимальная случайная MAE: "
        f"{permutation['random_mae'].min():.4f}"
    )

    print(
        f"Permutation p-value MAE: "
        f"{permutation['p_mae']:.4f}"
    )

    print(
        f"Средняя случайная Accuracy: "
        f"{permutation['random_accuracy'].mean():.4f}"
    )

    print(
        f"Permutation p-value Accuracy: "
        f"{permutation['p_accuracy']:.4f}"
    )

    # ----------------------------------------
    # Сохранение сравнения
    # ----------------------------------------

    comparison_rows = []

    for item in permutation["all_results"]:

        comparison_rows.append([
            item["dataset"],
            item["gamma"],
            item["alpha"],
            item["accuracy"],
            item["mae"]
        ])

    save_csv(
        OUTPUT_DIR / "C9_comparison.csv",
        [
            "representation",
            "gamma",
            "alpha",
            "accuracy",
            "circular_mae"
        ],
        comparison_rows
    )

    # ----------------------------------------
    # Сохранение предсказаний
    # ----------------------------------------

    prediction_rows = []

    for dataset_name, dataset in datasets.items():

        matching = [
            item
            for item in permutation["all_results"]
            if item["dataset"] == dataset_name
        ]

        if not matching:
            continue

        best_for_dataset = matching[0]

        for i, actual in enumerate(dataset["labels"]):

            predicted = int(
                best_for_dataset["predicted_labels"][i]
            )

            prediction_rows.append([
                dataset_name,
                int(actual),
                predicted,
                int(actual == predicted),
                float(best_for_dataset["errors"][i])
            ])

    save_csv(
        OUTPUT_DIR / "C9_predictions.csv",
        [
            "representation",
            "actual_scalar",
            "predicted_scalar",
            "correct",
            "circular_error"
        ],
        prediction_rows
    )

    # ----------------------------------------
    # Перестановочные результаты
    # ----------------------------------------

    save_csv(
        OUTPUT_DIR / "C9_permutation.csv",
        [
            "permutation",
            "best_mae",
            "best_accuracy"
        ],
        [
            [
                i + 1,
                permutation["random_mae"][i],
                permutation["random_accuracy"][i]
            ]
            for i in range(PERMUTATIONS)
        ]
    )

    # ----------------------------------------
    # Текстовый отчёт
    # ----------------------------------------

    report = []

    report.append("ЭКСПЕРИМЕНТ C9")
    report.append("=" * 70)

    report.append(
        f"Дата: {datetime.now().isoformat()}"
    )

    report.append(
        f"Кривая: y^2 = x^3 + 7 mod {P}"
    )

    report.append(
        f"Генератор: {G}"
    )

    report.append(
        f"Порядок: {ORDER}"
    )

    report.append("")

    report.append("BLOCKED CROSS-VALIDATION")

    for item in permutation["all_results"]:

        report.append(
            f"{item['dataset']} | "
            f"gamma={item['gamma']} | "
            f"alpha={item['alpha']} | "
            f"accuracy={item['accuracy']:.4f} | "
            f"MAE={item['mae']:.4f}"
        )

    report.append("")
    report.append("ЛУЧШАЯ КОНФИГУРАЦИЯ")

    report.append(
        f"Представление: {best['dataset']}"
    )

    report.append(
        f"Gamma: {best['gamma']}"
    )

    report.append(
        f"Alpha: {best['alpha']}"
    )

    report.append(
        f"Accuracy: {best['accuracy']}"
    )

    report.append(
        f"Circular MAE: {best['mae']}"
    )

    report.append("")
    report.append("ПЕРЕСТАНОВОЧНЫЙ КОНТРОЛЬ")

    report.append(
        f"Random MAE mean: "
        f"{permutation['random_mae'].mean()}"
    )

    report.append(
        f"Random MAE min: "
        f"{permutation['random_mae'].min()}"
    )

    report.append(
        f"Permutation p-value MAE: "
        f"{permutation['p_mae']}"
    )

    report.append(
        f"Random Accuracy mean: "
        f"{permutation['random_accuracy'].mean()}"
    )

    report.append(
        f"Permutation p-value Accuracy: "
        f"{permutation['p_accuracy']}"
    )

    report.append("")
    report.append("МЕТОДОЛОГИЧЕСКИЕ ОГРАНИЧЕНИЯ")

    report.append(
        "1. Используется небольшая игрушечная группа порядка 73."
    )

    report.append(
        "2. Опорные точки являются известными кратными генератора."
    )

    report.append(
        "3. Число ориентиров существенно меньше порядка группы."
    )

    report.append(
        "4. Геометрические расстояния зависят от представления "
        "координат в конечном поле."
    )

    report.append(
        "5. Blocked CV исключает тестовый участок и защитную "
        "зону вокруг него."
    )

    report.append(
        "6. Перестановочный контроль учитывает выбор лучшей "
        "конфигурации среди проверенных вариантов."
    )

    report.append(
        "7. Результат не доказывает возможность восстановления "
        "скаляра на secp256k1."
    )

    with open(
        OUTPUT_DIR / "C9_report.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "\n".join(report)
        )

    # ----------------------------------------
    # ZIP-АРХИВ
    # ----------------------------------------

    with zipfile.ZipFile(
        ARCHIVE_PATH,
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
    print(ARCHIVE_PATH.resolve())

    print()
    print("C9 ЗАВЕРШЁН.")


if __name__ == "__main__":
    main()
