
import csv
import math
import zipfile
from pathlib import Path
from datetime import datetime

import numpy as np


# ============================================================
# ЭКСПЕРИМЕНТ C8
#
# Нелинейная аппроксимация скаляра через локальную
# алгебраическую геометрию точки эллиптической кривой.
#
# Кривая:
# y^2 = x^3 + 7 (mod 499)
#
# Генератор:
# G = (3, 212)
#
# Порядок генератора:
# 73
#
# Основная идея:
# По точке Q = nG строим небольшую конфигурацию:
#
# Q-2G, Q-G, Q, Q+G, Q+2G
#
# Из неё извлекаем геометрические и алгебраические
# признаки, после чего пытаемся аппроксимировать
# круговую фазу скаляра нелинейной RBF-моделью.
#
# Прямой перебор скаляра для предсказания не используется.
# Таблица соответствий точек скалярам не используется
# в качестве признаков.
#
# Для оценки применяется Leave-One-Out Cross-Validation.
# Перестановочный тест повторяет весь процесс выбора модели.
# ============================================================


# ============================================================
# ПАРАМЕТРЫ
# ============================================================

P = 499
A = 0
B = 7

G = (3, 212)
ORDER = 73

WINDOW = 2

PERMUTATIONS = 500

RANDOM_SEED = 20261102

OUTPUT_DIR = Path("experimentC8_results")
ARCHIVE_PATH = Path("experimentC8.zip")


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

    y3 = (
        lam * (x1 - x3) - y1
    ) % P

    return x3, y3


def point_negate(point):

    if point is None:
        return None

    return (
        point[0],
        (-point[1]) % P
    )


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
# ПОСТРОЕНИЕ ГЕНЕРАТОРА И СДВИГОВ
# ============================================================

def build_offsets():

    offsets = {}

    for j in range(-WINDOW, WINDOW + 1):

        if j >= 0:
            offsets[j] = scalar_mult(j, G)

        else:
            offsets[j] = point_negate(
                scalar_mult(-j, G)
            )

    return offsets


# ============================================================
# ЦИКЛИЧЕСКОЕ КОДИРОВАНИЕ КООРДИНАТ
# ============================================================

def encode_residue(value):

    """
    Кодируем элемент конечного поля через окружность.

    Значения 0 и P-1 оказываются рядом на окружности,
    а не на противоположных концах числовой шкалы.
    """

    angle = 2.0 * math.pi * (value % P) / P

    return [
        math.cos(angle),
        math.sin(angle)
    ]


def encode_point_torus(point):

    if point is None:
        raise ValueError(
            "Нельзя кодировать бесконечно удалённую точку"
        )

    return (
        encode_residue(point[0]) +
        encode_residue(point[1])
    )


# ============================================================
# АЛГЕБРАИЧЕСКИЕ ПРИЗНАКИ ПЕРЕХОДОВ
# ============================================================

def chord_slope(P1, P2):

    """
    Наклон секущей в конечном поле.

    Если разность x равна нулю, возвращаем специальный
    маркер P-1. Это позволяет отдельно представить
    вертикальный переход.
    """

    if P1 is None or P2 is None:
        raise ValueError("Нельзя вычислить наклон через O")

    dx = (P2[0] - P1[0]) % P
    dy = (P2[1] - P1[1]) % P

    if dx == 0:
        return P - 1

    return (
        dy * inverse_mod(dx, P)
    ) % P


def transition_signature(Q, offsets):

    """
    Строит признаки переходов от центральной точки Q
    к Q-G, Q+G, Q-2G и Q+2G.

    Используются только координаты и групповые операции.
    Скаляр n сюда не передаётся.
    """

    features = []

    for j in [-2, -1, 1, 2]:

        neighbor = point_add(
            Q,
            offsets[j]
        )

        if neighbor is None:
            raise ValueError(
                "В окне встретилась точка O"
            )

        dx = (neighbor[0] - Q[0]) % P
        dy = (neighbor[1] - Q[1]) % P

        slope = chord_slope(
            Q,
            neighbor
        )

        # Кодируем каждую величину как элемент Z/PZ.
        features.extend(
            encode_residue(dx)
        )

        features.extend(
            encode_residue(dy)
        )

        features.extend(
            encode_residue(slope)
        )

    return features


# ============================================================
# ФОРМИРОВАНИЕ ГЕОМЕТРИЧЕСКИХ ПРИЗНАКОВ
# ============================================================

def build_features(Q, offsets):

    points = {}

    for j in range(-WINDOW, WINDOW + 1):

        points[j] = point_add(
            Q,
            offsets[j]
        )

        if points[j] is None:
            raise ValueError(
                "В локальном окне встретилась точка O"
            )

    # ----------------------------------------
    # M1. Только центральная точка
    # ----------------------------------------

    central = encode_point_torus(Q)

    # ----------------------------------------
    # M2. Все пять точек на окружности
    # ----------------------------------------

    local_geometry = []

    for j in range(-WINDOW, WINDOW + 1):

        local_geometry.extend(
            encode_point_torus(points[j])
        )

    # ----------------------------------------
    # M3. Алгебраические переходы
    # ----------------------------------------

    algebraic = transition_signature(
        Q,
        offsets
    )

    # ----------------------------------------
    # M4. Совместное представление
    # ----------------------------------------

    combined = (
        local_geometry +
        algebraic
    )

    return {
        "central_torus": central,
        "local_torus": local_geometry,
        "algebraic_transitions": algebraic,
        "combined": combined
    }


# ============================================================
# СОЗДАНИЕ ВЫБОРКИ
# ============================================================

def build_dataset():

    offsets = build_offsets()

    feature_names = [
        "central_torus",
        "local_torus",
        "algebraic_transitions",
        "combined"
    ]

    data = {
        name: []
        for name in feature_names
    }

    labels = []
    sample_indices = []

    # Используем один и тот же диапазон скаляров
    # для всех моделей.
    #
    # При n=1,2,71,72 локальное окно может содержать O.
    # Поэтому используем n=3..70.

    for n in range(3, 71):

        Q = scalar_mult(n, G)

        if Q is None:
            continue

        features = build_features(
            Q,
            offsets
        )

        for name in feature_names:

            data[name].append(
                features[name]
            )

        labels.append(n)
        sample_indices.append(n)

    for name in feature_names:

        data[name] = np.asarray(
            data[name],
            dtype=float
        )

    return (
        data,
        np.asarray(labels, dtype=int),
        np.asarray(sample_indices, dtype=int)
    )


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

    labels = np.rint(
        angles * ORDER / (2.0 * np.pi)
    ).astype(int) % ORDER

    return labels


def circular_errors(prediction, target):

    predicted_angles = np.arctan2(
        prediction[:, 1],
        prediction[:, 0]
    )

    actual_angles = np.arctan2(
        target[:, 1],
        target[:, 0]
    )

    difference = (
        predicted_angles - actual_angles
    )

    difference = np.arctan2(
        np.sin(difference),
        np.cos(difference)
    )

    return (
        np.abs(difference) *
        ORDER /
        (2.0 * np.pi)
    )


# ============================================================
# ПРЕДОБРАБОТКА ПРИЗНАКОВ
# ============================================================

def standardize_features(X):

    """
    Масштабирование использует только сами признаки,
    не используя скалярные метки.

    Это позволяет подготовить матрицу расстояний
    до перестановочного теста.
    """

    mean = X.mean(axis=0)

    std = X.std(axis=0)

    std[std < 1e-12] = 1.0

    return (X - mean) / std


def pairwise_squared_distances(X):

    norms = np.sum(
        X * X,
        axis=1
    )

    distances = (
        norms[:, None] +
        norms[None, :] -
        2.0 * (X @ X.T)
    )

    return np.maximum(
        distances,
        0.0
    )


# ============================================================
# RBF KERNEL
# ============================================================

def build_kernel(distance_matrix, gamma):

    return np.exp(
        -gamma * distance_matrix
    )


# ============================================================
# LOO-ОПЕРАТОР ДЛЯ KERNEL RIDGE
# ============================================================

def build_loo_operator(kernel, alpha):

    """
    Для Kernel Ridge:

        A = K + alpha * I

        W = inverse(A)

        alpha_coefficients = W @ Y

    Формула LOO:

        prediction_i =
            Y_i - alpha_coefficients_i / W_ii

    Благодаря этому не нужно заново обучать модель
    для каждого исключённого образца.

    Получаем матрицу M, такую что:

        Y_LOO = M @ Y
    """

    count = kernel.shape[0]

    matrix = (
        kernel +
        alpha * np.eye(count)
    )

    inverse = np.linalg.inv(matrix)

    diagonal = np.diag(inverse)

    if np.any(np.abs(diagonal) < 1e-12):
        raise RuntimeError(
            "Нестабильная диагональ обратной матрицы"
        )

    operator = (
        np.eye(count) -
        inverse / diagonal[:, None]
    )

    return operator


# ============================================================
# ОЦЕНКА ПРЕДСКАЗАНИЯ
# ============================================================

def evaluate_prediction(prediction, labels):

    target = make_target(labels)

    predicted_labels = decode_prediction(
        prediction
    )

    accuracy = np.mean(
        predicted_labels == labels
    )

    errors = circular_errors(
        prediction,
        target
    )

    return {
        "accuracy": float(accuracy),
        "circular_mae": float(np.mean(errors)),
        "predicted_labels": predicted_labels,
        "errors": errors
    }


# ============================================================
# ПЕРЕБОР ПАРАМЕТРОВ НЕЛИНЕЙНОЙ МОДЕЛИ
# ============================================================

def prepare_model_grid(data):

    """
    Перебираем:
        4 группы признаков;
        5 значений gamma;
        4 значения alpha.

    Всего 80 конфигураций.

    Gamma задаётся относительно медианного
    попарного расстояния, чтобы масштабирование
    было сопоставимым для разных групп признаков.
    """

    gamma_multipliers = [
        0.1,
        0.3,
        1.0,
        3.0,
        10.0
    ]

    alpha_values = [
        0.0001,
        0.001,
        0.01,
        0.1
    ]

    configurations = []

    for feature_name, raw_X in data.items():

        X = standardize_features(raw_X)

        distances = pairwise_squared_distances(X)

        upper = distances[
            np.triu_indices(
                len(X),
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

        for multiplier in gamma_multipliers:

            gamma = (
                multiplier /
                median_distance
            )

            kernel = build_kernel(
                distances,
                gamma
            )

            for alpha in alpha_values:

                operator = build_loo_operator(
                    kernel,
                    alpha
                )

                configurations.append({
                    "feature_name": feature_name,
                    "feature_count": raw_X.shape[1],
                    "gamma_multiplier": multiplier,
                    "gamma": gamma,
                    "alpha": alpha,
                    "operator": operator
                })

    return configurations


# ============================================================
# ВЫБОР ЛУЧШЕЙ МОДЕЛИ
# ============================================================

def select_best_model(configurations, labels):

    target = make_target(labels)

    candidates = []

    for config in configurations:

        prediction = (
            config["operator"] @ target
        )

        result = evaluate_prediction(
            prediction,
            labels
        )

        candidates.append({
            "config": config,
            "prediction": prediction,
            "accuracy": result["accuracy"],
            "circular_mae": result["circular_mae"]
        })

    # Основной критерий:
    # минимальная Circular MAE.
    #
    # При равенстве выбираем большую Accuracy.

    candidates.sort(
        key=lambda item: (
            item["circular_mae"],
            -item["accuracy"]
        )
    )

    return candidates[0], candidates


# ============================================================
# ПЕРЕСТАНОВОЧНЫЙ КОНТРОЛЬ
# ============================================================

def permutation_test(
    configurations,
    labels,
    permutation_count,
    rng
):

    observed_best, observed_all = select_best_model(
        configurations,
        labels
    )

    observed_mae = observed_best["circular_mae"]
    observed_accuracy = observed_best["accuracy"]

    random_mae = []
    random_accuracy = []

    for iteration in range(permutation_count):

        shuffled_labels = rng.permutation(
            labels
        )

        best, all_models = select_best_model(
            configurations,
            shuffled_labels
        )

        random_mae.append(
            best["circular_mae"]
        )

        random_accuracy.append(
            best["accuracy"]
        )

        if (iteration + 1) % 100 == 0:

            print(
                f"Перестановки: "
                f"{iteration + 1}/{permutation_count}"
            )

    random_mae = np.asarray(
        random_mae
    )

    random_accuracy = np.asarray(
        random_accuracy
    )

    # Для MAE меньшие значения лучше.
    p_mae = (
        1 + np.sum(
            random_mae <= observed_mae
        )
    ) / (permutation_count + 1)

    # Для Accuracy большие значения лучше.
    p_accuracy = (
        1 + np.sum(
            random_accuracy >= observed_accuracy
        )
    ) / (permutation_count + 1)

    return {
        "observed_best": observed_best,
        "observed_all": observed_all,
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

    print("Запуск эксперимента C8...")
    print("Нелинейная аппроксимация скаляра")
    print("=" * 78)

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ----------------------------------------
    # Проверка кривой и генератора
    # ----------------------------------------

    if not verify_point(G):
        raise RuntimeError(
            "Генератор не лежит на кривой"
        )

    if scalar_mult(ORDER, G) is not None:
        raise RuntimeError(
            "Указанный порядок генератора неверен"
        )

    print(f"Кривая: y^2 = x^3 + 7 (mod {P})")
    print(f"Генератор: {G}")
    print(f"Порядок: {ORDER}")

    # ----------------------------------------
    # Создание выборки
    # ----------------------------------------

    data, labels, indices = build_dataset()

    print()
    print("ОБЩАЯ ВЫБОРКА")
    print("-" * 78)

    print(f"Образцов: {len(labels)}")
    print(f"Диапазон скаляров: {labels[0]}..{labels[-1]}")

    print(
        "Центральные скаляры:",
        labels.tolist()
    )

    # ----------------------------------------
    # Подготовка конфигураций
    # ----------------------------------------

    print()
    print("ПОДГОТОВКА НЕЛИНЕЙНЫХ МОДЕЛЕЙ")
    print("-" * 78)

    configurations = prepare_model_grid(
        data
    )

    print(
        f"Количество конфигураций: "
        f"{len(configurations)}"
    )

    # ----------------------------------------
    # Выбор лучшей конфигурации
    # ----------------------------------------

    best, all_models = select_best_model(
        configurations,
        labels
    )

    print()
    print("СРАВНЕНИЕ МОДЕЛЕЙ")
    print("-" * 78)

    print(
        f"{'Представление':25s}"
        f"{'Признаки':>10s}"
        f"{'Gamma x':>10s}"
        f"{'Alpha':>12s}"
        f"{'Accuracy':>12s}"
        f"{'Circular MAE':>16s}"
    )

    for item in all_models:

        config = item["config"]

        print(
            f"{config['feature_name']:25s}"
            f"{config['feature_count']:10d}"
            f"{config['gamma_multiplier']:10.3f}"
            f"{config['alpha']:12.4f}"
            f"{item['accuracy']:12.4f}"
            f"{item['circular_mae']:16.3f}"
        )

    # ----------------------------------------
    # Перестановочный контроль
    # ----------------------------------------

    print()
    print("ПЕРЕСТАНОВОЧНЫЙ КОНТРОЛЬ")
    print("-" * 78)

    permutation = permutation_test(
        configurations,
        labels,
        PERMUTATIONS,
        rng
    )

    best = permutation["observed_best"]
    best_config = best["config"]

    print()
    print("ЛУЧШАЯ КОНФИГУРАЦИЯ")
    print("-" * 78)

    print(
        f"Признаки: {best_config['feature_name']}"
    )

    print(
        f"Количество признаков: "
        f"{best_config['feature_count']}"
    )

    print(
        f"Gamma multiplier: "
        f"{best_config['gamma_multiplier']}"
    )

    print(
        f"Alpha: {best_config['alpha']}"
    )

    print(
        f"Accuracy: {best['accuracy']:.4f}"
    )

    print(
        f"Circular MAE: "
        f"{best['circular_mae']:.4f}"
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
        f"Permutation p-value (MAE): "
        f"{permutation['p_mae']:.4f}"
    )

    print(
        f"Средняя случайная Accuracy: "
        f"{permutation['random_accuracy'].mean():.4f}"
    )

    print(
        f"Permutation p-value (Accuracy): "
        f"{permutation['p_accuracy']:.4f}"
    )

    # ----------------------------------------
    # Подробные предсказания лучшей модели
    # ----------------------------------------

    prediction = best["prediction"]

    predicted_labels = decode_prediction(
        prediction
    )

    target = make_target(labels)

    errors = circular_errors(
        prediction,
        target
    )

    prediction_rows = []

    for i in range(len(labels)):

        prediction_rows.append([
            int(labels[i]),
            int(predicted_labels[i]),
            int(predicted_labels[i] == labels[i]),
            float(errors[i]),
            float(prediction[i, 0]),
            float(prediction[i, 1])
        ])

    save_csv(
        OUTPUT_DIR / "C8_predictions.csv",
        [
            "actual_scalar",
            "predicted_scalar",
            "correct",
            "circular_error",
            "predicted_cos",
            "predicted_sin"
        ],
        prediction_rows
    )

    # ----------------------------------------
    # Все конфигурации
    # ----------------------------------------

    config_rows = []

    for item in all_models:

        config = item["config"]

        config_rows.append([
            config["feature_name"],
            config["feature_count"],
            config["gamma_multiplier"],
            config["gamma"],
            config["alpha"],
            item["accuracy"],
            item["circular_mae"]
        ])

    save_csv(
        OUTPUT_DIR / "C8_model_comparison.csv",
        [
            "feature_group",
            "feature_count",
            "gamma_multiplier",
            "gamma",
            "alpha",
            "accuracy",
            "circular_mae"
        ],
        config_rows
    )

    # ----------------------------------------
    # Перестановочные результаты
    # ----------------------------------------

    permutation_rows = []

    for i in range(PERMUTATIONS):

        permutation_rows.append([
            i + 1,
            permutation["random_mae"][i],
            permutation["random_accuracy"][i]
        ])

    save_csv(
        OUTPUT_DIR / "C8_permutation.csv",
        [
            "permutation",
            "best_circular_mae",
            "selected_model_accuracy"
        ],
        permutation_rows
    )

    # ----------------------------------------
    # Текстовый отчёт
    # ----------------------------------------

    report = []

    report.append("ЭКСПЕРИМЕНТ C8")
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

    report.append(
        f"Количество образцов: {len(labels)}"
    )

    report.append("")

    report.append("ЛУЧШАЯ МОДЕЛЬ")

    report.append(
        f"Представление: {best_config['feature_name']}"
    )

    report.append(
        f"Признаков: {best_config['feature_count']}"
    )

    report.append(
        f"Gamma: {best_config['gamma']}"
    )

    report.append(
        f"Alpha: {best_config['alpha']}"
    )

    report.append(
        f"Accuracy: {best['accuracy']}"
    )

    report.append(
        f"Circular MAE: {best['circular_mae']}"
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
        f"Random accuracy mean: "
        f"{permutation['random_accuracy'].mean()}"
    )

    report.append(
        f"Permutation p-value Accuracy: "
        f"{permutation['p_accuracy']}"
    )

    report.append("")

    report.append("МЕТОДОЛОГИЧЕСКИЕ ОГРАНИЧЕНИЯ")

    report.append(
        "1. Выборка содержит только 68 точек."
    )

    report.append(
        "2. Нелинейная модель может переобучаться."
    )

    report.append(
        "3. Перестановочный тест учитывает выбор лучшей "
        "конфигурации среди всех проверенных вариантов."
    )

    report.append(
        "4. Все признаки строятся из Q и фиксированных "
        "сдвигов Q±G и Q±2G."
    )

    report.append(
        "5. Скаляр n не используется при построении признаков."
    )

    report.append(
        "6. Эксперимент не использует таблицу соответствий "
        "точек и скаляров как вход модели."
    )

    report.append(
        "7. Результат на игрушечной кривой не доказывает "
        "переносимость на secp256k1."
    )

    with open(
        OUTPUT_DIR / "C8_report.txt",
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

    # ----------------------------------------
    # ЗАВЕРШЕНИЕ
    # ----------------------------------------

    print()
    print("РЕЗУЛЬТАТЫ СОХРАНЕНЫ:")
    print(OUTPUT_DIR.resolve())

    print()
    print("ZIP-АРХИВ:")
    print(ARCHIVE_PATH.resolve())

    print()
    print("C8 ЗАВЕРШЁН.")


if __name__ == "__main__":
    main()
