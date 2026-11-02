
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ЭКСПЕРИМЕНТ C4

Геометрическая аппроксимация скаляра по локальной
арифметике перехода P_n -> P_(n+1).

Кривая:
    y^2 = x^3 + 7 (mod 499)

Генератор:
    G = (3, 212)

Порядок:
    73

Исследовательская цель:
Проверить, содержат ли локальные арифметические признаки
группового перехода информацию, позволяющую аппроксимировать
скаляр без прямого перебора и таблицы соответствий.

Установка:
    python -m pip install numpy

Запуск:
    python experimentC4.py

Результаты:
    experimentC4_results/
    experimentC4_results.zip
"""

from __future__ import annotations

import csv
import json
import math
import random
import zipfile

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# ============================================================
# ПАРАМЕТРЫ
# ============================================================

P = 499
A = 0
B = 7

G = (3, 212)

SEED = 20261103

RIDGE_ALPHA = 3.0

PERMUTATIONS = 500

AUTOCORR_PERMUTATIONS = 5000

OUT_DIR = Path("experimentC4_results")

Point = Optional[Tuple[int, int]]


# ============================================================
# АРИФМЕТИКА ЭЛЛИПТИЧЕСКОЙ КРИВОЙ
# ============================================================

def inv_mod(value: int, p: int = P) -> int:
    """Обратный элемент по модулю простого числа."""

    value %= p

    if value == 0:
        raise ZeroDivisionError(
            "Ноль не имеет обратного элемента"
        )

    return pow(value, -1, p)


def point_add(p1: Point, p2: Point) -> Point:
    """
    Сложение точек на кривой:

        y^2 = x^3 + A*x + B (mod P)
    """

    if p1 is None:
        return p2

    if p2 is None:
        return p1

    x1, y1 = p1
    x2, y2 = p2

    # Противоположные точки дают точку O.
    if x1 == x2 and (y1 + y2) % P == 0:
        return None

    if p1 == p2:

        # Удвоение точки.
        if y1 % P == 0:
            return None

        numerator = 3 * x1 * x1 + A
        denominator = 2 * y1

    else:

        # Сложение различных точек.
        numerator = y2 - y1
        denominator = x2 - x1

    lam = (
        numerator * inv_mod(denominator)
    ) % P

    x3 = (
        lam * lam - x1 - x2
    ) % P

    y3 = (
        lam * (x1 - x3) - y1
    ) % P

    return x3, y3


def scalar_mul(k: int, point: Point) -> Point:
    """Двоичное умножение точки на скаляр."""

    result = None
    addend = point

    while k:

        if k & 1:
            result = point_add(result, addend)

        addend = point_add(addend, addend)

        k >>= 1

    return result


def point_order(point: Point, limit: int = 10000) -> int:
    """Вычисление порядка точки."""

    current = None

    for k in range(1, limit + 1):

        current = point_add(current, point)

        if current is None:
            return k

    raise RuntimeError(
        "Не удалось найти порядок точки"
    )


def is_on_curve(point: Point) -> bool:
    """Проверка принадлежности кривой."""

    if point is None:
        return True

    x, y = point

    return (
        y * y - (x * x * x + A * x + B)
    ) % P == 0


def build_subgroup(order: int):
    """
    Строит точки P_n=nG.

    Метки n не используются внутри формул
    построения геометрических признаков.
    """

    points = {}

    current = None

    for n in range(1, order):

        current = point_add(current, G)

        if current is None:
            raise AssertionError(
                "Неожиданная точка на бесконечности"
            )

        assert is_on_curve(current)

        points[n] = current

    assert point_add(
        points[order - 1],
        G
    ) is None

    return points


# ============================================================
# ЛОКАЛЬНАЯ АРИФМЕТИЧЕСКАЯ СИГНАТУРА
# ============================================================

def transition_signature(
    n: int,
    point: Tuple[int, int]
) -> dict:
    """
    Строит локальную сигнатуру перехода:

        P_n + G = P_(n+1)

    Для n=1 используется удвоение G.
    Для остальных n используется сложение точек.

    Целочисленные поправки:

        lambda * denominator =
            numerator + q_slope * P

        x_raw = x_next + q_x * P

        y_raw = y_next + q_y * P
    """

    x1, y1 = point

    x2, y2 = G

    if n == 1:

        operation = "doubling"

        numerator = (
            3 * x1 * x1 + A
        )

        denominator = (
            2 * y1
        )

    else:

        operation = "addition"

        numerator = (
            y2 - y1
        )

        denominator = (
            x2 - x1
        )

    # Канонический остаток наклона.
    lam = (
        numerator * inv_mod(denominator)
    ) % P

    # Целочисленная поправка наклона.
    q_slope = (
        lam * denominator - numerator
    ) // P

    # Целочисленное вычисление x.
    x_raw = (
        lam * lam - x1 - x2
    )

    x_next = x_raw % P

    q_x = (
        x_raw - x_next
    ) // P

    # Целочисленное вычисление y.
    y_raw = (
        lam * (x1 - x_next) - y1
    )

    y_next = y_raw % P

    q_y = (
        y_raw - y_next
    ) // P

    # Проверки точных равенств.
    assert (
        lam * denominator
        == numerator + q_slope * P
    )

    assert (
        x_raw == x_next + q_x * P
    )

    assert (
        y_raw == y_next + q_y * P
    )

    next_point = (
        x_next,
        y_next
    )

    assert is_on_curve(next_point)

    assert point_add(
        point,
        G
    ) == next_point

    return {

        "n": n,

        "operation": operation,

        "x": x1,
        "y": y1,

        "numerator": numerator,
        "denominator": denominator,

        "lambda": lam,

        "q_slope": q_slope,

        "x_raw": x_raw,
        "x_next": x_next,
        "q_x": q_x,

        "y_raw": y_raw,
        "y_next": y_next,
        "q_y": q_y,
    }


# ============================================================
# ФОРМИРОВАНИЕ ГЕОМЕТРИЧЕСКИХ ПРИЗНАКОВ
# ============================================================

def make_feature_sets(rows):
    """
    Создаёт три независимых набора признаков.

    ВАЖНО:
    Скаляр n не включается ни в один набор.
    """

    coordinate = []

    transition = []

    combined = []

    for r in rows:

        x = r["x"]
        y = r["y"]

        num = r["numerator"]
        den = r["denominator"]

        lam = r["lambda"]

        qs = r["q_slope"]
        qx = r["q_x"]
        qy = r["q_y"]

        # Только координаты.
        c = [

            x / P,

            y / P,

        ]

        # Локальная арифметика перехода.
        t = [

            lam / P,

            qs / P,

            qx / P,

            qy / P,

            num / (P * P),

            den / P,

            abs(qs) / P,

            abs(qx) / P,

            abs(qy) / P,

            (lam * lam % P) / P,

            ((x - G[0]) % P) / P,

            ((y - G[1]) % P) / P,

        ]

        coordinate.append(c)

        transition.append(t)

        combined.append(c + t)

    return {

        "coordinates_xy": np.asarray(
            coordinate,
            dtype=float
        ),

        "transition_arithmetic": np.asarray(
            transition,
            dtype=float
        ),

        "coordinates_plus_transition": np.asarray(
            combined,
            dtype=float
        ),

    }


# ============================================================
# КРУГОВАЯ РЕГРЕССИЯ
# ============================================================

def circular_targets(
    labels: np.ndarray,
    order: int
) -> np.ndarray:
    """
    Представляет скаляр углом на окружности.
    """

    angle = (
        2.0 * np.pi * labels.astype(float) / order
    )

    return np.column_stack((

        np.cos(angle),

        np.sin(angle),

    ))


def decode_angle(
    vector: np.ndarray,
    order: int
) -> int:
    """Перевод предсказанного вектора обратно в скаляр."""

    angle = math.atan2(
        float(vector[1]),
        float(vector[0])
    )

    if angle < 0:
        angle += 2.0 * math.pi

    return int(
        round(
            angle * order / (2.0 * math.pi)
        )
    ) % order


def circular_distance(
    a: int,
    b: int,
    order: int
) -> int:
    """Кратчайшее расстояние между скалярами на цикле."""

    d = abs((a - b) % order)

    return min(
        d,
        order - d
    )


def loo_ridge(
    X,
    labels,
    order,
    alpha=RIDGE_ALPHA
):
    """
    Leave-one-out Ridge.

    Нормализация проводится отдельно
    внутри каждого обучающего фолда.
    """

    n = len(labels)

    Y = circular_targets(
        labels,
        order
    )

    predictions = []

    errors = []

    for test_i in range(n):

        mask = np.ones(
            n,
            dtype=bool
        )

        mask[test_i] = False

        X_train = X[mask]

        X_test = X[test_i:test_i + 1]

        Y_train = Y[mask]

        mean = X_train.mean(axis=0)

        std = X_train.std(axis=0)

        std[std < 1e-12] = 1.0

        X_train = (
            X_train - mean
        ) / std

        X_test = (
            X_test - mean
        ) / std

        X_train = np.column_stack((

            np.ones(len(X_train)),

            X_train,

        ))

        X_test = np.column_stack((

            np.ones(len(X_test)),

            X_test,

        ))

        penalty = (
            np.eye(X_train.shape[1]) * alpha
        )

        penalty[0, 0] = 0.0

        coef = np.linalg.solve(

            X_train.T @ X_train + penalty,

            X_train.T @ Y_train

        )

        vector = (
            X_test @ coef
        )[0]

        pred = decode_angle(
            vector,
            order
        )

        actual = int(
            labels[test_i]
        )

        predictions.append(pred)

        errors.append(
            circular_distance(
                pred,
                actual,
                order
            )
        )

    errors = np.asarray(
        errors,
        dtype=float
    )

    return {

        "accuracy": float(
            np.mean(errors == 0)
        ),

        "circular_mae": float(
            np.mean(errors)
        ),

        "median_error": float(
            np.median(errors)
        ),

        "predictions": predictions,

        "errors": errors.astype(int).tolist(),

    }


# ============================================================
# КОНТРОЛЬ СО СЛУЧАЙНЫМИ МЕТКАМИ
# ============================================================

def permutation_control(
    X,
    labels,
    order,
    rng,
    repetitions
):
    """Проверяет точность при случайной перестановке меток."""

    scores = []

    for _ in range(repetitions):

        shuffled = rng.permutation(labels)

        result = loo_ridge(
            X,
            shuffled,
            order
        )

        scores.append(
            result["accuracy"]
        )

    scores = np.asarray(
        scores,
        dtype=float
    )

    return {

        "permutations": repetitions,

        "accuracy_mean": float(
            scores.mean()
        ),

        "accuracy_std": float(
            scores.std()
        ),

        "accuracy_max": float(
            scores.max()
        ),

    }


# ============================================================
# АВТОКОРРЕЛЯЦИЯ q_y
# ============================================================

def lag_autocorrelation(
    values,
    lag=1
):
    """Вычисляет автокорреляцию для указанного лага."""

    values = np.asarray(
        values,
        dtype=float
    )

    centered = values - values.mean()

    denominator = float(
        np.dot(centered, centered)
    )

    if denominator == 0:
        return 0.0

    return float(

        np.dot(
            centered[:-lag],
            centered[lag:]
        ) / denominator

    )


def autocorrelation_permutation_test(
    values,
    lag,
    repetitions,
    rng
):
    """
    Проверяет, насколько необычна автокорреляция
    относительно случайных перестановок той же последовательности.
    """

    observed = abs(
        lag_autocorrelation(
            values,
            lag
        )
    )

    values = list(values)

    extreme = 0

    for _ in range(repetitions):

        shuffled = values.copy()

        rng.shuffle(shuffled)

        score = abs(
            lag_autocorrelation(
                shuffled,
                lag
            )
        )

        if score >= observed - 1e-15:
            extreme += 1

    return {

        "lag": lag,

        "observed_autocorrelation":
            lag_autocorrelation(values, lag),

        "two_sided_permutation_p":
            (extreme + 1) / (repetitions + 1),

        "permutations": repetitions,

    }


# ============================================================
# СОХРАНЕНИЕ CSV
# ============================================================

def write_csv(path, rows):

    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(

            f,

            fieldnames=list(
                rows[0].keys()
            )

        )

        writer.writeheader()

        writer.writerows(rows)


# ============================================================
# ОСНОВНОЙ ЭКСПЕРИМЕНТ
# ============================================================

def main():

    print("Запуск эксперимента C4...")

    print(
        "Построение локальных арифметических сигнатур переходов."
    )

    assert is_on_curve(G)

    order = point_order(G)

    assert order == 73, (
        f"Ожидался порядок 73, получен {order}"
    )

    points = build_subgroup(order)

    # Переходы n=1,...,71.
    transition_rows = [

        transition_signature(
            n,
            points[n]
        )

        for n in range(1, order - 1)

    ]

    # Последний переход отдельно:
    # P_72 + G = O.
    assert points[order - 1] == (

        G[0],

        (-G[1]) % P

    )

    assert point_add(
        points[order - 1],
        G
    ) is None

    # Метки используются только как целевая переменная.
    labels = np.asarray(

        [
            r["n"]
            for r in transition_rows
        ],

        dtype=int

    )

    feature_sets = make_feature_sets(
        transition_rows
    )

    rng = np.random.default_rng(
        SEED
    )

    model_results = []

    prediction_rows = []

    # --------------------------------------------------------
    # Три модели
    # --------------------------------------------------------

    for name, X in feature_sets.items():

        result = loo_ridge(
            X,
            labels,
            order
        )

        control = permutation_control(

            X,

            labels,

            order,

            rng,

            PERMUTATIONS

        )

        model_results.append({

            "model": name,

            "feature_count": int(
                X.shape[1]
            ),

            "sample_count": len(labels),

            "accuracy": result["accuracy"],

            "circular_mae": result["circular_mae"],

            "median_error": result["median_error"],

            "permutation_accuracy_mean":
                control["accuracy_mean"],

            "permutation_accuracy_std":
                control["accuracy_std"],

            "permutation_accuracy_max":
                control["accuracy_max"],

        })

        for i, (
            actual,
            predicted,
            error
        ) in enumerate(zip(

            labels,

            result["predictions"],

            result["errors"]

        )):

            prediction_rows.append({

                "model": name,

                "row_index": i,

                "actual_scalar": int(actual),

                "predicted_scalar": int(predicted),

                "circular_error": int(error),

            })

    # --------------------------------------------------------
    # Отдельная проверка q_y
    # --------------------------------------------------------

    qy_values = [

        r["q_y"]

        for r in transition_rows

    ]

    autocorr_rng = random.Random(
        SEED
    )

    qy_lag1_test = autocorrelation_permutation_test(

        qy_values,

        lag=1,

        repetitions=AUTOCORR_PERMUTATIONS,

        rng=autocorr_rng

    )

    # Модель только по q_y.
    qy_only = np.asarray(

        [
            [r["q_y"] / P]
            for r in transition_rows
        ],

        dtype=float

    )

    qy_only_result = loo_ridge(

        qy_only,

        labels,

        order

    )

    # --------------------------------------------------------
    # Метаданные
    # --------------------------------------------------------

    metadata = {

        "experiment": "C4",

        "created_utc":
            datetime.now(timezone.utc).isoformat(),

        "curve":
            f"y^2 = x^3 + {B} (mod {P})",

        "generator": {

            "x": G[0],

            "y": G[1],

        },

        "generator_order": order,

        "regression_samples": len(labels),

        "excluded_terminal_transition":
            "P_72 + G = O",

        "target":
            "n mod 73, encoded as cos/sin phase",

        "features_do_not_include_scalar": True,

        "ridge_alpha": RIDGE_ALPHA,

        "permutation_repetitions": PERMUTATIONS,

        "autocorrelation_permutations":
            AUTOCORR_PERMUTATIONS,

        "random_seed": SEED,

        "interpretation": (
            "Проверяется аппроксимация по локальной "
            "сигнатуре группового перехода. "
            "Это не прямой перебор и не таблица "
            "соответствий точек скалярам."
        ),

    }

    # --------------------------------------------------------
    # Сохранение результатов
    # --------------------------------------------------------

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    write_csv(

        OUT_DIR / "C4_transitions.csv",

        transition_rows

    )

    write_csv(

        OUT_DIR / "C4_model_summary.csv",

        model_results

    )

    write_csv(

        OUT_DIR / "C4_predictions.csv",

        prediction_rows

    )

    result_json = {

        "metadata": metadata,

        "model_summary": model_results,

        "predictions": prediction_rows,

        "qy_lag1_permutation_test": qy_lag1_test,

        "qy_only_model": qy_only_result,

        "transitions": transition_rows,

    }

    with (

        OUT_DIR / "C4_results.json"

    ).open(

        "w",

        encoding="utf-8"

    ) as f:

        json.dump(

            result_json,

            f,

            ensure_ascii=False,

            indent=2

        )

    # --------------------------------------------------------
    # README
    # --------------------------------------------------------

    readme = f"""
ЭКСПЕРИМЕНТ C4
============================================================

Кривая: {metadata['curve']}

Генератор: G={G}

Порядок: {order}

Количество конечных переходов: {len(labels)}

ЦЕЛЬ

Проверить, дают ли локальные целочисленные поправки перехода
P_n+G геометрическое представление, по которому можно
аппроксимировать скаляр n.

В модель не передаётся n как признак.
n используется только как целевая метка.

СРАВНИВАЕМЫЕ МОДЕЛИ

1. coordinates_xy:
   только координаты x,y.

2. transition_arithmetic:
   только локальная арифметика перехода.

3. coordinates_plus_transition:
   координаты и арифметика вместе.

МЕТОД

Leave-one-out Ridge с круговой целью cos/sin.

Перестановочный контроль случайно перемешивает метки скаляров.

Переход P_72+G=O не включён в регрессию, так как для него
нет конечного наклона и координат следующей точки.

ДОПОЛНИТЕЛЬНО

Проверяется автокорреляция q_y на лаге 1 с перестановочным тестом.

Значимая автокорреляция последовательности сама по себе
не означает, что по одному локальному признаку можно
восстановить n.

ФАЙЛЫ

C4_transitions.csv
    Локальные сигнатуры 71 перехода.

C4_model_summary.csv
    Качество моделей и контроль перестановок.

C4_predictions.csv
    LOO-предсказания по каждому объекту.

C4_results.json
    Полные результаты и параметры.

README.txt
    Это описание.

ОГРАНИЧЕНИЯ

Эксперимент проверяет выбранные признаки и модель
на игрушечной кривой.

Успех не доказывает возможность решения ECDLP на secp256k1.

Неуспех не исключает другие геометрические представления.
"""

    (
        OUT_DIR / "README.txt"
    ).write_text(
        readme,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # ZIP
    # --------------------------------------------------------

    zip_path = Path(
        "experimentC4_results.zip"
    )

    with zipfile.ZipFile(

        zip_path,

        "w",

        compression=zipfile.ZIP_DEFLATED

    ) as archive:

        for file in sorted(
            OUT_DIR.iterdir()
        ):

            if file.is_file():

                archive.write(

                    file,

                    arcname=(
                        f"{OUT_DIR.name}/{file.name}"
                    )

                )

    # --------------------------------------------------------
    # ОТЧЁТ
    # --------------------------------------------------------

    print("\nЭКСПЕРИМЕНТ C4 ЗАВЕРШЁН")

    print("=" * 78)

    print(
        f"Кривая: {metadata['curve']}"
    )

    print(
        f"Генератор G={G}, порядок={order}"
    )

    print(
        f"Число конечных переходов: {len(labels)}"
    )

    print("\nАппроксимация скаляра по LOO:")

    print(

        f"{'Модель':30} "
        f"{'Признаки':>9} "
        f"{'Acc':>9} "
        f"{'Circular MAE':>14} "
        f"{'Perm.mean':>12}"

    )

    print("-" * 82)

    for r in model_results:

        print(

            f"{r['model']:30} "

            f"{r['feature_count']:9d} "

            f"{r['accuracy']:9.4f} "

            f"{r['circular_mae']:14.3f} "

            f"{r['permutation_accuracy_mean']:12.4f}"

        )

    print("\nОтдельная проверка q_y:")

    print(

        f"q_y lag-1 autocorrelation = "
        f"{qy_lag1_test['observed_autocorrelation']:.4f}"

    )

    print(

        f"Перестановочное p-value = "
        f"{qy_lag1_test['two_sided_permutation_p']:.4f}"

    )

    print(

        f"Модель только по q_y: "
        f"accuracy={qy_only_result['accuracy']:.4f}, "
        f"MAE={qy_only_result['circular_mae']:.3f}"

    )

    print("\nРезультаты сохранены:")

    print(
        f"Папка: {OUT_DIR.resolve()}"
    )

    print(
        f"Архив: {zip_path.resolve()}"
    )


if __name__ == "__main__":
    main()
