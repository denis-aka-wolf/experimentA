
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Эксперимент C2: устойчивость геометрических признаков к выбору
целочисленных представителей координат эллиптической кривой.

Кривая по умолчанию: y^2 = x^3 + 7 (mod 499)
Генератор: G = (3, 212)

Зависимость: numpy
Установка: python -m pip install numpy
Запуск:    python experimentC2.py

Скрипт не перебирает скаляр для заданной точки и не пытается решить ECDLP.
Он строит известную игрушечную подгруппу для контролируемого эксперимента,
а затем проверяет, содержат ли выбранные геометрические признаки информацию
о скаляре при разных представлениях координат.
"""

from __future__ import annotations

import csv
import json
import math
import random
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# ПАРАМЕТРЫ ЭКСПЕРИМЕНТА
# ---------------------------------------------------------------------------

P = 499
A_CURVE = 0
B_CURVE = 7
G = (3, 212)

RIDGE_ALPHA = 1.0
PERMUTATIONS = 100
LIFT_REPEATS = 10
LIFT_RANGE = 3
RANDOM_SEED = 20261102

OUTPUT_DIR = Path("experimentC2_results")

Point = Optional[Tuple[int, int]]  # None обозначает точку на бесконечности O


# ---------------------------------------------------------------------------
# АРИФМЕТИКА ЭЛЛИПТИЧЕСКОЙ КРИВОЙ
# ---------------------------------------------------------------------------

def inverse_mod(value: int, p: int) -> int:
    """Обратный элемент по модулю простого p."""
    value %= p
    if value == 0:
        raise ZeroDivisionError(
            "Нулевой элемент не имеет обратного по модулю p"
        )
    return pow(value, -1, p)


def point_add(P1: Point, P2: Point, p: int = P) -> Point:
    """Сложение точек на y^2 = x^3 + A_CURVE*x + B_CURVE (mod p)."""
    if P1 is None:
        return P2
    if P2 is None:
        return P1

    x1, y1 = P1
    x2, y2 = P2

    # Вертикальная прямая: P + (-P) = O.
    if x1 == x2 and (y1 + y2) % p == 0:
        return None

    if P1 == P2:
        if y1 % p == 0:
            return None
        numerator = (3 * x1 * x1 + A_CURVE) % p
        denominator = (2 * y1) % p
    else:
        numerator = (y2 - y1) % p
        denominator = (x2 - x1) % p

    slope = (
        numerator * inverse_mod(denominator, p)
    ) % p

    x3 = (slope * slope - x1 - x2) % p
    y3 = (slope * (x1 - x3) - y1) % p

    return (x3, y3)


def scalar_mul(k: int, point: Point, p: int = P) -> Point:
    """Двоичное умножение точки на целое число."""
    if k < 0:
        if point is None:
            return None
        point = (point[0], (-point[1]) % p)
        k = -k

    result: Point = None
    addend = point

    while k:
        if k & 1:
            result = point_add(result, addend, p)

        addend = point_add(addend, addend, p)
        k >>= 1

    return result


def is_on_curve(point: Point, p: int = P) -> bool:
    """Проверка принадлежности точки кривой."""
    if point is None:
        return True

    x, y = point

    left = (y * y) % p
    right = (
        x * x * x + A_CURVE * x + B_CURVE
    ) % p

    return left == right


def point_order(
    point: Point,
    p: int = P,
    limit: int = 100000
) -> int:
    """Находит порядок точки последовательным сложением."""
    current: Point = None

    for k in range(1, limit + 1):
        current = point_add(current, point, p)

        if current is None:
            return k

    raise RuntimeError(
        "Порядок точки не найден в заданном пределе"
    )


def build_subgroup(
    generator: Point,
    order: int
) -> List[Tuple[int, int, int]]:
    """Возвращает (скаляр, x, y) для скаляров 1,...,order-1."""
    rows: List[Tuple[int, int, int]] = []
    current: Point = None

    for k in range(1, order):
        current = point_add(current, generator)

        if current is None:
            raise AssertionError(
                "Точка на бесконечности встретилась раньше "
                "заявленного порядка"
            )

        rows.append((k, current[0], current[1]))

    assert point_add(current, generator) is None, (
        "Проверка порядка подгруппы не пройдена"
    )

    return rows


# ---------------------------------------------------------------------------
# ГЕОМЕТРИЧЕСКИЕ ПРИЗНАКИ И ПОДЪЁМЫ КООРДИНАТ
# ---------------------------------------------------------------------------

def centered_residue(value: int, p: int) -> int:
    """Центрированный представитель остатка."""
    value %= p
    return value if value <= p // 2 else value - p


def feature_vector(
    x: int,
    y: int,
    p: int,
    mode: str,
    lift_a: int = 0,
    lift_b: int = 0
) -> np.ndarray:
    """Формирует признаки исключительно из координат точки."""

    if mode == "residue_xy":
        xx, yy = x % p, y % p
        return np.array(
            [xx / p, yy / p],
            dtype=float
        )

    if mode == "centered_xy":
        xx = centered_residue(x, p)
        yy = centered_residue(y, p)

        return np.array(
            [xx / p, yy / p],
            dtype=float
        )

    if mode == "quadratic_centered":
        xx = centered_residue(x, p) / p
        yy = centered_residue(y, p) / p

        return np.array(
            [xx, yy, xx * xx, yy * yy, xx * yy],
            dtype=float
        )

    if mode == "random_lift_xy":
        # Подъёмы задаются независимо для каждой точки.
        # Остатки по модулю p при этом не меняются.
        xx = (x % p + lift_a * p) / p
        yy = (y % p + lift_b * p) / p

        return np.array(
            [xx, yy],
            dtype=float
        )

    if mode == "random_lift_quadratic":
        xx = (x % p + lift_a * p) / p
        yy = (y % p + lift_b * p) / p

        return np.array(
            [xx, yy, xx * xx, yy * yy, xx * yy],
            dtype=float
        )

    raise ValueError(
        f"Неизвестный режим признаков: {mode}"
    )


def make_feature_matrix(
    rows: Sequence[Tuple[int, int, int]],
    mode: str,
    p: int,
    rng: random.Random,
    lift_range: int = LIFT_RANGE
) -> np.ndarray:
    """Строит матрицу признаков без использования скаляра."""
    features = []

    for _scalar, x, y in rows:
        if mode.startswith("random_lift"):
            a = rng.randint(-lift_range, lift_range)
            b = rng.randint(-lift_range, lift_range)
        else:
            a = b = 0

        features.append(
            feature_vector(x, y, p, mode, a, b)
        )

    return np.vstack(features)


# ---------------------------------------------------------------------------
# ОЦЕНКА: LEAVE-ONE-OUT RIDGE ДЛЯ КРУГОВОЙ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ
# ---------------------------------------------------------------------------

def scalar_targets(
    scalars: np.ndarray,
    order: int
) -> np.ndarray:
    """Преобразует скаляры в (cos(theta), sin(theta))."""
    angles = (
        2.0 * np.pi * scalars.astype(float) / order
    )

    return np.column_stack(
        (np.cos(angles), np.sin(angles))
    )


def decode_circular(
    prediction: np.ndarray,
    order: int
) -> int:
    """Переводит предсказанный вектор в ближайший скаляр."""
    angle = math.atan2(
        float(prediction[1]),
        float(prediction[0])
    )

    if angle < 0:
        angle += 2.0 * math.pi

    return int(
        round(angle * order / (2.0 * math.pi))
    ) % order


def circular_distance(
    a: int,
    b: int,
    order: int
) -> int:
    """Кратчайшее расстояние между двумя скалярами на цикле."""
    difference = abs((a - b) % order)

    return min(
        difference,
        order - difference
    )


def loo_ridge(
    X: np.ndarray,
    labels: np.ndarray,
    order: int,
    alpha: float = RIDGE_ALPHA
) -> Dict[str, float]:
    """
    Leave-one-out Ridge.

    Стандартизация выполняется отдельно внутри каждого фолда,
    чтобы тестовая точка не влияла на параметры нормализации.
    """
    n = len(labels)
    targets = scalar_targets(labels, order)

    predictions: List[int] = []
    errors: List[int] = []

    for test_index in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_index] = False

        X_train = X[train_mask]
        X_test = X[test_index:test_index + 1]
        y_train = targets[train_mask]

        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std < 1e-12] = 1.0

        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        # Свободный член не регуляризуется.
        X_train = np.column_stack(
            (np.ones(len(X_train)), X_train)
        )

        X_test = np.column_stack(
            (np.ones(len(X_test)), X_test)
        )

        penalty = (
            np.eye(X_train.shape[1]) * alpha
        )

        penalty[0, 0] = 0.0

        coefficients = np.linalg.solve(
            X_train.T @ X_train + penalty,
            X_train.T @ y_train
        )

        predicted_vector = (
            X_test @ coefficients
        )[0]

        predicted_scalar = decode_circular(
            predicted_vector,
            order
        )

        actual_scalar = int(labels[test_index])

        predictions.append(predicted_scalar)

        errors.append(
            circular_distance(
                predicted_scalar,
                actual_scalar,
                order
            )
        )

    errors_array = np.asarray(
        errors,
        dtype=float
    )

    return {
        "accuracy": float(
            np.mean(errors_array == 0)
        ),
        "circular_mae": float(
            np.mean(errors_array)
        ),
        "median_circular_error": float(
            np.median(errors_array)
        ),
        "max_circular_error": float(
            np.max(errors_array)
        ),
        "n": int(n),
        "predictions": predictions,
    }


def permutation_control(
    X: np.ndarray,
    labels: np.ndarray,
    order: int,
    rng: np.random.Generator,
    repetitions: int
) -> Dict[str, float]:
    """Контроль со случайной перестановкой меток скаляров."""
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

    scores_array = np.asarray(
        scores,
        dtype=float
    )

    return {
        "permutation_count": int(repetitions),
        "accuracy_mean": float(
            scores_array.mean()
        ),
        "accuracy_std": float(
            scores_array.std(ddof=0)
        ),
        "accuracy_max": float(
            scores_array.max()
        ),
    }


# ---------------------------------------------------------------------------
# СЕРИЯ ЭКСПЕРИМЕНТОВ
# ---------------------------------------------------------------------------

def run_experiment():
    py_rng = random.Random(RANDOM_SEED)

    np_rng = np.random.default_rng(
        RANDOM_SEED
    )

    assert is_on_curve(G), (
        "Генератор G не принадлежит заданной кривой"
    )

    order = point_order(G)

    rows = build_subgroup(
        G,
        order
    )

    labels = np.asarray(
        [row[0] for row in rows],
        dtype=int
    )

    # Проверка количества и уникальности точек.
    assert len(rows) == order - 1

    assert len({
        (x, y)
        for _, x, y in rows
    }) == order - 1

    assert all(
        is_on_curve((x, y))
        for _, x, y in rows
    )

    modes = [
        "residue_xy",
        "centered_xy",
        "quadratic_centered",
    ]

    results = []
    detailed_predictions = []

    def evaluate(
        name,
        X,
        current_labels,
        context,
        run_permutation=True
    ):
        model = loo_ridge(
            X,
            current_labels,
            order
        )

        if run_permutation:
            control = permutation_control(
                X,
                current_labels,
                order,
                np_rng,
                PERMUTATIONS
            )
        else:
            control = {}

        row = {
            "experiment": name,
            "context": context,
            "accuracy": model["accuracy"],
            "circular_mae": model["circular_mae"],
            "median_circular_error":
                model["median_circular_error"],
            "max_circular_error":
                model["max_circular_error"],
            "n": model["n"],
            "permutation_accuracy_mean":
                control.get("accuracy_mean"),
            "permutation_accuracy_std":
                control.get("accuracy_std"),
            "permutation_accuracy_max":
                control.get("accuracy_max"),
        }

        results.append(row)

        for index, (actual, predicted) in enumerate(
            zip(
                current_labels,
                model["predictions"]
            )
        ):
            detailed_predictions.append({
                "experiment": name,
                "context": context,
                "point_index": index,
                "actual_scalar": int(actual),
                "predicted_scalar": int(predicted),
                "circular_error": circular_distance(
                    int(actual),
                    int(predicted),
                    order
                ),
            })

    # 1. Основные представления координат.
    for mode in modes:
        X = make_feature_matrix(
            rows,
            mode,
            P,
            py_rng
        )

        evaluate(
            mode,
            X,
            labels,
            "исходная нумерация генератором G"
        )

    # 2. Случайные подъёмы координат.
    for repeat in range(LIFT_REPEATS):
        for mode in (
            "random_lift_xy",
            "random_lift_quadratic"
        ):
            X = make_feature_matrix(
                rows,
                mode,
                P,
                py_rng
            )

            evaluate(
                mode,
                X,
                labels,
                f"случайные подъёмы, повтор {repeat + 1}"
            )

    # 3. Сдвиг начала отсчёта.
    # Это циклическая смена меток, а не изменение точек.
    for shift in (1, 7, 19, 36):
        shifted_labels = (
            labels + shift
        ) % order

        X = make_feature_matrix(
            rows,
            "quadratic_centered",
            P,
            py_rng
        )

        evaluate(
            "origin_shift",
            X,
            shifted_labels,
            f"сдвиг меток на {shift}"
        )

    # 4. Смена генератора G' = rG.
    # Для Q=nG выполняется Q=n'G',
    # где n' = n * r^(-1) (mod order).
    generator_multipliers = [
        r for r in (2, 3, 5, 7, 11)
        if math.gcd(r, order) == 1
    ]

    for r in generator_multipliers:
        new_generator = scalar_mul(r, G)

        assert new_generator is not None

        inverse_r = pow(
            r,
            -1,
            order
        )

        relabeled = (
            labels * inverse_r
        ) % order

        X = make_feature_matrix(
            rows,
            "quadratic_centered",
            P,
            py_rng
        )

        evaluate(
            "generator_change",
            X,
            relabeled,
            (
                f"G' = {r}G; новые метки "
                f"n' = n * {inverse_r} mod {order}"
            )
        )

    metadata = {
        "experiment": "C2",
        "created_utc":
            datetime.now(timezone.utc).isoformat(),
        "curve":
            f"y^2 = x^3 + {B_CURVE} (mod {P})",
        "field_prime": P,
        "generator": {
            "x": G[0],
            "y": G[1]
        },
        "generator_order": order,
        "subgroup_nonzero_points": len(rows),
        "ridge_alpha": RIDGE_ALPHA,
        "permutations": PERMUTATIONS,
        "lift_repeats": LIFT_REPEATS,
        "lift_range": [
            -LIFT_RANGE,
            LIFT_RANGE
        ],
        "random_seed": RANDOM_SEED,
        "chance_accuracy_reference":
            1.0 / order,
        "interpretation_warning": (
            "Результат на игрушечной кривой не является "
            "методом решения ECDLP. LOO-оценка на малой "
            "подгруппе характеризует только выбранные "
            "признаки и модель."
        ),
    }

    return results, {
        "metadata": metadata,
        "predictions": detailed_predictions
    }


# ---------------------------------------------------------------------------
# СОХРАНЕНИЕ CSV, JSON, README И ZIP
# ---------------------------------------------------------------------------

def save_results(results, details):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Сводная таблица.
    results_csv = (
        OUTPUT_DIR / "C2_summary.csv"
    )

    if results:
        with results_csv.open(
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(results[0].keys())
            )

            writer.writeheader()
            writer.writerows(results)

    # Подробности предсказаний.
    predictions_csv = (
        OUTPUT_DIR / "C2_predictions.csv"
    )

    prediction_rows = details["predictions"]

    if prediction_rows:
        with predictions_csv.open(
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(
                    prediction_rows[0].keys()
                )
            )

            writer.writeheader()
            writer.writerows(prediction_rows)

    # Полный JSON.
    full_json = (
        OUTPUT_DIR / "C2_results.json"
    )

    with full_json.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            {
                "metadata": details["metadata"],
                "summary": results,
                "predictions": prediction_rows
            },
            file,
            ensure_ascii=False,
            indent=2
        )

    # README.
    metadata = details["metadata"]

    readme = (
        OUTPUT_DIR / "README.txt"
    )

    with readme.open(
        "w",
        encoding="utf-8"
    ) as file:
        file.write(
            "ЭКСПЕРИМЕНТ C2: УСТОЙЧИВОСТЬ "
            "ГЕОМЕТРИЧЕСКИХ ПРИЗНАКОВ\n"
        )

        file.write("=" * 62 + "\n\n")

        file.write(
            f"Кривая: {metadata['curve']}\n"
        )

        file.write(
            f"Поле: p={metadata['field_prime']}\n"
        )

        file.write(
            f"Генератор: {metadata['generator']}\n"
        )

        file.write(
            f"Порядок генератора: "
            f"{metadata['generator_order']}\n"
        )

        file.write(
            f"Ненулевых точек подгруппы в выборке: "
            f"{metadata['subgroup_nonzero_points']}\n"
        )

        file.write(
            f"Ориентир случайной точности: "
            f"{metadata['chance_accuracy_reference']:.6f}\n\n"
        )

        file.write("Файлы:\n")

        file.write(
            "- C2_summary.csv: сводные метрики по сериям.\n"
        )

        file.write(
            "- C2_predictions.csv: фактические и "
            "предсказанные скаляры по точкам.\n"
        )

        file.write(
            "- C2_results.json: полные результаты "
            "и параметры запуска.\n\n"
        )

        file.write(
            "Метод: leave-one-out Ridge для круговой "
            "цели (cos(theta), sin(theta)).\n"
        )

        file.write(
            "Контроль: случайная перестановка меток скаляров.\n\n"
        )

        file.write("ОГРАНИЧЕНИЯ\n")

        file.write(
            "Этот эксперимент не решает задачу "
            "дискретного логарифмирования.\n"
        )

        file.write(
            "Он исследует только указанные признаки, "
            "модель и игрушечную кривую.\n"
        )

        file.write(
            "Случайные подъёмы координат меняют вещественное "
            "представление, но не остатки координат по модулю p.\n"
        )

        file.write(
            "Поэтому падение качества при подъёмах указывает "
            "на зависимость признака от произвольного выбора "
            "представителя.\n"
        )

    # ZIP-архив.
    zip_path = Path(
        "experimentC2_results.zip"
    )

    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED
    ) as archive:

        for path in sorted(
            OUTPUT_DIR.iterdir()
        ):
            if path.is_file():
                archive.write(
                    path,
                    arcname=(
                        f"{OUTPUT_DIR.name}/{path.name}"
                    )
                )

    return zip_path


# ---------------------------------------------------------------------------
# ВЫВОД ОТЧЁТА
# ---------------------------------------------------------------------------

def print_report(results, metadata):
    print("\nЭКСПЕРИМЕНТ C2 ЗАВЕРШЁН")
    print("=" * 72)

    print(
        f"Кривая: {metadata['curve']}"
    )

    print(
        f"Генератор G = {G}, "
        f"порядок = {metadata['generator_order']}"
    )

    print(
        f"Число точек в выборке: "
        f"{metadata['subgroup_nonzero_points']}"
    )

    print(
        f"Случайный ориентир accuracy: "
        f"{metadata['chance_accuracy_reference']:.4%}\n"
    )

    print(
        f"{'Эксперимент':25} "
        f"{'Контекст':35} "
        f"{'Acc':>8} "
        f"{'MAE':>8} "
        f"{'Perm.mean':>11}"
    )

    print("-" * 92)

    for row in results:
        name = str(row["experiment"])
        context = str(row["context"])

        if len(context) > 34:
            context = context[:31] + "..."

        perm = row["permutation_accuracy_mean"]

        perm_text = (
            f"{float(perm):.4f}"
            if perm is not None
            else "-"
        )

        print(
            f"{name:25} "
            f"{context:35} "
            f"{float(row['accuracy']):8.4f} "
            f"{float(row['circular_mae']):8.3f} "
            f"{perm_text:>11}"
        )

    print("\nИнтерпретация:")

    print(
        "1. Сравните точность с ориентиром 1/order "
        "и контролем перестановок."
    )

    print(
        "2. Сравните обычные признаки с результатами "
        "случайных подъёмов."
    )

    print(
        "3. Если качество исчезает при случайных подъёмах, "
        "признак зависит от произвольного целочисленного "
        "представителя."
    )

    print(
        "4. Устойчивость к сдвигу начала цикла и смене "
        "генератора нужно оценивать отдельно."
    )


# ---------------------------------------------------------------------------
# ТОЧКА ВХОДА
# ---------------------------------------------------------------------------

def main():
    print("Запуск эксперимента C2...")
    print(
        "Проверка кривой и построение игрушечной подгруппы."
    )

    results, details = run_experiment()

    zip_path = save_results(
        results,
        details
    )

    print_report(
        results,
        details["metadata"]
    )

    print(
        f"\nРезультаты сохранены в папке: "
        f"{OUTPUT_DIR.resolve()}"
    )

    print(
        f"ZIP-архив создан: {zip_path.resolve()}"
    )


if __name__ == "__main__":
    main()
