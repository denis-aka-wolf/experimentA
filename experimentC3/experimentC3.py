
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Эксперимент C3: целочисленные поправки в арифметике переходов P_n -> P_(n+1).

Кривая: y^2 = x^3 + 7 (mod 499)
Генератор: G = (3, 212), порядок 73.

Исследуются:
  q_slope: целочисленная поправка в уравнении lambda*den = num + q_slope*p
  q_x:     целочисленная поправка при вычислении x_(n+1)
  q_y:     целочисленная поправка при вычислении y_(n+1)

Скаляр n сохраняется в таблице только как метка для последующего анализа.
Он НЕ используется при построении q_slope, q_x, q_y.

Зависимость: только стандартная библиотека Python.
Запуск: python experimentC3.py
"""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

P = 499
A = 0
B = 7
G = (3, 212)
SEED = 20261102
PERMUTATIONS = 5000

OUT_DIR = Path("experimentC3_results")
Point = Optional[Tuple[int, int]]


def inv_mod(value: int, p: int = P) -> int:
    """Обратный элемент по модулю простого числа."""
    value %= p
    if value == 0:
        raise ZeroDivisionError("Нулевой элемент не обратим")
    return pow(value, -1, p)


def add_points(p1: Point, p2: Point) -> Point:
    """Сложение точек на y^2 = x^3 + A*x + B (mod P)."""
    if p1 is None:
        return p2
    if p2 is None:
        return p1

    x1, y1 = p1
    x2, y2 = p2

    if x1 == x2 and (y1 + y2) % P == 0:
        return None

    if p1 == p2:
        if y1 % P == 0:
            return None
        numerator = 3 * x1 * x1 + A
        denominator = 2 * y1
    else:
        numerator = y2 - y1
        denominator = x2 - x1

    lam = (numerator * inv_mod(denominator)) % P
    x3 = (lam * lam - x1 - x2) % P
    y3 = (lam * (x1 - x3) - y1) % P
    return (x3, y3)


def mul_point(k: int, point: Point) -> Point:
    """Двоичное умножение точки на скаляр."""
    if k < 0:
        if point is None:
            return None
        point = (point[0], (-point[1]) % P)
        k = -k

    result = None
    addend = point

    while k:
        if k & 1:
            result = add_points(result, addend)
        addend = add_points(addend, addend)
        k >>= 1

    return result


def point_order(point: Point, limit: int = 10000) -> int:
    current = None

    for k in range(1, limit + 1):
        current = add_points(current, point)

        if current is None:
            return k

    raise RuntimeError("Порядок точки не найден")


def on_curve(point: Point) -> bool:
    if point is None:
        return True

    x, y = point

    return (
        y * y - (x * x * x + A * x + B)
    ) % P == 0


def build_points(order: int):
    """Строит P_n=nG для n=1,...,order-1."""
    result = {}
    current = None

    for n in range(1, order):
        current = add_points(current, G)

        if current is None:
            raise AssertionError("Неожиданная точка O")

        assert on_curve(current)
        result[n] = current

    assert add_points(result[order - 1], G) is None

    return result


def exact_transition(n: int, point: Tuple[int, int]):
    """
    Вычисляет один переход целочисленно.

    Для n=1 используется формула удвоения.
    Для n>=2 используется сложение P_n + G.
    """
    x1, y1 = point
    x2, y2 = G

    if n == 1:
        operation = "doubling"
        numerator = 3 * x1 * x1 + A
        denominator = 2 * y1
    else:
        operation = "addition"
        numerator = y2 - y1
        denominator = x2 - x1

    # Канонический остаток наклона: от 0 до P-1.
    lam = (
        numerator * inv_mod(denominator)
    ) % P

    # Точное равенство:
    # lambda * denominator = numerator + q_slope * P.
    slope_delta = lam * denominator - numerator

    assert slope_delta % P == 0

    q_slope = slope_delta // P

    # Целочисленный подъём формулы x.
    x_raw = lam * lam - x1 - x2
    x3 = x_raw % P
    q_x = (x_raw - x3) // P

    # Целочисленный подъём формулы y.
    y_raw = lam * (x1 - x3) - y1
    y3 = y_raw % P
    q_y = (y_raw - y3) // P

    next_point = (x3, y3)

    assert on_curve(next_point)
    assert add_points(point, G) == next_point

    # Повторная проверка точных целочисленных равенств.
    assert lam * denominator == numerator + q_slope * P
    assert x_raw == x3 + q_x * P
    assert y_raw == y3 + q_y * P

    return {
        "n": n,
        "operation": operation,
        "x_n": x1,
        "y_n": y1,
        "x_G": x2,
        "y_G": y2,
        "numerator": numerator,
        "denominator": denominator,
        "lambda_mod_p": lam,
        "q_slope": q_slope,
        "x_raw": x_raw,
        "x_next": x3,
        "q_x": q_x,
        "y_raw": y_raw,
        "y_next": y3,
        "q_y": q_y,
    }


def pearson(x, y):
    """Коэффициент корреляции Пирсона."""
    mx = statistics.mean(x)
    my = statistics.mean(y)

    dx = [v - mx for v in x]
    dy = [v - my for v in y]

    sx = math.sqrt(sum(v * v for v in dx))
    sy = math.sqrt(sum(v * v for v in dy))

    if sx == 0 or sy == 0:
        return 0.0

    return sum(
        a * b for a, b in zip(dx, dy)
    ) / (sx * sy)


def ranks(values):
    """Средние ранги при совпадающих значениях."""
    order = sorted(
        range(len(values)),
        key=lambda i: values[i]
    )

    result = [0.0] * len(values)
    i = 0

    while i < len(order):
        j = i + 1

        while (
            j < len(order)
            and values[order[j]] == values[order[i]]
        ):
            j += 1

        avg_rank = (i + 1 + j) / 2.0

        for k in range(i, j):
            result[order[k]] = avg_rank

        i = j

    return result


def spearman(x, y):
    return pearson(ranks(x), ranks(y))


def permutation_test_spearman(
    x,
    y,
    repetitions,
    rng
):
    """Двусторонний перестановочный тест для |rho Spearman|."""
    observed = abs(spearman(x, y))
    extreme = 0
    shuffled = list(y)

    for _ in range(repetitions):
        rng.shuffle(shuffled)

        if abs(spearman(x, shuffled)) >= observed - 1e-15:
            extreme += 1

    return (extreme + 1) / (repetitions + 1)


def autocorrelations(values, max_lag=20):
    """Автокорреляция последовательности для разных лагов."""
    mean = statistics.mean(values)
    centered = [v - mean for v in values]

    denom = sum(v * v for v in centered)

    if denom == 0:
        return []

    result = []

    for lag in range(
        1,
        min(max_lag, len(values) - 1) + 1
    ):
        numerator = sum(
            centered[i] * centered[i + lag]
            for i in range(len(values) - lag)
        )

        result.append({
            "lag": lag,
            "autocorrelation": numerator / denom
        })

    return result


def summarize(rows):
    """Сводные метрики и тесты связи поправок с номером перехода."""
    rng = random.Random(SEED)

    n_values = [r["n"] for r in rows]

    fields = [
        "q_slope",
        "q_x",
        "q_y",
        "lambda_mod_p"
    ]

    summaries = []
    autocorr = {}

    for field in fields:
        values = [r[field] for r in rows]

        rho = spearman(n_values, values)

        p_value = permutation_test_spearman(
            n_values,
            values,
            PERMUTATIONS,
            rng
        )

        counts = Counter(values)

        summaries.append({
            "feature": field,
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "distinct_values": len(counts),
            "most_common_values": json.dumps(
                counts.most_common(10),
                ensure_ascii=False
            ),
            "pearson_with_n": pearson(
                n_values,
                values
            ),
            "spearman_with_n": rho,
            "spearman_permutation_p": p_value,
        })

        autocorr[field] = autocorrelations(values)

    # Сравнение противоположных точек:
    # -P_n = P_(order-n).
    order = point_order(G)

    by_n = {
        r["n"]: r
        for r in rows
    }

    symmetry = []

    for n in range(1, (order // 2) + 1):
        m = order - n

        if n in by_n and m in by_n:
            symmetry.append({
                "n": n,
                "opposite_n": m,
                "q_slope_n": by_n[n]["q_slope"],
                "q_slope_opposite": by_n[m]["q_slope"],
                "q_x_n": by_n[n]["q_x"],
                "q_x_opposite": by_n[m]["q_x"],
                "q_y_n": by_n[n]["q_y"],
                "q_y_opposite": by_n[m]["q_y"],
            })

    return summaries, autocorr, symmetry


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
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    print("Запуск эксперимента C3...")
    print("Проверка кривой и арифметики переходов.")

    assert on_curve(G), "Генератор не лежит на кривой"

    order = point_order(G)

    assert order == 73, (
        f"Ожидался порядок 73, получен {order}"
    )

    points = build_points(order)
    rows = []

    # Конечные переходы n=1,...,71.
    for n in range(1, order - 1):
        row = exact_transition(
            n,
            points[n]
        )

        rows.append(row)

    # Последний переход P_72 + G = O.
    terminal = {
        "n": order - 1,
        "operation": "inverse_to_infinity",
        "x_n": points[order - 1][0],
        "y_n": points[order - 1][1],
        "x_G": G[0],
        "y_G": G[1],
        "numerator": None,
        "denominator": None,
        "lambda_mod_p": None,
        "q_slope": None,
        "x_raw": None,
        "x_next": None,
        "q_x": None,
        "y_raw": None,
        "y_next": None,
        "q_y": None,
    }

    assert points[order - 1] == (
        G[0],
        (-G[1]) % P
    )

    assert add_points(
        points[order - 1],
        G
    ) is None

    all_rows = rows + [terminal]

    summaries, autocorr, symmetry = summarize(rows)

    metadata = {
        "experiment": "C3",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "curve": f"y^2 = x^3 + {B} (mod {P})",
        "field_prime": P,
        "generator": {
            "x": G[0],
            "y": G[1]
        },
        "generator_order": order,
        "finite_transitions": len(rows),
        "terminal_transition": "P_72 + G = O",
        "slope_lift_definition":
            "lambda_mod_p * denominator = numerator + q_slope * p",
        "coordinate_lift_definitions": [
            "x_raw = x_next + q_x * p",
            "y_raw = y_next + q_y * p"
        ],
        "permutation_repetitions": PERMUTATIONS,
        "random_seed": SEED,
        "important_note":
            "n используется только как метка при анализе. "
            "Поправки вычисляются из координат и формул "
            "групповой операции."
    }

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    write_csv(
        OUT_DIR / "C3_transitions.csv",
        all_rows
    )

    write_csv(
        OUT_DIR / "C3_summary.csv",
        summaries
    )

    result_json = {
        "metadata": metadata,
        "summary": summaries,
        "autocorrelations": autocorr,
        "opposite_point_symmetry": symmetry,
        "transitions": all_rows,
    }

    with (
        OUT_DIR / "C3_results.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(
            result_json,
            f,
            ensure_ascii=False,
            indent=2
        )

    readme = f"""ЭКСПЕРИМЕНТ C3: ЦЕЛОЧИСЛЕННЫЕ ПОПРАВКИ
===============================================================

Кривая: {metadata['curve']}
Генератор: G={G}
Порядок генератора: {order}

В таблице C3_transitions.csv находятся переходы:
P_n + G = P_(n+1), n=1,...,71,
а также отдельная строка P_72 + G = O.

Для n=1 используется удвоение точки G.
Для n=2,...,71 используется обычное сложение.
Для n=72 получается точка на бесконечности.

Определения:
  lambda_mod_p * denominator = numerator + q_slope * p
  x_raw = x_next + q_x * p
  y_raw = y_next + q_y * p

Все три q являются точными целыми числами.
Проверки равенств выполняются через assert.

Скаляр n не участвует в вычислении признаков.
Он записывается в CSV как метка для последующего анализа.

Файлы:
  C3_transitions.csv - таблица переходов и поправок.
  C3_summary.csv - диапазоны, частоты и корреляции.
  C3_results.json - полные данные и автокорреляции.
  README.txt - описание эксперимента.

Ограничения:
  Корреляция не доказывает возможность восстановления скаляра.
  Отсутствие корреляции не доказывает отсутствие любой структуры.
  Результат относится к конкретной игрушечной кривой и выбранным
  каноническим представителям lambda в диапазоне 0,...,p-1.
"""

    (
        OUT_DIR / "README.txt"
    ).write_text(
        readme,
        encoding="utf-8"
    )

    zip_path = Path(
        "experimentC3_results.zip"
    )

    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED
    ) as zf:

        for file in sorted(OUT_DIR.iterdir()):
            if file.is_file():
                zf.write(
                    file,
                    arcname=(
                        f"{OUT_DIR.name}/{file.name}"
                    )
                )

    print("\nЭКСПЕРИМЕНТ C3 ЗАВЕРШЁН")
    print("=" * 76)

    print(f"Кривая: {metadata['curve']}")
    print(f"Генератор G = {G}, порядок = {order}")
    print(f"Конечных переходов: {len(rows)}")
    print("Дополнительно: P_72 + G = O")

    print("\nСводка по целочисленным поправкам:")

    print(
        f"{'Признак':18} {'min':>8} {'max':>8} "
        f"{'mean':>10} {'Spearman(n)':>13} {'perm.p':>10}"
    )

    print("-" * 72)

    for item in summaries:
        print(
            f"{item['feature']:18} "
            f"{item['min']:8} "
            f"{item['max']:8} "
            f"{item['mean']:10.3f} "
            f"{item['spearman_with_n']:13.4f} "
            f"{item['spearman_permutation_p']:10.4f}"
        )

    print(
        "\nСильнейшие автокорреляции по модулю:"
    )

    for field, values in autocorr.items():
        best = sorted(
            values,
            key=lambda x: abs(
                x["autocorrelation"]
            ),
            reverse=True
        )[:3]

        formatted = ", ".join(
            f"лаг {v['lag']}: "
            f"{v['autocorrelation']:.3f}"
            for v in best
        )

        print(
            f"{field}: "
            f"{formatted if formatted else 'нет данных'}"
        )

    print("\nПроверка противоположных точек:")
    print(
        f"Сопоставлено пар P_n и -P_n: {len(symmetry)}"
    )

    print(
        f"\nПапка результатов: {OUT_DIR.resolve()}"
    )

    print(
        f"ZIP-архив: {zip_path.resolve()}"
    )


if __name__ == "__main__":
    main()
