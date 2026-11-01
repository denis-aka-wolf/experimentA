
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ЭКСПЕРИМЕНТ B6
Исследование алгебраической структуры комплексных отпечатков
на игрушечной эллиптической кривой.

Кривая:
    y^2 = x^3 + 7 (mod 499)

Базовая точка:
    G = (3, 212)

Порядок базовой точки:
    73

Отпечаток:
    T_t(P) = A_t(x_P) * exp(-2*pi*i*t*y_P/p)

Эксперимент исследует:

1. Совпадение прямой и оптимизированной формул.
2. Восстановление координаты y из отношения
   отпечатков противоположных точек.
3. Возможность различать координаты x по модулю отпечатка.
4. Зависимость отпечатков P, P+G и P-G.
5. Возможность описать переход между соседними
   отпечатками простой линейной формулой.

Все результаты сохраняются в CSV, JSON, README и ZIP.
Используются только стандартные библиотеки Python.
"""

import csv
import json
import math
import os
import zipfile
from collections import defaultdict
from datetime import datetime, timezone


# ============================================================
# НАСТРОЙКИ ЭКСПЕРИМЕНТА
# ============================================================

P_MOD = 499
A_CURVE = 0
B_CURVE = 7

G = (3, 212)
ORDER = 73

FREQUENCIES = [1, 174]

OUTPUT_DIR = "experimentB6_results"
ARCHIVE_NAME = "experimentB6_results.zip"

TAU = 2.0 * math.pi
ROUND_DIGITS = 10
EPS = 1e-12


# ============================================================
# АРИФМЕТИКА КОНЕЧНОГО ПОЛЯ
# ============================================================

def mod_inv(a, p=P_MOD):
    """Вычисляет обратный элемент по модулю p."""

    a %= p

    if a == 0:
        raise ZeroDivisionError("Обратного элемента для нуля нет.")

    return pow(a, -1, p)


def is_on_curve(point):
    """Проверяет, принадлежит ли точка эллиптической кривой."""

    if point is None:
        return True

    x, y = point

    return (
        y * y - (x * x * x + B_CURVE)
    ) % P_MOD == 0


def point_neg(point):
    """Вычисляет противоположную точку."""

    if point is None:
        return None

    x, y = point

    return (x, (-y) % P_MOD)


def point_add(P, Q):
    """Складывает две точки эллиптической кривой."""

    # Точка на бесконечности является нейтральным элементом.
    if P is None:
        return Q

    if Q is None:
        return P

    x1, y1 = P
    x2, y2 = Q

    # Сумма противоположных точек равна точке на бесконечности.
    if x1 == x2 and (y1 + y2) % P_MOD == 0:
        return None

    # Удвоение точки.
    if P == Q:

        if y1 % P_MOD == 0:
            return None

        numerator = (3 * x1 * x1 + A_CURVE) % P_MOD
        denominator = (2 * y1) % P_MOD

    # Сложение двух разных точек.
    else:

        numerator = (y2 - y1) % P_MOD
        denominator = (x2 - x1) % P_MOD

    slope = (
        numerator * mod_inv(denominator)
    ) % P_MOD

    x3 = (slope * slope - x1 - x2) % P_MOD
    y3 = (slope * (x1 - x3) - y1) % P_MOD

    result = (x3, y3)

    if not is_on_curve(result):
        raise ArithmeticError(
            f"Получена точка, не принадлежащая кривой: {result}"
        )

    return result


def scalar_mul(k, point):
    """Умножает точку на скаляр методом удвоения и сложения."""

    if k < 0:
        return scalar_mul(-k, point_neg(point))

    result = None
    addend = point

    while k:

        if k & 1:
            result = point_add(result, addend)

        addend = point_add(addend, addend)
        k >>= 1

    return result


def enumerate_curve_points():
    """Перебирает все аффинные точки эллиптической кривой."""

    points = []

    for x in range(P_MOD):

        rhs = (x**3 + B_CURVE) % P_MOD

        for y in range(P_MOD):

            if (y * y) % P_MOD == rhs:
                points.append((x, y))

    return points


# ============================================================
# КУБИЧЕСКИЙ МУЛЬТИПЛИКАТИВНЫЙ ХАРАКТЕР
# ============================================================

def build_discrete_log_table():
    """
    Строит таблицу дискретных логарифмов по основанию 7.

    Для ненулевого z:
        z = 7^e mod 499

    Кубический характер:
        chi_3(z) = exp(2*pi*i*e/3)

    Для нуля:
        chi_3(0) = 0
    """

    primitive_root = 7

    logs = {}
    value = 1

    for exponent in range(P_MOD - 1):

        if value in logs:
            raise ArithmeticError(
                "Число 7 не является первообразным корнем по модулю 499."
            )

        logs[value] = exponent
        value = (value * primitive_root) % P_MOD

    if len(logs) != P_MOD - 1:
        raise ArithmeticError(
            "Таблица дискретных логарифмов построена не полностью."
        )

    return logs


LOG_TABLE = build_discrete_log_table()


def chi3(z):
    """Вычисляет значение кубического мультипликативного характера."""

    z %= P_MOD

    if z == 0:
        return 0.0 + 0.0j

    exponent = LOG_TABLE[z] % 3
    angle = TAU * exponent / 3.0

    return complex(
        math.cos(angle),
        math.sin(angle)
    )


# ============================================================
# АДДИТИВНЫЙ ХАРАКТЕР И ОТПЕЧАТКИ
# ============================================================

def psi(t, value):
    """Вычисляет аддитивный характер exp(2*pi*i*t*value/p)."""

    angle = TAU * ((t * value) % P_MOD) / P_MOD

    return complex(
        math.cos(angle),
        math.sin(angle)
    )


def compute_A(t, x, curve_points):
    """
    Вычисляет:

        A_t(x) = сумма chi_3(u-x) * exp(2*pi*i*t*v/p)
    """

    total = 0.0 + 0.0j

    for u, v in curve_points:
        total += chi3(u - x) * psi(t, v)

    return total


def compute_T_optimized(t, point, A_cache):
    """
    Вычисляет отпечаток по оптимизированной формуле:

        T_t(P) = A_t(x_P) * exp(-2*pi*i*t*y_P/p)
    """

    x, y = point

    return A_cache[x] * psi(t, -y)


def compute_T_direct(t, point, curve_points):
    """
    Вычисляет отпечаток непосредственно по исходной формуле:

        T_t(P) = сумма chi_3(u-x_P) *
                 exp(2*pi*i*t*(v-y_P)/p)
    """

    x, y = point
    total = 0.0 + 0.0j

    for u, v in curve_points:
        total += chi3(u - x) * psi(t, v - y)

    return total


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ МАТЕМАТИЧЕСКИЕ ФУНКЦИИ
# ============================================================

def mean(values):

    if not values:
        return 0.0

    return sum(values) / len(values)


def min_or_zero(values):

    if not values:
        return 0.0

    return min(values)


def max_or_zero(values):

    if not values:
        return 0.0

    return max(values)


def complex_mean(values):

    if not values:
        return 0.0 + 0.0j

    return sum(values) / len(values)


def complex_to_text(z):

    return f"{z.real:.15g}{z.imag:+.15g}j"


def complex_affine_fit(xs, ys):
    """
    Подбирает комплексную линейную зависимость:

        y приблизительно равно a*x + b

    Метод наименьших квадратов.
    """

    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError(
            "Недостаточно данных для построения линейной модели."
        )

    mx = complex_mean(xs)
    my = complex_mean(ys)

    numerator = sum(
        (x - mx).conjugate() * (y - my)
        for x, y in zip(xs, ys)
    )

    denominator = sum(
        abs(x - mx) ** 2
        for x in xs
    )

    if denominator < EPS:
        a = 0.0 + 0.0j
    else:
        a = numerator / denominator

    b = my - a * mx

    predictions = [
        a * x + b
        for x in xs
    ]

    errors = [
        abs(y - prediction)
        for y, prediction in zip(ys, predictions)
    ]

    return {
        "a": a,
        "b": b,
        "rmse": math.sqrt(mean([e * e for e in errors])),
        "mean_error": mean(errors),
        "max_error": max_or_zero(errors),
    }


# ============================================================
# ВОССТАНОВЛЕНИЕ Y ИЗ ОТНОШЕНИЯ ПРОТИВОПОЛОЖНЫХ ОТПЕЧАТКОВ
# ============================================================

def recover_y_from_antipodal_ratio(t, ratio):
    """
    Использует соотношение:

        T(P) / T(-P) = exp(-4*pi*i*t*y/p)

    Следовательно:

        ratio = exp(2*pi*i*m/p)

        m = (-2*t*y) mod p

    Извлекаем m из фазы комплексного числа, затем вычисляем:

        y = m * inverse(-2*t) mod p

    Метод предполагает, что:
        - t не равен нулю по модулю p;
        - A_t(x) не равен нулю;
        - численная погрешность фазы достаточно мала.
    """

    if abs(ratio) < EPS:
        return None, None, "нулевое_отношение"

    angle = math.atan2(ratio.imag, ratio.real)

    fraction = (angle / TAU) % 1.0

    m = int(round(fraction * P_MOD)) % P_MOD

    coefficient = (-2 * t) % P_MOD

    if coefficient == 0:
        return None, m, "частота_необратима"

    recovered_y = (
        m * mod_inv(coefficient)
    ) % P_MOD

    return recovered_y, m, "успешно"


def analyze_antipodal_pairs(t, subgroup, fingerprints):
    """Проверяет восстановление y для 36 пар противоположных точек."""

    rows = []

    for k in range(1, 37):

        opposite_k = ORDER - k

        P = subgroup[k]
        minus_P = subgroup[opposite_k]

        T_P = fingerprints[k]
        T_minus_P = fingerprints[opposite_k]

        if abs(T_minus_P) < EPS:

            ratio = None
            recovered_y = None
            recovered_m = None
            status = "нулевой_знаменатель"

        else:

            ratio = T_P / T_minus_P

            recovered_y, recovered_m, status = (
                recover_y_from_antipodal_ratio(
                    t,
                    ratio
                )
            )

        expected_ratio = psi(t, -2 * P[1])

        ratio_error = (
            abs(ratio - expected_ratio)
            if ratio is not None
            else None
        )

        y_correct = (
            recovered_y == P[1]
            if recovered_y is not None
            else False
        )

        rows.append({
            "k": k,
            "противоположный_k": opposite_k,
            "x": P[0],
            "y": P[1],
            "противоположный_x": minus_P[0],
            "противоположный_y": minus_P[1],
            "T_действительная_часть": T_P.real,
            "T_мнимая_часть": T_P.imag,
            "T_противоположной_действительная": T_minus_P.real,
            "T_противоположной_мнимая": T_minus_P.imag,
            "отношение_действительная": (
                ratio.real if ratio is not None else ""
            ),
            "отношение_мнимая": (
                ratio.imag if ratio is not None else ""
            ),
            "ожидаемое_отношение_действительная": expected_ratio.real,
            "ожидаемое_отношение_мнимая": expected_ratio.imag,
            "ошибка_отношения": (
                ratio_error if ratio_error is not None else ""
            ),
            "восстановленное_m": (
                recovered_m if recovered_m is not None else ""
            ),
            "восстановленное_y": (
                recovered_y if recovered_y is not None else ""
            ),
            "y_восстановлена_верно": y_correct,
            "статус": status,
        })

    successful = sum(
        1 for row in rows
        if row["y_восстановлена_верно"]
    )

    errors = [
        row["ошибка_отношения"]
        for row in rows
        if isinstance(row["ошибка_отношения"], float)
    ]

    return rows, {
        "количество_пар": len(rows),
        "успешное_восстановление_y": successful,
        "неудачное_восстановление_y": len(rows) - successful,
        "максимальная_ошибка_отношения": max_or_zero(errors),
        "средняя_ошибка_отношения": mean(errors),
    }


# ============================================================
# ПОИСК X ПО МОДУЛЮ ОТПЕЧАТКА
# ============================================================

def analyze_magnitude_lookup(t, unique_x, A_cache):
    """
    Проверяет, можно ли различать координаты x по величине |A_t(x)|.

    Важно:
    при поиске ближайшего значения исключается сама координата x.
    Иначе результат всегда был бы тривиально правильным.
    """

    magnitudes = {
        x: abs(A_cache[x])
        for x in unique_x
    }

    rows = []

    for x in unique_x:

        target = magnitudes[x]

        # Ищем ближайшую ДРУГУЮ координату x.
        candidates = sorted(
            (
                abs(target - magnitudes[other]),
                other
            )
            for other in unique_x
            if other != x
        )

        nearest_distance, nearest_x = candidates[0]

        second_distance = (
            candidates[1][0]
            if len(candidates) > 1
            else float("inf")
        )

        rows.append({
            "x": x,
            "A_действительная_часть": A_cache[x].real,
            "A_мнимая_часть": A_cache[x].imag,
            "модуль_A": target,
            "ближайшая_другая_x": nearest_x,
            "расстояние_до_ближайшей": nearest_distance,
            "расстояние_до_второй": second_distance,
            "ближайшая_другая_x_совпадает": nearest_x == x,
        })

    values = list(magnitudes.values())

    pairwise_separations = []

    for i in range(len(values)):
        for j in range(i + 1, len(values)):

            pairwise_separations.append(
                abs(values[i] - values[j])
            )

    rounded_groups = defaultdict(list)

    for x, magnitude in magnitudes.items():

        rounded_groups[
            round(magnitude, ROUND_DIGITS)
        ].append(x)

    collision_groups = [
        group
        for group in rounded_groups.values()
        if len(group) > 1
    ]

    nearest_other_distances = [
        row["расстояние_до_ближайшей"]
        for row in rows
    ]

    return rows, {
        "количество_координат_x": len(unique_x),
        "количество_различных_округленных_модулей": len(rounded_groups),
        "группы_коллизий": collision_groups,
        "минимальное_расстояние_между_модулями": min_or_zero(
            pairwise_separations
        ),
        "максимальное_расстояние_между_модулями": max_or_zero(
            pairwise_separations
        ),
        "минимальное_расстояние_до_другой_x": min_or_zero(
            nearest_other_distances
        ),
    }


# ============================================================
# СВЯЗЬ СОСЕДНИХ ТОЧЕК P, P+G И P-G
# ============================================================

def analyze_neighbor_relations(t, subgroup, fingerprints):
    """Исследует изменение отпечатка при прибавлении и вычитании G."""

    rows = []

    differences_plus = []
    differences_minus = []

    ratios_plus = []
    ratios_minus = []

    for k in range(1, ORDER):

        P = subgroup[k]

        P_plus = point_add(P, G)
        P_minus = point_add(P, point_neg(G))

        T_P = fingerprints[k]

        # ----------------------------------------------------
        # Переход P -> P+G
        # ----------------------------------------------------

        if P_plus is not None and k + 1 < ORDER:

            k_plus = k + 1
            T_plus = fingerprints[k_plus]

            ratio_plus = (
                T_plus / T_P
                if abs(T_P) > EPS
                else None
            )

            difference_plus = abs(T_plus - T_P)

            differences_plus.append(difference_plus)

            if ratio_plus is not None:
                ratios_plus.append(ratio_plus)

        else:

            k_plus = 0
            T_plus = None
            ratio_plus = None
            difference_plus = None

        # ----------------------------------------------------
        # Переход P -> P-G
        # ----------------------------------------------------

        if P_minus is not None and k - 1 >= 1:

            k_minus = k - 1
            T_minus = fingerprints[k_minus]

            ratio_minus = (
                T_minus / T_P
                if abs(T_P) > EPS
                else None
            )

            difference_minus = abs(T_minus - T_P)

            differences_minus.append(difference_minus)

            if ratio_minus is not None:
                ratios_minus.append(ratio_minus)

        else:

            k_minus = 0
            T_minus = None
            ratio_minus = None
            difference_minus = None

        rows.append({
            "k": k,
            "x": P[0],
            "y": P[1],
            "k_плюс_G": k_plus,
            "T_действительная_часть": T_P.real,
            "T_мнимая_часть": T_P.imag,
            "T_плюс_G_действительная": (
                T_plus.real if T_plus is not None else ""
            ),
            "T_плюс_G_мнимая": (
                T_plus.imag if T_plus is not None else ""
            ),
            "отношение_плюс_G_действительная": (
                ratio_plus.real if ratio_plus is not None else ""
            ),
            "отношение_плюс_G_мнимая": (
                ratio_plus.imag if ratio_plus is not None else ""
            ),
            "расстояние_плюс_G": (
                difference_plus if difference_plus is not None else ""
            ),
            "k_минус_G": k_minus,
            "T_минус_G_действительная": (
                T_minus.real if T_minus is not None else ""
            ),
            "T_минус_G_мнимая": (
                T_minus.imag if T_minus is not None else ""
            ),
            "отношение_минус_G_действительная": (
                ratio_minus.real if ratio_minus is not None else ""
            ),
            "отношение_минус_G_мнимая": (
                ratio_minus.imag if ratio_minus is not None else ""
            ),
            "расстояние_минус_G": (
                difference_minus if difference_minus is not None else ""
            ),
        })

    return rows, {
        "количество_переходов_плюс_G": len(differences_plus),
        "количество_переходов_минус_G": len(differences_minus),
        "среднее_расстояние_плюс_G": mean(differences_plus),
        "среднее_расстояние_минус_G": mean(differences_minus),
        "минимальное_расстояние_плюс_G": min_or_zero(differences_plus),
        "минимальное_расстояние_минус_G": min_or_zero(differences_minus),
        "среднее_отношение_плюс_G": complex_to_text(
            complex_mean(ratios_plus)
        ),
        "среднее_отношение_минус_G": complex_to_text(
            complex_mean(ratios_minus)
        ),
    }


# ============================================================
# ПРОВЕРКА ПРОСТОЙ РЕКУРРЕНТНОЙ ФОРМУЛЫ
# ============================================================

def analyze_recurrence(fingerprints):
    """
    Проверяет приближенную формулу:

        T((k+1)G) = a*T(kG) + b

    Используются переходы k=1..71.
    Переход от 72G к точке на бесконечности не включается.
    """

    xs = []
    ys = []

    for k in range(1, ORDER - 1):

        xs.append(fingerprints[k])
        ys.append(fingerprints[k + 1])

    fit = complex_affine_fit(xs, ys)

    a = fit["a"]
    b = fit["b"]

    rows = []

    for k in range(1, ORDER - 1):

        current = fingerprints[k]
        actual = fingerprints[k + 1]

        predicted = a * current + b

        rows.append({
            "k": k,
            "следующий_k": k + 1,
            "текущий_T_действительная": current.real,
            "текущий_T_мнимая": current.imag,
            "фактический_следующий_T_действительная": actual.real,
            "фактический_следующий_T_мнимая": actual.imag,
            "предсказанный_T_действительная": predicted.real,
            "предсказанный_T_мнимая": predicted.imag,
            "абсолютная_ошибка": abs(actual - predicted),
        })

    result = {
        "количество_переходов": len(xs),
        "формула": "T_следующий = a * T_текущий + b",
        "a_действительная": a.real,
        "a_мнимая": a.imag,
        "b_действительная": b.real,
        "b_мнимая": b.imag,
        "среднеквадратичная_ошибка": fit["rmse"],
        "средняя_абсолютная_ошибка": fit["mean_error"],
        "максимальная_абсолютная_ошибка": fit["max_error"],
        "средний_модуль_целевого_отпечатка": mean([
            abs(value) for value in ys
        ]),
    }

    return rows, result


# ============================================================
# ТАБЛИЦА ОТПЕЧАТКОВ И МИНИМАЛЬНЫЕ РАССТОЯНИЯ
# ============================================================

def build_fingerprint_rows(t, subgroup, A_cache, fingerprints):

    rows = []

    for k in range(1, ORDER):

        x, y = subgroup[k]

        A = A_cache[x]
        T = fingerprints[k]

        rows.append({
            "частота_t": t,
            "k": k,
            "x": x,
            "y": y,
            "A_действительная": A.real,
            "A_мнимая": A.imag,
            "модуль_A": abs(A),
            "T_действительная": T.real,
            "T_мнимая": T.imag,
            "модуль_T": abs(T),
            "фаза_T": math.atan2(T.imag, T.real),
        })

    return rows


def analyze_fingerprint_separation(fingerprints):

    values = [
        fingerprints[k]
        for k in range(1, ORDER)
    ]

    distances = []

    for i in range(len(values)):

        for j in range(i + 1, len(values)):

            distances.append(
                abs(values[i] - values[j])
            )

    rounded = {
        (
            round(value.real, ROUND_DIGITS),
            round(value.imag, ROUND_DIGITS)
        )
        for value in values
    }

    minimum_distance = min_or_zero(distances)

    return {
        "количество_конечных_отпечатков": len(values),
        "количество_различных_округленных_отпечатков": len(rounded),
        "минимальное_расстояние_между_отпечатками": minimum_distance,
        "максимальное_расстояние_между_отпечатками": max_or_zero(distances),
        "гарантированный_радиус_шума": minimum_distance / 2.0,
    }


# ============================================================
# СОХРАНЕНИЕ CSV
# ============================================================

def write_csv(filename, rows):

    if not rows:
        return

    path = os.path.join(OUTPUT_DIR, filename)

    with open(path, "w", newline="", encoding="utf-8-sig") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
            delimiter=";"
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# СОЗДАНИЕ ТЕКСТОВОГО ОТЧЁТА
# ============================================================

def write_readme(summary):

    path = os.path.join(
        OUTPUT_DIR,
        "README_results.txt"
    )

    with open(path, "w", encoding="utf-8") as f:

        f.write("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА B6\n")
        f.write("=" * 65 + "\n\n")

        f.write("Эллиптическая кривая:\n")
        f.write("y^2 = x^3 + 7 (mod 499)\n\n")

        f.write(f"Базовая точка: {G}\n")
        f.write(f"Порядок базовой точки: {ORDER}\n")
        f.write(
            f"Количество аффинных точек кривой: "
            f"{summary['curve']['количество_аффинных_точек']}\n\n"
        )

        f.write("Исследуемая формула:\n")
        f.write(
            "T_t(P) = A_t(x_P) * exp(-2*pi*i*t*y_P/499)\n\n"
        )

        f.write(
            "Эксперимент исследует структуру отпечатков, "
            "но сам по себе не является алгоритмом решения ECDLP.\n\n"
        )

        for t in FREQUENCIES:

            data = summary["частоты"][str(t)]

            f.write("-" * 65 + "\n")
            f.write(f"ЧАСТОТА t = {t}\n")
            f.write("-" * 65 + "\n\n")

            f.write("Проверка прямой формулы:\n")
            f.write(
                f"Максимальная ошибка: "
                f"{data['проверка_прямой_формулы']['максимальная_ошибка']}\n"
            )
            f.write(
                f"Проверка пройдена: "
                f"{data['проверка_прямой_формулы']['проверка_пройдена']}\n\n"
            )

            f.write("Восстановление координаты y:\n")
            f.write(
                f"Успешно: "
                f"{data['противоположные_точки']['успешное_восстановление_y']}/"
                f"{data['противоположные_точки']['количество_пар']}\n\n"
            )

            f.write("Различимость координат x по модулю:\n")
            f.write(
                f"Количество координат x: "
                f"{data['модуль']['количество_координат_x']}\n"
            )
            f.write(
                f"Минимальное расстояние между модулями: "
                f"{data['модуль']['минимальное_расстояние_между_модулями']}\n\n"
            )

            f.write("Переход между соседними отпечатками:\n")
            f.write(
                f"Среднее расстояние при прибавлении G: "
                f"{data['соседние_точки']['среднее_расстояние_плюс_G']}\n"
            )
            f.write(
                f"Среднее расстояние при вычитании G: "
                f"{data['соседние_точки']['среднее_расстояние_минус_G']}\n\n"
            )

            f.write("Линейная рекуррентная модель:\n")
            f.write(
                f"Среднеквадратичная ошибка: "
                f"{data['рекуррентная_модель']['среднеквадратичная_ошибка']}\n"
            )
            f.write(
                f"Средняя абсолютная ошибка: "
                f"{data['рекуррентная_модель']['средняя_абсолютная_ошибка']}\n\n"
            )

            f.write("Разделение отпечатков:\n")
            f.write(
                f"Минимальное расстояние: "
                f"{data['разделение_отпечатков']['минимальное_расстояние_между_отпечатками']}\n"
            )
            f.write(
                f"Гарантированный радиус шума: "
                f"{data['разделение_отпечатков']['гарантированный_радиус_шума']}\n\n"
            )

        f.write("\nВАЖНЫЕ ОГРАНИЧЕНИЯ\n")
        f.write("=" * 65 + "\n")

        f.write(
            "1. Восстановление y использует одновременно "
            "отпечатки P и -P.\n"
        )

        f.write(
            "2. Поиск x по модулю является проверкой на конечной "
            "таблице координат подгруппы.\n"
        )

        f.write(
            "3. Линейная рекуррентная модель является численной "
            "аппроксимацией, а не доказанным тождеством.\n"
        )

        f.write(
            "4. Вычисления комплексных чисел выполняются "
            "с плавающей точкой.\n"
        )

        f.write(
            "5. Результаты относятся только к исследуемой "
            "игрушечной кривой и выбранным частотам.\n"
        )

    return path


# ============================================================
# ПРОВЕРКА ИСХОДНЫХ ДАННЫХ
# ============================================================

def validate_environment(curve_points):

    print("=" * 65)
    print("ПРОВЕРКА ИСХОДНЫХ ДАННЫХ")
    print("=" * 65)

    assert is_on_curve(G), "Базовая точка не принадлежит кривой."

    assert len(curve_points) == 510, (
        "Неожиданное количество аффинных точек."
    )

    assert scalar_mul(ORDER, G) is None, (
        "Порядок базовой точки не подтверждён."
    )

    assert scalar_mul(ORDER - 1, G) == point_neg(G), (
        "Ошибка проверки порядка базовой точки."
    )

    subgroup = {
        k: scalar_mul(k, G)
        for k in range(1, ORDER)
    }

    assert len(set(subgroup.values())) == ORDER - 1
    assert None not in subgroup.values()

    print("Количество аффинных точек:", len(curve_points))
    print("Базовая точка:", G)
    print("Порядок базовой точки:", ORDER)
    print("Количество конечных точек подгруппы:", len(subgroup))
    print("Все начальные проверки пройдены.")

    return subgroup


# ============================================================
# ОСНОВНОЙ РАСЧЁТ
# ============================================================

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\nЭКСПЕРИМЕНТ B6")
    print("=" * 65)

    print("Перебор точек эллиптической кривой...")

    curve_points = enumerate_curve_points()

    subgroup = validate_environment(curve_points)

    summary = {
        "эксперимент": "B6",
        "время_UTC": datetime.now(timezone.utc).isoformat(),
        "кривая": "y^2 = x^3 + 7 mod 499",
        "curve": {
            "модуль": P_MOD,
            "базовая_точка": list(G),
            "порядок_базовой_точки": ORDER,
            "количество_аффинных_точек": len(curve_points),
        },
        "частоты": {},
    }

    for t in FREQUENCIES:

        print("\n" + "=" * 65)
        print(f"ОБРАБОТКА ЧАСТОТЫ t = {t}")
        print("=" * 65)

        # Получаем все x-координаты подгруппы.
        unique_x = sorted({
            point[0]
            for point in subgroup.values()
        })

        print("Количество различных координат x:", len(unique_x))

        # Вычисляем A_t(x) для каждой координаты.
        print("Вычисление A_t(x)...")

        A_cache = {
            x: compute_A(t, x, curve_points)
            for x in unique_x
        }

        # Вычисляем все отпечатки подгруппы.
        print("Вычисление отпечатков...")

        fingerprints = {
            k: compute_T_optimized(
                t,
                point,
                A_cache
            )
            for k, point in subgroup.items()
        }

        # ----------------------------------------------------
        # Проверка прямой и оптимизированной формул
        # ----------------------------------------------------

        print("Проверка прямой формулы...")

        direct_errors = []

        for k, point in subgroup.items():

            direct = compute_T_direct(
                t,
                point,
                curve_points
            )

            optimized = fingerprints[k]

            direct_errors.append(
                abs(direct - optimized)
            )

        direct_check = {
            "количество_проверок": len(direct_errors),
            "максимальная_ошибка": max_or_zero(direct_errors),
            "средняя_ошибка": mean(direct_errors),
            "проверка_пройдена": max_or_zero(direct_errors) < 1e-8,
        }

        # ----------------------------------------------------
        # Противоположные точки и восстановление y
        # ----------------------------------------------------

        print("Исследование противоположных точек...")

        antipodal_rows, antipodal_summary = (
            analyze_antipodal_pairs(
                t,
                subgroup,
                fingerprints
            )
        )

        # ----------------------------------------------------
        # Поиск x по модулю
        # ----------------------------------------------------

        print("Исследование модулей отпечатков...")

        magnitude_rows, magnitude_summary = (
            analyze_magnitude_lookup(
                t,
                unique_x,
                A_cache
            )
        )

        # ----------------------------------------------------
        # Соседние точки
        # ----------------------------------------------------

        print("Исследование соседних точек...")

        neighbor_rows, neighbor_summary = (
            analyze_neighbor_relations(
                t,
                subgroup,
                fingerprints
            )
        )

        # ----------------------------------------------------
        # Рекуррентная модель
        # ----------------------------------------------------

        print("Проверка линейной рекуррентной модели...")

        recurrence_rows, recurrence_summary = (
            analyze_recurrence(fingerprints)
        )

        # ----------------------------------------------------
        # Отпечатки и минимальные расстояния
        # ----------------------------------------------------

        fingerprint_rows = build_fingerprint_rows(
            t,
            subgroup,
            A_cache,
            fingerprints
        )

        separation_summary = (
            analyze_fingerprint_separation(
                fingerprints
            )
        )

        # ----------------------------------------------------
        # Сохранение CSV
        # ----------------------------------------------------

        write_csv(
            f"fingerprints_t{t}.csv",
            fingerprint_rows
        )

        write_csv(
            f"antipodal_recovery_t{t}.csv",
            antipodal_rows
        )

        write_csv(
            f"magnitude_lookup_t{t}.csv",
            magnitude_rows
        )

        write_csv(
            f"neighbor_relations_t{t}.csv",
            neighbor_rows
        )

        write_csv(
            f"recurrence_fit_t{t}.csv",
            recurrence_rows
        )

        # ----------------------------------------------------
        # Сохранение результатов частоты
        # ----------------------------------------------------

        summary["частоты"][str(t)] = {
            "проверка_прямой_формулы": direct_check,
            "противоположные_точки": antipodal_summary,
            "модуль": magnitude_summary,
            "соседние_точки": neighbor_summary,
            "рекуррентная_модель": recurrence_summary,
            "разделение_отпечатков": separation_summary,
        }

        # ----------------------------------------------------
        # Вывод результатов
        # ----------------------------------------------------

        print("\nРЕЗУЛЬТАТЫ ЧАСТОТЫ", t)

        print("\n1. ПРОВЕРКА ФОРМУЛЫ")

        print(
            "Максимальная ошибка:",
            direct_check["максимальная_ошибка"]
        )

        print(
            "Проверка пройдена:",
            direct_check["проверка_пройдена"]
        )

        print("\n2. ВОССТАНОВЛЕНИЕ Y")

        print(
            "Успешно восстановлено:",
            antipodal_summary["успешное_восстановление_y"],
            "из",
            antipodal_summary["количество_пар"]
        )

        print(
            "Максимальная ошибка отношения:",
            antipodal_summary["максимальная_ошибка_отношения"]
        )

        print("\n3. МОДУЛИ ОТПЕЧАТКОВ")

        print(
            "Количество координат x:",
            magnitude_summary["количество_координат_x"]
        )

        print(
            "Различных округлённых модулей:",
            magnitude_summary["количество_различных_округленных_модулей"]
        )

        print(
            "Минимальное расстояние между модулями:",
            magnitude_summary["минимальное_расстояние_между_модулями"]
        )

        print(
            "Группы коллизий:",
            magnitude_summary["группы_коллизий"]
        )

        print("\n4. СОСЕДНИЕ ТОЧКИ")

        print(
            "Среднее расстояние при прибавлении G:",
            neighbor_summary["среднее_расстояние_плюс_G"]
        )

        print(
            "Среднее расстояние при вычитании G:",
            neighbor_summary["среднее_расстояние_минус_G"]
        )

        print("\n5. РЕКУРРЕНТНАЯ МОДЕЛЬ")

        print("Модель: T_следующий = a*T_текущий + b")

        print(
            "a =",
            complex_to_text(complex(
                recurrence_summary["a_действительная"],
                recurrence_summary["a_мнимая"]
            ))
        )

        print(
            "b =",
            complex_to_text(complex(
                recurrence_summary["b_действительная"],
                recurrence_summary["b_мнимая"]
            ))
        )

        print(
            "Среднеквадратичная ошибка:",
            recurrence_summary["среднеквадратичная_ошибка"]
        )

        print(
            "Средняя абсолютная ошибка:",
            recurrence_summary["средняя_абсолютная_ошибка"]
        )

        print("\n6. РАЗДЕЛЕНИЕ ОТПЕЧАТКОВ")

        print(
            "Количество различных отпечатков:",
            separation_summary["количество_различных_округленных_отпечатков"],
            "из",
            separation_summary["количество_конечных_отпечатков"]
        )

        print(
            "Минимальное расстояние:",
            separation_summary["минимальное_расстояние_между_отпечатками"]
        )

        print(
            "Гарантированный радиус шума:",
            separation_summary["гарантированный_радиус_шума"]
        )

    # ========================================================
    # СОХРАНЕНИЕ JSON
    # ========================================================

    json_path = os.path.join(
        OUTPUT_DIR,
        "summary.json"
    )

    with open(json_path, "w", encoding="utf-8") as f:

        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=4
        )

    # ========================================================
    # СОЗДАНИЕ README
    # ========================================================

    readme_path = write_readme(summary)

    # ========================================================
    # СОЗДАНИЕ ZIP-АРХИВА
    # ========================================================

    if os.path.exists(ARCHIVE_NAME):
        os.remove(ARCHIVE_NAME)

    with zipfile.ZipFile(
        ARCHIVE_NAME,
        "w",
        compression=zipfile.ZIP_DEFLATED
    ) as archive:

        for filename in sorted(os.listdir(OUTPUT_DIR)):

            full_path = os.path.join(
                OUTPUT_DIR,
                filename
            )

            if os.path.isfile(full_path):

                archive.write(
                    full_path,
                    arcname=os.path.join(
                        OUTPUT_DIR,
                        filename
                    )
                )

    # ========================================================
    # ЗАВЕРШЕНИЕ
    # ========================================================

    print("\n" + "=" * 65)
    print("ЭКСПЕРИМЕНТ B6 ЗАВЕРШЁН")
    print("=" * 65)

    print("Папка с результатами:", OUTPUT_DIR)
    print("Файл JSON:", json_path)
    print("Текстовый отчёт:", readme_path)
    print("ZIP-архив:", ARCHIVE_NAME)

    print("\nСОЗДАННЫЕ ФАЙЛЫ:")

    for filename in sorted(os.listdir(OUTPUT_DIR)):
        print(" -", filename)

    print("\nАрхив успешно сформирован:", ARCHIVE_NAME)


if __name__ == "__main__":
    main()
