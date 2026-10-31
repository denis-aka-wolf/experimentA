
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EXPERIMENT B2

Кривая:
    y^2 = x^3 + 7 (mod 499)

Базовая точка:
    G = (3, 212), order(G) = 73

Исследуем:
    T_t(P) = sum_{R=(u,v) in E(F_p)}
             chi_3(u-x_P) * exp(2*pi*i*t*(v-y_P)/p)

Проверяем:
1. Различимость точек подгруппы.
2. Различимость P и -P.
3. Коллизии для каждого t.
4. Объединение T_t с прежними S3 и S6.
5. Все ненулевые t с учетом комплексного сопряжения.

Создает:
    experimentB2_results/
        fingerprints.csv
        curve_points.csv
        t_scores.csv
        combined_scores.csv
        collisions.json
        report.json
        README_results.txt

    experimentB2.zip
"""

import csv
import json
import math
import zipfile

from pathlib import Path
from collections import defaultdict


# ============================================================
# ПАРАМЕТРЫ
# ============================================================

P = 499
A = 0
B = 7

G = (3, 212)
ORDER_G = 73

ROUND_DIGITS = 10

T_VALUES = range(1, (P - 1) // 2 + 1)

OUT = Path(__file__).resolve().parent / "experimentB2_results"


# ============================================================
# АРИФМЕТИКА ЭЛЛИПТИЧЕСКОЙ КРИВОЙ
# ============================================================

def inv_mod(x):
    x %= P
    if x == 0:
        raise ZeroDivisionError("Обратного элемента не существует")
    return pow(x, -1, P)


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

    lam = numerator * inv_mod(denominator) % P

    x3 = (lam * lam - x1 - x2) % P
    y3 = (lam * (x1 - x3) - y1) % P

    return x3, y3


def scalar_mul(k, point):

    result = None
    addend = point

    while k > 0:

        if k & 1:
            result = point_add(result, addend)

        addend = point_add(addend, addend)
        k >>= 1

    return result


def negate(point):

    if point is None:
        return None

    x, y = point

    return x, (-y) % P


def on_curve(point):

    if point is None:
        return True

    x, y = point

    return (
        y * y -
        (x * x * x + A * x + B)
    ) % P == 0


# ============================================================
# ПОСТРОЕНИЕ ПОДГРУППЫ
# ============================================================

def build_subgroup():

    result = []
    current = None

    for k in range(ORDER_G):

        result.append({
            "k": k,
            "point": current
        })

        current = point_add(current, G)

    if current is not None:
        raise RuntimeError("Ошибка проверки порядка G")

    if len(set(row["point"] for row in result)) != ORDER_G:
        raise RuntimeError("Повторяющиеся точки подгруппы")

    return result


# ============================================================
# ПОЛНАЯ КРИВАЯ
# ============================================================

def build_full_curve():

    points = []

    for x in range(P):

        rhs = (x**3 + A*x + B) % P

        for y in range(P):

            if y*y % P == rhs:
                points.append((x, y))

    return points


# ============================================================
# ПЕРВООБРАЗНЫЙ КОРЕНЬ И ХАРАКТЕРЫ
# ============================================================

def primitive_root():

    phi = P - 1

    factors = []
    n = phi
    d = 2

    while d*d <= n:

        if n % d == 0:

            factors.append(d)

            while n % d == 0:
                n //= d

        d += 1

    if n > 1:
        factors.append(n)

    for g in range(2, P):

        if all(
            pow(g, phi // q, P) != 1
            for q in factors
        ):
            return g

    raise RuntimeError("Не найден первообразный корень")


def discrete_log_table(g):

    table = {}

    value = 1

    for exponent in range(P - 1):

        table[value] = exponent
        value = value * g % P

    return table


def multiplicative_character(order, log_table):

    values = [0j] * P

    for a in range(1, P):

        exponent = log_table[a]

        angle = (
            2.0 * math.pi *
            (exponent % order) /
            order
        )

        values[a] = complex(
            math.cos(angle),
            math.sin(angle)
        )

    return values


# ============================================================
# ОКРУГЛЕНИЕ ОТПЕЧАТКОВ
# ============================================================

def ckey(z):

    return (
        round(z.real, ROUND_DIGITS),
        round(z.imag, ROUND_DIGITS)
    )


# ============================================================
# СТАРЫЕ ХАРАКТЕРИСТИЧЕСКИЕ СУММЫ S3 И S6
# ============================================================

def compute_old(subgroup, chi2, chi3, chi6):

    result = {}

    weighted = [
        chi2[(x**3 + A*x + B) % P]
        for x in range(P)
    ]

    for row in subgroup:

        k = row["k"]
        point = row["point"]

        if point is None:

            result[k] = (0j, 0j)
            continue

        xP, yP = point

        s3 = 0j
        s6 = 0j

        for x in range(P):

            d = (x - xP) % P

            s3 += chi3[d] * weighted[x]
            s6 += chi6[d] * weighted[x]

        result[k] = (s3, s6)

    return result


# ============================================================
# ПРЕДВАРИТЕЛЬНЫЕ СУММЫ ПО ВЕРТИКАЛЯМ
# ============================================================

def precompute_vertical(full_curve, t_values):

    by_x = [[] for _ in range(P)]

    for x, y in full_curve:
        by_x[x].append(y)

    exp_table = []

    for z in range(P):

        angle = 2.0 * math.pi * z / P

        exp_table.append(
            complex(
                math.cos(angle),
                math.sin(angle)
            )
        )

    vertical = {}

    for t in t_values:

        row = [0j] * P

        for u in range(P):

            total = 0j

            for v in by_x[u]:

                total += exp_table[(t*v) % P]

            row[u] = total

        vertical[t] = row

    return vertical, exp_table


# ============================================================
# ВЫЧИСЛЕНИЕ T_t(P)
# ============================================================

def compute_T(
    subgroup,
    chi3,
    t_values,
    vertical,
    exp_table
):

    result = {}

    for row in subgroup:

        k = row["k"]
        point = row["point"]

        result[k] = {}

        if point is None:
            continue

        xP, yP = point

        character_row = [
            chi3[(u-xP) % P]
            for u in range(P)
        ]

        for t in t_values:

            inner = 0j

            for u in range(P):

                c = character_row[u]

                if c != 0:
                    inner += c * vertical[t][u]

            phase = exp_table[(-t*yP) % P]

            result[k][t] = phase * inner

    return result


# ============================================================
# АНАЛИЗ КОЛЛИЗИЙ
# ============================================================

def collision_groups(keys):

    groups = defaultdict(list)

    for k, key in keys.items():
        groups[key].append(k)

    return sorted(
        [
            sorted(group)
            for group in groups.values()
            if len(group) > 1
        ]
    )


def unique_count(keys):

    return len(set(keys.values()))


def symmetry_stats(keys):

    checked = 0
    distinguished = 0
    details = []

    for k in range(1, ORDER_G):

        if k > ORDER_G-k:
            continue

        opposite = ORDER_G-k

        same = keys[k] == keys[opposite]

        checked += 1

        if not same:
            distinguished += 1

        details.append({
            "k": k,
            "negative_k": opposite,
            "same_fingerprint": same
        })

    return {
        "pairs_checked": checked,
        "pairs_distinguished": distinguished,
        "pairs_same": checked-distinguished,
        "details": details
    }


# ============================================================
# ОСНОВНОЙ ЭКСПЕРИМЕНТ
# ============================================================

def main():

    OUT.mkdir(exist_ok=True)

    print("EXPERIMENT B2")
    print("="*55)

    subgroup = build_subgroup()
    full_curve = build_full_curve()

    print("Порядок G:", ORDER_G)
    print("Проверка 371G:", scalar_mul(371, G))
    print("Аффинных точек полной кривой:", len(full_curve))
    print("Точек подгруппы:", len(subgroup))

    if scalar_mul(371, G) != (449, 483):
        raise RuntimeError("Ошибка контрольной точки 371G")

    g = primitive_root()
    logs = discrete_log_table(g)

    chi2 = multiplicative_character(2, logs)
    chi3 = multiplicative_character(3, logs)
    chi6 = multiplicative_character(6, logs)

    print("Первообразный корень:", g)

    old = compute_old(
        subgroup,
        chi2,
        chi3,
        chi6
    )

    print("Вычисление вертикальных сумм...")

    vertical, exp_table = precompute_vertical(
        full_curve,
        T_VALUES
    )

    print("Вычисление T_t...")

    all_T = compute_T(
        subgroup,
        chi3,
        T_VALUES,
        vertical,
        exp_table
    )

    # --------------------------------------------------------
    # ОЦЕНКА КАЖДОГО t
    # --------------------------------------------------------

    scores = []
    keys_by_t = {}

    for t in T_VALUES:

        keys = {}

        for row in subgroup:

            k = row["k"]

            if row["point"] is None:
                keys[k] = ("O",)
            else:
                keys[k] = (ckey(all_T[k][t]),)

        keys_by_t[t] = keys

        collisions = collision_groups(keys)
        symmetry = symmetry_stats(keys)

        scores.append({
            "t": t,
            "unique": unique_count(keys),
            "collision_groups": len(collisions),
            "points_in_collisions": sum(map(len, collisions)),
            "pm_distinguished": symmetry["pairs_distinguished"],
            "pm_same": symmetry["pairs_same"]
        })

    scores.sort(
        key=lambda r: (
            r["unique"],
            r["pm_distinguished"],
            -r["collision_groups"]
        ),
        reverse=True
    )

    best_t = scores[0]["t"]

    # --------------------------------------------------------
    # ОБЪЕДИНЕНИЕ S3 + S6 + T_t
    # --------------------------------------------------------

    combined_scores = []
    combined_keys = {}

    for t in T_VALUES:

        keys = {}

        for row in subgroup:

            k = row["k"]

            if row["point"] is None:
                keys[k] = ("O",)
            else:

                s3, s6 = old[k]

                keys[k] = (
                    ckey(s3),
                    ckey(s6),
                    ckey(all_T[k][t])
                )

        combined_keys[t] = keys

        collisions = collision_groups(keys)
        symmetry = symmetry_stats(keys)

        combined_scores.append({
            "t": t,
            "unique": unique_count(keys),
            "collision_groups": len(collisions),
            "points_in_collisions": sum(map(len, collisions)),
            "pm_distinguished": symmetry["pairs_distinguished"],
            "pm_same": symmetry["pairs_same"]
        })

    combined_scores.sort(
        key=lambda r: (
            r["unique"],
            r["pm_distinguished"],
            -r["collision_groups"]
        ),
        reverse=True
    )

    best_combined_t = combined_scores[0]["t"]

    # --------------------------------------------------------
    # ОБЪЕДИНЕНИЕ СО ВСЕМИ T
    # --------------------------------------------------------

    all_keys = {}

    for row in subgroup:

        k = row["k"]

        if row["point"] is None:
            all_keys[k] = ("O",)
        else:

            s3, s6 = old[k]

            all_keys[k] = (
                ckey(s3),
                ckey(s6),
                tuple(
                    ckey(all_T[k][t])
                    for t in T_VALUES
                )
            )

    all_collisions = collision_groups(all_keys)

    # --------------------------------------------------------
    # CSV: ВСЕ ОТПЕЧАТКИ
    # --------------------------------------------------------

    with open(
        OUT/"fingerprints.csv",
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        header = [
            "k", "x", "y",
            "S3_real", "S3_imag",
            "S6_real", "S6_imag"
        ]

        for t in T_VALUES:
            header += [
                f"T{t}_real",
                f"T{t}_imag"
            ]

        writer.writerow(header)

        for row in subgroup:

            k = row["k"]
            point = row["point"]

            s3, s6 = old[k]

            if point is None:
                x = y = ""
            else:
                x, y = point

            values = [
                k, x, y,
                s3.real, s3.imag,
                s6.real, s6.imag
            ]

            for t in T_VALUES:

                if point is None:
                    values += ["", ""]
                else:
                    z = all_T[k][t]
                    values += [z.real, z.imag]

            writer.writerow(values)

    # --------------------------------------------------------
    # CSV: ВСЕ ТОЧКИ КРИВОЙ
    # --------------------------------------------------------

    with open(
        OUT/"curve_points.csv",
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(["x", "y"])
        writer.writerows(full_curve)

    # --------------------------------------------------------
    # CSV: РЕЙТИНГ t
    # --------------------------------------------------------

    with open(
        OUT/"t_scores.csv",
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(scores[0].keys())
        )

        writer.writeheader()
        writer.writerows(scores)

    with open(
        OUT/"combined_scores.csv",
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(combined_scores[0].keys())
        )

        writer.writeheader()
        writer.writerows(combined_scores)

    # --------------------------------------------------------
    # JSON: КОЛЛИЗИИ
    # --------------------------------------------------------

    def describe_collisions(groups):

        result = []

        for group in groups:

            result.append({
                "k_values": group,
                "points": [
                    subgroup[k]["point"]
                    for k in group
                ]
            })

        return result

    best_single_collisions = collision_groups(
        keys_by_t[best_t]
    )

    best_combined_collisions = collision_groups(
        combined_keys[best_combined_t]
    )

    collision_data = {
        "best_single_t": best_t,
        "best_single_collisions":
            describe_collisions(best_single_collisions),

        "best_combined_t": best_combined_t,
        "best_combined_collisions":
            describe_collisions(best_combined_collisions),

        "all_t_collisions":
            describe_collisions(all_collisions)
    }

    with open(
        OUT/"collisions.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            collision_data,
            f,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # JSON: ПОЛНЫЙ ОТЧЕТ
    # --------------------------------------------------------

    report = {
        "experiment": "B2",

        "curve": {
            "equation": "y^2 = x^3 + 7 mod 499",
            "p": P,
            "G": G,
            "order_G": ORDER_G,
            "primitive_root": g,
            "affine_curve_points": len(full_curve)
        },

        "tested_t_values": list(T_VALUES),

        "best_single_T": {
            "t": best_t,
            "metrics": scores[0],
            "symmetry": symmetry_stats(
                keys_by_t[best_t]
            )
        },

        "best_S3_S6_T": {
            "t": best_combined_t,
            "metrics": combined_scores[0],
            "symmetry": symmetry_stats(
                combined_keys[best_combined_t]
            )
        },

        "all_T_combined": {
            "unique_fingerprints": unique_count(all_keys),
            "collision_groups": len(all_collisions),
            "collisions": describe_collisions(all_collisions),
            "symmetry": symmetry_stats(all_keys)
        },

        "warning": (
            "Уникальность отпечатков на игрушечной подгруппе "
            "не доказывает ускорение ECDLP. Вычисление отпечатка "
            "требует знания точки P и обхода точек кривой."
        )
    }

    with open(
        OUT/"report.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # ТЕКСТОВЫЙ ОТЧЕТ
    # --------------------------------------------------------

    summary = f"""
EXPERIMENT B2

Кривая: y^2 = x^3 + 7 mod {P}
Базовая точка: {G}
Порядок G: {ORDER_G}

Аффинных точек полной кривой: {len(full_curve)}
Точек подгруппы: {len(subgroup)}

Проверено значений t: {len(list(T_VALUES))}

ЛУЧШИЙ ОДИНОЧНЫЙ T_t
t = {best_t}
Уникальных отпечатков: {unique_count(keys_by_t[best_t])}
Коллизионных групп: {len(best_single_collisions)}
Различимых пар P и -P:
{symmetry_stats(keys_by_t[best_t])['pairs_distinguished']} из 36

ЛУЧШЕЕ ОБЪЕДИНЕНИЕ S3 + S6 + T_t
t = {best_combined_t}
Уникальных отпечатков: {unique_count(combined_keys[best_combined_t])}
Коллизионных групп: {len(best_combined_collisions)}
Различимых пар P и -P:
{symmetry_stats(combined_keys[best_combined_t])['pairs_distinguished']} из 36

S3 + S6 + ВСЕ T_t
Уникальных отпечатков: {unique_count(all_keys)}
Коллизионных групп: {len(all_collisions)}
Различимых пар P и -P:
{symmetry_stats(all_keys)['pairs_distinguished']} из 36

ВАЖНО:
Уникальность отпечатков не является сама по себе
алгоритмом решения ECDLP или доказательством ускорения.
"""

    (OUT/"README_results.txt").write_text(
        summary,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # ZIP-АРХИВ
    # --------------------------------------------------------

    zip_path = Path(__file__).resolve().parent / "experimentB2.zip"

    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED
    ) as archive:

        archive.write(
            Path(__file__).resolve(),
            "experimentB2.py"
        )

        for file in OUT.iterdir():

            if file.is_file():

                archive.write(
                    file,
                    f"experimentB2_results/{file.name}"
                )

    # --------------------------------------------------------
    # ИТОГ В КОНСОЛИ
    # --------------------------------------------------------

    print()
    print("="*55)
    print("РЕЗУЛЬТАТЫ B2")
    print("="*55)

    print()
    print("Лучший одиночный T_t:")
    print("t =", best_t)
    print("Уникальных отпечатков:", unique_count(keys_by_t[best_t]))
    print("Коллизий:", len(best_single_collisions))
    print(
        "Различимых пар +/-:",
        symmetry_stats(keys_by_t[best_t])["pairs_distinguished"],
        "/ 36"
    )

    print()
    print("Лучшее S3 + S6 + T_t:")
    print("t =", best_combined_t)
    print("Уникальных отпечатков:", unique_count(combined_keys[best_combined_t]))
    print("Коллизий:", len(best_combined_collisions))
    print(
        "Различимых пар +/-:",
        symmetry_stats(combined_keys[best_combined_t])["pairs_distinguished"],
        "/ 36"
    )

    print()
    print("S3 + S6 + все T_t:")
    print("Уникальных отпечатков:", unique_count(all_keys))
    print("Коллизий:", len(all_collisions))
    print(
        "Различимых пар +/-:",
        symmetry_stats(all_keys)["pairs_distinguished"],
        "/ 36"
    )

    print()
    print("Результаты:", OUT)
    print("Архив:", zip_path)


if __name__ == "__main__":
    main()
