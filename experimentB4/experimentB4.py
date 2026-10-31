#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EXPERIMENT B4
Исследование геометрии комплексных отпечатков эллиптической кривой.

Кривая:
    y^2 = x^3 + 7 (mod 499)

Базовая точка:
    G = (3, 212)

Порядок:
    ord(G) = 73

Отпечаток:
    A_t(x) = sum chi3(u-x) * exp(2*pi*i*t*v/p)

    T_t(P) = A_t(x_P) * exp(-2*pi*i*t*y_P/p)

Проверяем частоты:
    t = 174
    t = 1

Точка на бесконечности не имеет комплексного отпечатка.
Исследуем конечные точки kG, где k = 1,...,72.
"""

import csv
import json
import math
import os
import zipfile
from datetime import datetime, timezone


# ============================================================
# 1. ПАРАМЕТРЫ
# ============================================================

P_MOD = 499
CURVE_B = 7

G = (3, 212)
EXPECTED_ORDER = 73

PRIMITIVE_ROOT = 7

FREQUENCIES = (174, 1)

OUT_DIR = "experimentB4_results"


# ============================================================
# 2. АРИФМЕТИКА ЭЛЛИПТИЧЕСКОЙ КРИВОЙ
# ============================================================

def inv_mod(a, p=P_MOD):
    return pow(a % p, -1, p)


def on_curve(point):
    if point is None:
        return True

    x, y = point

    return (
        y * y - (x * x * x + CURVE_B)
    ) % P_MOD == 0


def point_add(P, Q):
    """
    Сложение точек эллиптической кривой.

    None означает точку на бесконечности.
    """

    if P is None:
        return Q

    if Q is None:
        return P

    x1, y1 = P
    x2, y2 = Q

    if x1 == x2 and (y1 + y2) % P_MOD == 0:
        return None

    if P == Q:

        if y1 % P_MOD == 0:
            return None

        lam = (
            3 * x1 * x1 * inv_mod(2 * y1)
        ) % P_MOD

    else:

        lam = (
            (y2 - y1) * inv_mod(x2 - x1)
        ) % P_MOD

    x3 = (lam * lam - x1 - x2) % P_MOD

    y3 = (
        lam * (x1 - x3) - y1
    ) % P_MOD

    return x3, y3


def scalar_mult(k, point):

    result = None
    addend = point

    k = int(k)

    while k > 0:

        if k & 1:
            result = point_add(result, addend)

        addend = point_add(addend, addend)

        k >>= 1

    return result


# ============================================================
# 3. ПЕРЕЧИСЛЕНИЕ ТОЧЕК КРИВОЙ
# ============================================================

def enumerate_curve_points():

    points = []

    for x in range(P_MOD):

        rhs = (x * x * x + CURVE_B) % P_MOD

        for y in range(P_MOD):

            if (y * y) % P_MOD == rhs:
                points.append((x, y))

    return points


# ============================================================
# 4. КУБИЧНЫЙ МУЛЬТИПЛИКАТИВНЫЙ ХАРАКТЕР
# ============================================================

def build_discrete_log_table():

    """
    Создаём таблицу дискретных логарифмов
    для примитивного корня 7 по модулю 499.
    """

    table = {}

    value = 1

    for exponent in range(P_MOD - 1):

        if value in table:
            raise RuntimeError(
                "7 не является примитивным корнем"
            )

        table[value] = exponent

        value = (value * PRIMITIVE_ROOT) % P_MOD

    if len(table) != P_MOD - 1 or value != 1:
        raise RuntimeError(
            "Ошибка построения таблицы логарифмов"
        )

    return table


def build_cubic_character(log_table):

    omega = complex(
        math.cos(2 * math.pi / 3),
        math.sin(2 * math.pi / 3)
    )

    def chi3(a):

        a %= P_MOD

        if a == 0:
            return 0j

        exponent = log_table[a] % 3

        return omega ** exponent

    return chi3


# ============================================================
# 5. ВЫЧИСЛЕНИЕ КОМПЛЕКСНЫХ ОТПЕЧАТКОВ
# ============================================================

def build_fingerprints(
    curve_points,
    subgroup_points,
    t,
    chi3
):

    """
    Сначала вычисляем:

        A_t(x) = sum chi3(u-x) * exp(2*pi*i*t*v/p)

    Затем:

        T_t(P) = A_t(x_P) * exp(-2*pi*i*t*y_P/p)

    A_t(x) вычисляется один раз для каждого x.
    """

    exp_table = []

    for v in range(P_MOD):

        angle = (
            2 * math.pi * (t * v % P_MOD) / P_MOD
        )

        exp_table.append(
            complex(math.cos(angle), math.sin(angle))
        )

    x_values = sorted({
        point[0]
        for point in subgroup_points
        if point is not None
    })

    A_by_x = {}

    for x0 in x_values:

        total = 0j

        for u, v in curve_points:

            total += (
                chi3(u - x0) * exp_table[v]
            )

        A_by_x[x0] = total

    rows = []

    for k in range(1, EXPECTED_ORDER):

        point = subgroup_points[k]

        if point is None:
            raise RuntimeError(
                f"Неожиданная точка на бесконечности: k={k}"
            )

        x, y = point

        phase = exp_table[(-y) % P_MOD]

        z = A_by_x[x] * phase

        rows.append({

            "k": k,

            "x": x,
            "y": y,

            "re": z.real,
            "im": z.imag,

            "abs": abs(z),

            "arg": math.atan2(z.imag, z.real),

            "A_re": A_by_x[x].real,
            "A_im": A_by_x[x].imag,

            "A_abs": abs(A_by_x[x]),

            "fingerprint": z
        })

    # Разворачиваем аргумент, чтобы убрать скачки через +/- pi.

    previous = None
    offset = 0.0

    for row in rows:

        arg = row["arg"]

        if previous is not None:

            jump = arg - previous

            if jump > math.pi:
                offset -= 2 * math.pi

            elif jump < -math.pi:
                offset += 2 * math.pi

        row["arg_unwrapped"] = arg + offset

        previous = arg

    return rows, A_by_x


# ============================================================
# 6. КОРРЕЛЯЦИИ
# ============================================================

def pearson(xs, ys):

    n = len(xs)

    mx = sum(xs) / n
    my = sum(ys) / n

    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]

    denom = math.sqrt(
        sum(v * v for v in dx) *
        sum(v * v for v in dy)
    )

    if denom == 0:
        return None

    return sum(
        a * b for a, b in zip(dx, dy)
    ) / denom


def ranks(values):

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

        average_rank = (i + 1 + j) / 2.0

        for pos in range(i, j):
            result[order[pos]] = average_rank

        i = j

    return result


def spearman(xs, ys):

    return pearson(
        ranks(xs),
        ranks(ys)
    )


def analyze_correlations(rows, t):

    k = [
        float(row["k"])
        for row in rows
    ]

    fields = (
        "re",
        "im",
        "abs",
        "arg",
        "arg_unwrapped",
        "A_abs"
    )

    result = []

    for field in fields:

        values = [
            float(row[field])
            for row in rows
        ]

        result.append({

            "t": t,

            "feature": field,

            "pearson_r": pearson(k, values),

            "spearman_rho": spearman(k, values)
        })

    # Отдельно проверяем циклическую информацию аргумента.

    cos_values = [
        math.cos(row["arg"])
        for row in rows
    ]

    sin_values = [
        math.sin(row["arg"])
        for row in rows
    ]

    result.append({

        "t": t,
        "feature": "cos(arg)",

        "pearson_r": pearson(k, cos_values),
        "spearman_rho": spearman(k, cos_values)
    })

    result.append({

        "t": t,
        "feature": "sin(arg)",

        "pearson_r": pearson(k, sin_values),
        "spearman_rho": spearman(k, sin_values)
    })

    return result


# ============================================================
# 7. ГЕОМЕТРИЯ ПЕРЕХОДОВ МЕЖДУ СКАЛЯРАМИ
# ============================================================

def analyze_geometry(rows, t):

    by_k = {
        row["k"]: row["fingerprint"]
        for row in rows
    }

    result = []

    for delta in range(1, 37):

        distances = [

            abs(
                by_k[k + delta] - by_k[k]
            )

            for k in range(
                1,
                EXPECTED_ORDER - delta
            )
        ]

        result.append({

            "t": t,

            "delta_k": delta,

            "pair_count": len(distances),

            "min_distance": min(distances),

            "mean_distance": (
                sum(distances) / len(distances)
            ),

            "max_distance": max(distances)
        })

    return result


# ============================================================
# 8. ПОИСК БЛИЖАЙШЕГО ОТПЕЧАТКА
# ============================================================

def analyze_nearest_lookup(rows, t):

    """
    Для каждого отпечатка ищем ближайший элемент
    в заранее построенной таблице.

    Собственный отпечаток тоже присутствует в таблице.
    Поэтому расстояние до него должно быть нулевым.

    Второе ближайшее расстояние показывает запас
    до ближайшего конкурирующего отпечатка.
    """

    result = []

    for row in rows:

        z = row["fingerprint"]

        distances = sorted(

            (
                abs(z - other["fingerprint"]),
                other["k"]
            )

            for other in rows
        )

        nearest_distance, nearest_k = distances[0]

        second_distance = distances[1][0]

        result.append({

            "t": t,

            "query_k": row["k"],

            "nearest_k": nearest_k,

            "nearest_distance": nearest_distance,

            "second_nearest_distance": second_distance,

            "margin": second_distance - nearest_distance,

            "correct": nearest_k == row["k"]
        })

    return result


# ============================================================
# 9. УСТОЙЧИВОСТЬ К ЧИСЛЕННЫМ ПОГРЕШНОСТЯМ
# ============================================================

def analyze_noise_robustness(rows, t):

    """
    Добавляем к каждому отпечатку комплексный шум
    заданной абсолютной величины.

    Направление шума детерминировано.
    """

    epsilons = (

        0.0,

        1e-12,
        1e-10,
        1e-8,
        1e-6,

        1e-4,
        1e-2,
        1e-1
    )

    refs = [

        (row["k"], row["fingerprint"])

        for row in rows
    ]

    result = []

    for eps in epsilons:

        correct = 0

        min_margin = None

        for k, z in refs:

            angle = (
                k * 2.399963229728653
            ) % (2 * math.pi)

            noise = eps * complex(
                math.cos(angle),
                math.sin(angle)
            )

            query = z + noise

            ordered = sorted(

                (
                    abs(query - other_z),
                    other_k
                )

                for other_k, other_z in refs
            )

            if ordered[0][1] == k:
                correct += 1

            margin = (
                ordered[1][0] - ordered[0][0]
            )

            if min_margin is None:
                min_margin = margin
            else:
                min_margin = min(min_margin, margin)

        result.append({

            "t": t,

            "noise_epsilon": eps,

            "correct_count": correct,

            "total": len(rows),

            "accuracy": correct / len(rows),

            "minimum_nearest_margin": min_margin
        })

    return result


# ============================================================
# 10. СРАВНЕНИЕ С ОБЫЧНЫМ ПЕРЕБОРОМ
# ============================================================

def build_baseline_rows(t, rows):

    return [

        {

            "t": t,

            "method": "brute_force_ECDLP",

            "precomputation_fingerprints": 0,

            "query_fingerprint_calculations": 0,

            "query_point_additions_upper_bound": (
                EXPECTED_ORDER - 1
            ),

            "query_table_distance_checks": 0,

            "note": (
                "Последовательная проверка kG=Q. "
                "Максимум 72 кандидата."
            )
        },

        {

            "t": t,

            "method": "precomputed_fingerprint_lookup",

            "precomputation_fingerprints": len(rows),

            "query_fingerprint_calculations": 1,

            "query_point_additions_upper_bound": 0,

            "query_table_distance_checks": len(rows),

            "note": (
                "Требуется заранее построенная таблица "
                "и вычисление T_t(Q)."
            )
        }
    ]


# ============================================================
# 11. ЗАПИСЬ CSV
# ============================================================

def write_csv(path, rows, fieldnames=None):

    if not rows:
        return

    if fieldnames is None:
        fieldnames = list(rows[0].keys())

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore"
        )

        writer.writeheader()

        writer.writerows(rows)


def fingerprint_csv_rows(rows):

    fields = (

        "k",
        "x",
        "y",

        "re",
        "im",

        "abs",

        "arg",
        "arg_unwrapped",

        "A_re",
        "A_im",
        "A_abs"
    )

    return [

        {
            key: row[key]
            for key in fields
        }

        for row in rows
    ]


# ============================================================
# 12. ГЛАВНАЯ ПРОГРАММА
# ============================================================

def main():

    os.makedirs(
        OUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Проверяем кривую
    # --------------------------------------------------------

    curve_points = enumerate_curve_points()

    if len(curve_points) != 510:

        raise RuntimeError(
            f"Ожидалось 510 точек, получено {len(curve_points)}"
        )

    if not on_curve(G):

        raise RuntimeError(
            "Базовая точка не принадлежит кривой"
        )

    # --------------------------------------------------------
    # Строим подгруппу
    # --------------------------------------------------------

    subgroup = [

        scalar_mult(k, G)

        for k in range(EXPECTED_ORDER)
    ]

    if subgroup[0] is not None:

        raise RuntimeError(
            "0G должен быть точкой на бесконечности"
        )

    if (
        subgroup[-1] is None
        or len(set(subgroup)) != EXPECTED_ORDER
    ):

        raise RuntimeError(
            "Порядок G не равен 73"
        )

    if scalar_mult(EXPECTED_ORDER, G) is not None:

        raise RuntimeError(
            "73G не является точкой на бесконечности"
        )

    if scalar_mult(371, G) != (449, 483):

        raise RuntimeError(
            "Контрольная точка 371G не совпала"
        )

    # --------------------------------------------------------
    # Создаём кубический характер
    # --------------------------------------------------------

    log_table = build_discrete_log_table()

    chi3 = build_cubic_character(log_table)

    # --------------------------------------------------------
    # Контейнеры результатов
    # --------------------------------------------------------

    summaries = {}

    all_geometry = []
    all_correlations = []
    all_nearest = []
    all_noise = []
    all_baseline = []

    # --------------------------------------------------------
    # Основной эксперимент
    # --------------------------------------------------------

    for t in FREQUENCIES:

        print(
            f"\nВычисление отпечатков для t={t}"
        )

        rows, A_by_x = build_fingerprints(

            curve_points,

            subgroup,

            t,

            chi3
        )

        # Отпечатки

        write_csv(

            os.path.join(
                OUT_DIR,
                f"fingerprints_t{t}.csv"
            ),

            fingerprint_csv_rows(rows)
        )

        # Геометрия

        geometry = analyze_geometry(
            rows,
            t
        )

        all_geometry.extend(geometry)

        # Корреляции

        correlations = analyze_correlations(
            rows,
            t
        )

        all_correlations.extend(correlations)

        # Ближайшие отпечатки

        nearest = analyze_nearest_lookup(
            rows,
            t
        )

        all_nearest.extend(nearest)

        # Устойчивость к шуму

        noise = analyze_noise_robustness(
            rows,
            t
        )

        all_noise.extend(noise)

        # Сравнение с перебором

        baseline = build_baseline_rows(
            t,
            rows
        )

        all_baseline.extend(baseline)

        # ----------------------------------------------------
        # Общая статистика
        # ----------------------------------------------------

        z_values = [

            row["fingerprint"]

            for row in rows
        ]

        pairwise = [

            abs(z_values[i] - z_values[j])

            for i in range(len(z_values))

            for j in range(i + 1, len(z_values))
        ]

        summaries[str(t)] = {

            "finite_subgroup_points": len(rows),

            "unique_fingerprints_exact_float": len(
                set(
                    (z.real, z.imag)
                    for z in z_values
                )
            ),

            "minimum_pairwise_distance": min(pairwise),

            "maximum_pairwise_distance": max(pairwise),

            "minimum_abs_A_t_x": min(
                abs(value)
                for value in A_by_x.values()
            ),

            "distinct_x_values": len(A_by_x),

            "nearest_lookup_correct": sum(
                1
                for row in nearest
                if row["correct"]
            ),

            "nearest_lookup_total": len(nearest),

            "minimum_nearest_margin": min(
                row["margin"]
                for row in nearest
            ),

            "noise_test": noise
        }

    # ========================================================
    # 13. СОХРАНЕНИЕ ВСЕХ CSV
    # ========================================================

    write_csv(

        os.path.join(
            OUT_DIR,
            "delta_geometry.csv"
        ),

        all_geometry
    )

    write_csv(

        os.path.join(
            OUT_DIR,
            "correlations.csv"
        ),

        all_correlations
    )

    write_csv(

        os.path.join(
            OUT_DIR,
            "nearest_lookup.csv"
        ),

        all_nearest
    )

    write_csv(

        os.path.join(
            OUT_DIR,
            "noise_robustness.csv"
        ),

        all_noise
    )

    write_csv(

        os.path.join(
            OUT_DIR,
            "baseline_comparison.csv"
        ),

        all_baseline
    )

    # ========================================================
    # 14. JSON-ОТЧЁТ
    # ========================================================

    report = {

        "experiment": "B4",

        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "curve": "y^2 = x^3 + 7 mod 499",

        "p": P_MOD,

        "affine_curve_points": len(curve_points),

        "base_point_G": list(G),

        "order_G": EXPECTED_ORDER,

        "control_371G": list(
            scalar_mult(371, G)
        ),

        "primitive_root": PRIMITIVE_ROOT,

        "frequencies": list(FREQUENCIES),

        "fingerprint_definition": (
            "T_t(P)=A_t(x_P)*exp(-2*pi*i*t*y_P/p); "
            "A_t(x)=sum chi3(u-x)*exp(2*pi*i*t*v/p)"
        ),

        "cubic_character": (
            "chi3(0)=0; "
            "chi3(7^e)=exp(2*pi*i*(e mod 3)/3)"
        ),

        "finite_subgroup_scalars": (
            "k=1,...,72; "
            "k=0 is infinity and has no complex fingerprint"
        ),

        "results": summaries,

        "interpretation_warning": [

            (
                "Уникальность отпечатков на 72 точках "
                "не доказывает инъективность на других кривых."
            ),

            (
                "Поиск ближайшего отпечатка использует "
                "заранее построенную таблицу."
            ),

            (
                "Вычисление отпечатка целевой точки Q "
                "также требует вычислений."
            ),

            (
                "Комплексные значения вычисляются "
                "с плавающей точкой."
            ),

            (
                "Корреляции описывают только эту конечную "
                "последовательность и не дают общей формулы "
                "восстановления скаляра."
            )
        ]
    }

    with open(

        os.path.join(
            OUT_DIR,
            "report.json"
        ),

        "w",

        encoding="utf-8"

    ) as f:

        json.dump(

            report,

            f,

            ensure_ascii=False,

            indent=2
        )

    # ========================================================
    # 15. README
    # ========================================================

    readme = f"""
EXPERIMENT B4

Curve:
    y^2 = x^3 + 7 (mod 499)

Base point:
    G = {G}

Order:
    73

Control:
    371G = {scalar_mult(371, G)}

Frequencies:
    {FREQUENCIES}


FILES:

fingerprints_t174.csv
    Отпечатки при t=174.

fingerprints_t1.csv
    Отпечатки при t=1.

delta_geometry.csv
    Расстояния между T(kG) и T((k+delta)G).

correlations.csv
    Корреляции между k и характеристиками отпечатка.

nearest_lookup.csv
    Поиск ближайшего отпечатка в таблице.

noise_robustness.csv
    Устойчивость поиска к комплексному шуму.

baseline_comparison.csv
    Сравнение с обычным перебором скаляра.

report.json
    Полный машинно-читаемый отчёт.


ВАЖНО:

Исследуются только конечные точки kG,
где k=1,...,72.

Точка k=0 является точкой на бесконечности.

Табличный поиск требует предварительного
построения отпечатков и не является доказанным
ускорением решения ECDLP.
"""

    with open(

        os.path.join(
            OUT_DIR,
            "README_results.txt"
        ),

        "w",

        encoding="utf-8"

    ) as f:

        f.write(readme)

    # ========================================================
    # 16. СОЗДАНИЕ ZIP-АРХИВА
    # ========================================================

    zip_path = os.path.join(
        OUT_DIR,
        "experimentB4.zip"
    )

    with zipfile.ZipFile(

        zip_path,

        "w",

        zipfile.ZIP_DEFLATED

    ) as archive:

        for filename in sorted(
            os.listdir(OUT_DIR)
        ):

            if filename == os.path.basename(zip_path):
                continue

            full_path = os.path.join(
                OUT_DIR,
                filename
            )

            if os.path.isfile(full_path):

                archive.write(
                    full_path,
                    arcname=filename
                )

    # ========================================================
    # 17. ИТОГОВЫЙ ВЫВОД
    # ========================================================

    print("\n")
    print("=" * 55)

    print("EXPERIMENT B4")

    print("=" * 55)

    print(
        f"Affine curve points: {len(curve_points)}"
    )

    print(
        f"Order of G: {EXPECTED_ORDER}"
    )

    print(
        f"371G: {scalar_mult(371, G)}"
    )

    for t in FREQUENCIES:

        s = summaries[str(t)]

        print("\n")
        print(f"RESULT FOR t={t}")

        print(
            "Unique fingerprints:",
            s["unique_fingerprints_exact_float"],
            "/ 72"
        )

        print(
            "Minimum pairwise distance:",
            f"{s['minimum_pairwise_distance']:.15g}"
        )

        print(
            "Minimum nearest-neighbor margin:",
            f"{s['minimum_nearest_margin']:.15g}"
        )

        print(
            "Nearest lookup correct:",
            s["nearest_lookup_correct"],
            "/",
            s["nearest_lookup_total"]
        )

        print(
            "Minimum |A_t(x)|:",
            f"{s['minimum_abs_A_t_x']:.15g}"
        )

    print("\n")
    print("=" * 55)

    print(
        "Results directory:",
        OUT_DIR
    )

    print(
        "Archive created:",
        zip_path
    )

    print("=" * 55)


if __name__ == "__main__":
    main()