
# -*- coding: utf-8 -*-
"""
Эксперимент B7.
Исследование кубической симметрии комплексных отпечатков
на эллиптической кривой:

    y^2 = x^3 + 7 (mod 499)

Проверяются:
1. Фазовые отношения A_t(139*x) / A_t(x) и A_t(359*x) / A_t(x).
2. Различимость координат x по фазе A_t(x).
3. Восстановление кандидатов (x, y) по одному T_t(P).
4. Различие между поиском по таблице эталонов и восстановлением
   кандидатов без таблицы скаляров k.

После выполнения создаются:
- CSV-файлы с результатами;
- JSON-файл со сводкой;
- README;
- ZIP-архив со всеми результатами и копией скрипта.
"""

import cmath
import csv
import json
import math
import zipfile

from pathlib import Path
from collections import Counter


# ============================================================
# 1. ПАРАМЕТРЫ ЭКСПЕРИМЕНТА
# ============================================================

P = 499
A_CURVE = 0
B_CURVE = 7

G = (3, 212)
ORDER = 73

OMEGA = 139
OMEGA2 = 359

FREQUENCIES = [1, 174]

EPSILON = 1e-12

OUTPUT_DIR = Path("experimentB7_results")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 72)
print("ЭКСПЕРИМЕНТ B7: ФАЗОВАЯ СИММЕТРИЯ И ВОССТАНОВЛЕНИЕ ТОЧКИ")
print("=" * 72)


# ============================================================
# 2. ПРОВЕРКА ПОЛЯ И КУБИЧЕСКИХ КОРНЕЙ ЕДИНИЦЫ
# ============================================================

def is_prime(n):
    """Проверка простоты числа."""
    if n < 2:
        return False

    for d in range(2, math.isqrt(n) + 1):
        if n % d == 0:
            return False

    return True


assert is_prime(P), "Модуль P должен быть простым."

assert pow(OMEGA, 3, P) == 1
assert pow(OMEGA2, 3, P) == 1

assert OMEGA != 1
assert OMEGA2 != 1
assert OMEGA != OMEGA2

assert (OMEGA * OMEGA2) % P == 1

print(f"\nМодуль поля: {P}")
print(f"Кубические корни единицы: {OMEGA}, {OMEGA2}")
print("Проверка omega^3 = 1: пройдена.")


# ============================================================
# 3. КУБИЧЕСКИЙ МУЛЬТИПЛИКАТИВНЫЙ ХАРАКТЕР
# ============================================================

# Используем 7 как первообразный корень поля.
# Строим таблицу дискретных логарифмов:
#
#     z = 7^e mod P
#
# Кубический характер:
#
#     chi(z) = exp(2*pi*i*(e mod 3)/3)
#
#     chi(0) = 0

LOG_TABLE = {}

value = 1

for exponent in range(P - 1):
    LOG_TABLE[value] = exponent
    value = (value * 7) % P

assert len(LOG_TABLE) == P - 1, (
    "Число 7 не оказалось первообразным корнем поля."
)


def chi(z):
    """Вычисление кубического мультипликативного характера."""
    z %= P

    if z == 0:
        return 0j

    exponent = LOG_TABLE[z] % 3

    return cmath.exp(
        2j * math.pi * exponent / 3
    )


CHI_OMEGA = chi(OMEGA)
CHI_OMEGA2 = chi(OMEGA2)

EXPECTED_PHASE_1 = cmath.exp(2j * math.pi / 3)
EXPECTED_PHASE_2 = cmath.exp(4j * math.pi / 3)

print("\nКубический характер:")
print(f"chi(139) = {CHI_OMEGA}")
print(f"chi(359) = {CHI_OMEGA2}")

print(
    "Отклонение chi(139) от exp(2*pi*i/3):",
    abs(CHI_OMEGA - EXPECTED_PHASE_1)
)

print(
    "Отклонение chi(359) от exp(4*pi*i/3):",
    abs(CHI_OMEGA2 - EXPECTED_PHASE_2)
)


# ============================================================
# 4. ТОЧКИ ЭЛЛИПТИЧЕСКОЙ КРИВОЙ
# ============================================================

def is_on_curve(point):
    """Проверка принадлежности точки кривой."""
    if point is None:
        return True

    x, y = point

    return (
        y * y - (x * x * x + B_CURVE)
    ) % P == 0


def point_add(P1, P2):
    """Сложение точек эллиптической кривой."""

    if P1 is None:
        return P2

    if P2 is None:
        return P1

    x1, y1 = P1
    x2, y2 = P2

    # Противоположные точки.
    if x1 == x2 and (y1 + y2) % P == 0:
        return None

    if P1 == P2:
        if y1 == 0:
            return None

        numerator = 3 * x1 * x1
        denominator = 2 * y1

    else:
        numerator = y2 - y1
        denominator = x2 - x1

    slope = (
        numerator
        * pow(denominator % P, -1, P)
    ) % P

    x3 = (slope * slope - x1 - x2) % P

    y3 = (
        slope * (x1 - x3) - y1
    ) % P

    return x3, y3


def scalar_mult(k, point):
    """Умножение точки на целый скаляр."""
    result = None
    current = point

    while k > 0:
        if k & 1:
            result = point_add(result, current)

        current = point_add(current, current)
        k >>= 1

    return result


assert is_on_curve(G), "Базовая точка не принадлежит кривой."

# Перечисляем все аффинные точки кривой.
ALL_POINTS = []

for x in range(P):
    rhs = (x ** 3 + B_CURVE) % P

    for y in range(P):
        if (y * y) % P == rhs:
            ALL_POINTS.append((x, y))

print("\nКривая:")
print(f"Количество аффинных точек: {len(ALL_POINTS)}")
print("Точка на бесконечности учитывается отдельно.")

assert len(ALL_POINTS) == 510


# ============================================================
# 5. ПОДГРУППА БАЗОВОЙ ТОЧКИ
# ============================================================

BY_K = {}

current = None

for k in range(ORDER):
    BY_K[k] = current
    current = point_add(current, G)

assert current is None, (
    "Порядок базовой точки не совпадает с ORDER."
)

assert BY_K[1] == G

SUBGROUP_POINTS = {
    k: BY_K[k]
    for k in range(1, ORDER)
}

SUBGROUP_X = sorted({
    point[0]
    for point in SUBGROUP_POINTS.values()
})

assert len(SUBGROUP_POINTS) == 72
assert len(SUBGROUP_X) == 36

print("\nПодгруппа:")
print(f"Порядок базовой точки: {ORDER}")
print(f"Ненулевых точек: {len(SUBGROUP_POINTS)}")
print(f"Различных координат x: {len(SUBGROUP_X)}")


# ============================================================
# 6. ПРОВЕРКА ОРБИТ ТРОЙНОЙ СИММЕТРИИ
# ============================================================

def orbit_of_x(x):
    """Орбита координаты x при умножении на omega."""
    return {
        x % P,
        (OMEGA * x) % P,
        (OMEGA2 * x) % P
    }


ORBITS = []
VISITED_X = set()

for x in SUBGROUP_X:
    if x in VISITED_X:
        continue

    orbit = orbit_of_x(x)

    assert len(orbit) == 3

    assert orbit.issubset(set(SUBGROUP_X)), (
        f"Орбита {orbit} выходит за пределы координат подгруппы."
    )

    ORBITS.append(sorted(orbit))
    VISITED_X.update(orbit)

assert len(ORBITS) == 12
assert len(VISITED_X) == 36

print("\nТройная симметрия:")
print(f"Количество орбит: {len(ORBITS)}")
print("Размер каждой орбиты: 3")
print("Все 36 координат распределены по орбитам.")


# ============================================================
# 7. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def psi(t, v):
    """Аддитивный характер exp(2*pi*i*t*v/P)."""
    return cmath.exp(
        2j * math.pi * ((t * v) % P) / P
    )


def phase_distance(a, b):
    """Кратчайшее расстояние между двумя фазами."""
    difference = abs(a - b) % (2 * math.pi)

    return min(
        difference,
        2 * math.pi - difference
    )


def phase_of(z):
    """Фаза комплексного числа в диапазоне [0, 2*pi)."""
    return math.atan2(z.imag, z.real) % (2 * math.pi)


def recover_y_from_phase(T_value, A_value, t):
    """
    Восстановление y из отношения:

        T / A = exp(-2*pi*i*t*y/P)

    Возвращает y по модулю P.
    """

    if abs(A_value) < EPSILON:
        return None

    ratio = T_value / A_value

    if abs(ratio) < EPSILON:
        return None

    # Нормализуем отношение, чтобы убрать
    # небольшую ошибку численного модуля.
    ratio /= abs(ratio)

    angle = phase_of(ratio)

    # Получаем q из:
    #
    # exp(2*pi*i*q/P) = T/A
    #
    # q = -t*y mod P

    q_float = angle * P / (2 * math.pi)

    q = int(math.floor(q_float + 0.5)) % P

    # y = -q * inverse(t) mod P

    inverse_t = pow(t, -1, P)

    y = (-q * inverse_t) % P

    return y


# ============================================================
# 8. ВЫЧИСЛЕНИЕ A_t(x) ДЛЯ ВСЕХ x ПОЛЯ
# ============================================================

def calculate_all_A(t):
    """
    Вычисляет:

        A_t(x) = sum chi(u-x) * psi(t*v)

    для всех x от 0 до P-1.

    Суммирование идёт по всем аффинным точкам кривой.
    """

    result = []

    psi_values = {
        v: psi(t, v)
        for _, v in ALL_POINTS
    }

    for x in range(P):
        total = 0j

        for u, v in ALL_POINTS:
            total += chi(u - x) * psi_values[v]

        result.append(total)

    return result


# ============================================================
# 9. ОСНОВНОЙ ЭКСПЕРИМЕНТ ДЛЯ КАЖДОЙ ЧАСТОТЫ
# ============================================================

SUMMARY = {}
ALL_PHASE_ROWS = []
ALL_ORBIT_ROWS = []
ALL_SINGLE_FINGERPRINT_ROWS = []

for t in FREQUENCIES:

    print("\n" + "=" * 72)
    print(f"ИССЛЕДОВАНИЕ ДЛЯ ЧАСТОТЫ t = {t}")
    print("=" * 72)

    A_VALUES = calculate_all_A(t)

    # --------------------------------------------------------
    # 9.1. Проверка фазовых отношений
    # --------------------------------------------------------

    orbit_errors_1 = []
    orbit_errors_2 = []

    orbit_rows = []

    for orbit_number, orbit in enumerate(ORBITS, start=1):

        x0 = orbit[0]
        x1 = (OMEGA * x0) % P
        x2 = (OMEGA2 * x0) % P

        A0 = A_VALUES[x0]
        A1 = A_VALUES[x1]
        A2 = A_VALUES[x2]

        if abs(A0) < EPSILON:
            raise RuntimeError(
                f"A_t({x0}) слишком близко к нулю."
            )

        ratio_1 = A1 / A0
        ratio_2 = A2 / A0

        error_1 = abs(ratio_1 - CHI_OMEGA)
        error_2 = abs(ratio_2 - CHI_OMEGA2)

        orbit_errors_1.append(error_1)
        orbit_errors_2.append(error_2)

        orbit_rows.append({
            "частота": t,
            "номер_орбиты": orbit_number,
            "x_представитель": x0,
            "x_умноженное_на_139": x1,
            "x_умноженное_на_359": x2,
            "ошибка_отношения_139": error_1,
            "ошибка_отношения_359": error_2,
            "отношение_139_действительная_часть": ratio_1.real,
            "отношение_139_мнимая_часть": ratio_1.imag,
            "отношение_359_действительная_часть": ratio_2.real,
            "отношение_359_мнимая_часть": ratio_2.imag
        })

    max_orbit_error_1 = max(orbit_errors_1)
    max_orbit_error_2 = max(orbit_errors_2)

    print("\nФазовые отношения:")
    print(
        "Максимальная ошибка для omega:",
        max_orbit_error_1
    )
    print(
        "Максимальная ошибка для omega^2:",
        max_orbit_error_2
    )

    # --------------------------------------------------------
    # 9.2. Проверка различимости x по фазе A_t(x)
    # --------------------------------------------------------

    phase_by_x = {
        x: phase_of(A_VALUES[x])
        for x in SUBGROUP_X
    }

    phase_distances = []

    for i, x1 in enumerate(SUBGROUP_X):
        for x2 in SUBGROUP_X[:i]:
            phase_distances.append(
                phase_distance(
                    phase_by_x[x1],
                    phase_by_x[x2]
                )
            )

    min_phase_distance = min(phase_distances)

    phase_lookup_correct = 0
    phase_lookup_rows = []

    for x in SUBGROUP_X:

        target_phase = phase_by_x[x]

        distances = [
            (
                phase_distance(
                    target_phase,
                    phase_by_x[candidate_x]
                ),
                candidate_x
            )
            for candidate_x in SUBGROUP_X
        ]

        distances.sort()

        predicted_x = distances[0][1]

        if predicted_x == x:
            phase_lookup_correct += 1

        orbit = orbit_of_x(x)

        phase_lookup_rows.append({
            "частота": t,
            "x": x,
            "номер_орбиты": ORBITS.index(sorted(orbit)) + 1,
            "фаза_A_радианы": target_phase,
            "фаза_A_градусы": math.degrees(target_phase),
            "ближайший_x_по_фазе": predicted_x,
            "расстояние_до_ближайшей_фазы": distances[0][0],
            "правильное_распознавание": predicted_x == x
        })

    print("\nРаспознавание координаты x по фазе A:")
    print(
        f"Правильных распознаваний: "
        f"{phase_lookup_correct}/{len(SUBGROUP_X)}"
    )
    print(
        "Минимальное угловое расстояние:",
        min_phase_distance
    )

    # --------------------------------------------------------
    # 9.3. Полный комплексный отпечаток T_t(P)
    # --------------------------------------------------------

    T_BY_K = {}

    for k, point in SUBGROUP_POINTS.items():

        x, y = point

        T_BY_K[k] = (
            A_VALUES[x]
            * psi(-t, y)
        )

    # Проверяем уникальность и минимальную дистанцию.
    fingerprint_distances = []

    for i, k1 in enumerate(T_BY_K):
        for k2 in list(T_BY_K)[:i]:
            fingerprint_distances.append(
                abs(T_BY_K[k1] - T_BY_K[k2])
            )

    min_fingerprint_distance = min(
        fingerprint_distances
    )

    # --------------------------------------------------------
    # 9.4. Поиск по полной таблице эталонов
    # --------------------------------------------------------

    lookup_correct = 0
    lookup_min_distances = []

    for k, target in T_BY_K.items():

        distances = [
            (abs(target - reference), reference_k)
            for reference_k, reference in T_BY_K.items()
        ]

        distances.sort()

        # Первый элемент должен быть сам отпечаток.
        predicted_k = distances[0][1]

        if predicted_k == k:
            lookup_correct += 1

        # Второй элемент показывает расстояние
        # до ближайшего другого эталона.
        if len(distances) > 1:
            lookup_min_distances.append(distances[1][0])

    min_lookup_margin = min(lookup_min_distances)


    print("\nПолный комплексный отпечаток:")

    # Подсчитываем уникальные отпечатки после округления.
    unique_fingerprints = set()

    for z in T_BY_K.values():
        fingerprint = (
            round(z.real, 10),
            round(z.imag, 10)
        )
        unique_fingerprints.add(fingerprint)

    print(
        f"Уникальных отпечатков: "
        f"{len(unique_fingerprints)}/72"
    )

    print(
        "Минимальная дистанция между отпечатками:",
        min_fingerprint_distance
    )

    print(
        "Минимальный отрыв от второго эталона:",
        min_lookup_margin
    )

    print(
        f"Распознавание по полной таблице: "
        f"{lookup_correct}/72"
    )

    # --------------------------------------------------------
    # 9.5. Восстановление кандидатов по одному T_t(P)
    # --------------------------------------------------------

    # ВАЖНО:
    # Здесь мы НЕ используем таблицу k -> P.
    #
    # Для каждого отпечатка:
    # 1. Берём все x от 0 до 498.
    # 2. Делим T на A_t(x).
    # 3. Восстанавливаем возможный y из фазы.
    # 4. Проверяем уравнение кривой.
    #
    # Таким образом, получаем все точки кривой,
    # совместимые с одним комплексным отпечатком
    # в рамках этого способа восстановления.

    candidate_counts = []
    candidate_point_counts = Counter()

    single_fingerprint_rows = []

    for k, point in SUBGROUP_POINTS.items():

        true_x, true_y = point

        target_T = T_BY_K[k]

        candidates = []

        for candidate_x in range(P):

            candidate_A = A_VALUES[candidate_x]

            if abs(candidate_A) < EPSILON:
                continue

            candidate_y = recover_y_from_phase(
                target_T,
                candidate_A,
                t
            )

            if candidate_y is None:
                continue

            # Проверяем уравнение кривой.
            left = (candidate_y * candidate_y) % P

            right = (
                candidate_x ** 3 + B_CURVE
            ) % P

            if left == right:
                candidates.append(
                    (candidate_x, candidate_y)
                )

        candidates = sorted(set(candidates))

        candidate_count = len(candidates)

        candidate_counts.append(candidate_count)

        candidate_point_counts[candidate_count] += 1

        true_point_found = (
            (true_x, true_y) in candidates
        )

        candidate_text = "; ".join(
            f"{x}:{y}"
            for x, y in candidates
        )

        single_fingerprint_rows.append({
            "частота": t,
            "k_для_проверки": k,
            "исходный_x": true_x,
            "исходный_y": true_y,
            "число_кандидатов": candidate_count,
            "исходная_точка_найдена": true_point_found,
            "кандидаты_x_y": candidate_text
        })

    candidate_success = sum(
        1
        for row in single_fingerprint_rows
        if row["исходная_точка_найдена"]
    )

    average_candidate_count = (
        sum(candidate_counts) / len(candidate_counts)
    )

    print("\nВосстановление по одному отпечатку:")
    print(
        f"Исходная точка присутствует среди кандидатов: "
        f"{candidate_success}/72"
    )
    print(
        "Среднее число кандидатов:",
        average_candidate_count
    )
    print(
        "Распределение числа кандидатов:",
        dict(sorted(candidate_point_counts.items()))
    )

    # --------------------------------------------------------
    # 9.6. Сохранение промежуточных таблиц
    # --------------------------------------------------------

    ALL_ORBIT_ROWS.extend(orbit_rows)
    ALL_PHASE_ROWS.extend(phase_lookup_rows)
    ALL_SINGLE_FINGERPRINT_ROWS.extend(
        single_fingerprint_rows
    )

    # --------------------------------------------------------
    # 9.7. Сводка частоты
    # --------------------------------------------------------

    SUMMARY[str(t)] = {
        "частота": t,

        "проверка_фазовых_отношений": {
            "ожидаемое_отношение_139": {
                "действительная_часть": CHI_OMEGA.real,
                "мнимая_часть": CHI_OMEGA.imag
            },
            "ожидаемое_отношение_359": {
                "действительная_часть": CHI_OMEGA2.real,
                "мнимая_часть": CHI_OMEGA2.imag
            },
            "максимальная_ошибка_139": max_orbit_error_1,
            "максимальная_ошибка_359": max_orbit_error_2,
            "число_проверенных_орбит": len(ORBITS)
        },

        "фаза_A": {
            "число_координат_x": len(SUBGROUP_X),
            "правильных_распознаваний": phase_lookup_correct,
            "минимальное_угловое_расстояние": min_phase_distance
        },

        "комплексный_отпечаток_T": {
            "число_точек": len(T_BY_K),
            "минимальная_дистанция": min_fingerprint_distance,
            "минимальный_отрыв_до_второго_эталона": min_lookup_margin,
            "правильных_распознаваний_по_таблице": lookup_correct
        },

        "один_отпечаток": {
            "исходная_точка_найдена_среди_кандидатов": candidate_success,
            "число_проверенных_точек": 72,
            "среднее_число_кандидатов": average_candidate_count,
            "распределение_числа_кандидатов": {
                str(key): value
                for key, value in sorted(
                    candidate_point_counts.items()
                )
            }
        }
    }


# ============================================================
# 10. СОХРАНЕНИЕ CSV
# ============================================================

def save_csv(filename, rows):
    """Сохранение списка словарей в CSV."""
    path = OUTPUT_DIR / filename

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            delimiter=";"
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"Создан файл: {path}")


save_csv(
    "B7_фазовые_отношения.csv",
    ALL_ORBIT_ROWS
)

save_csv(
    "B7_фазы_координат_x.csv",
    ALL_PHASE_ROWS
)

save_csv(
    "B7_восстановление_по_одному_отпечатку.csv",
    ALL_SINGLE_FINGERPRINT_ROWS
)


# ============================================================
# 11. СОХРАНЕНИЕ JSON
# ============================================================

json_path = OUTPUT_DIR / "B7_итоги.json"

full_summary = {
    "название": "Эксперимент B7",
    "поле": P,
    "кривая": "y^2 = x^3 + 7 mod 499",
    "базовая_точка": list(G),
    "порядок_базовой_точки": ORDER,
    "кубические_корни_единицы": [
        OMEGA,
        OMEGA2
    ],
    "число_аффинных_точек_кривой": len(ALL_POINTS),
    "число_точек_подгруппы_без_бесконечности": 72,
    "число_координат_x_подгруппы": 36,
    "число_орбит": len(ORBITS),
    "частоты": SUMMARY
}

with open(
    json_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        full_summary,
        file,
        ensure_ascii=False,
        indent=4
    )

print(f"Создан файл: {json_path}")


# ============================================================
# 12. README
# ============================================================

readme_text = f"""
ЭКСПЕРИМЕНТ B7
===============

Цель
----

Проверить тройную фазовую симметрию комплексных отпечатков
и исследовать возможность восстановления точки по одному
комплексному отпечатку.

Используемая кривая
-------------------

y^2 = x^3 + 7 (mod {P})

Базовая точка:
G = {G}

Порядок базовой точки:
{ORDER}

Кубические корни единицы:
omega = {OMEGA}
omega^2 = {OMEGA2}

Исследуемые частоты:
{FREQUENCIES}

Определения
-----------

Кубический характер:

chi(0) = 0

chi(z) = exp(2*pi*i*e/3),
если z = 7^e mod {P}.

Аддитивный характер:

psi(z) = exp(2*pi*i*z/{P}).

Комплексная функция:

A_t(x) = sum chi(u-x)*psi(t*v)

Суммирование производится по всем аффинным точкам
эллиптической кривой.

Отпечаток точки:

T_t(x,y) = A_t(x)*psi(-t*y).

Что проверяется
---------------

1. Фазовые отношения:

   A_t(omega*x)/A_t(x) = chi(omega)

   A_t(omega^2*x)/A_t(x) = chi(omega^2)

2. Различимость 36 координат x по фазе A_t(x).

3. Распознавание комплексного отпечатка по полной таблице
   эталонов подгруппы.

4. Восстановление кандидатов по одному T_t(P):

   - перебираются все x из поля;
   - из фазы T/A восстанавливается y;
   - проверяется уравнение кривой.

Создаваемые файлы
-----------------

B7_фазовые_отношения.csv
    Проверка фазовых отношений на всех орбитах.

B7_фазы_координат_x.csv
    Фазы A_t(x) для координат подгруппы.

B7_восстановление_по_одному_отпечатку.csv
    Кандидаты (x,y), полученные из одного T_t(P).

B7_итоги.json
    Сводные численные результаты.

Ограничения интерпретации
-------------------------

Поиск по полной таблице эталонов является lookup-процедурой.

Перебор всех x поля не является алгоритмом сублинейной
сложности по отношению к размеру поля.

Восстановление y требует кандидата x, поскольку используется
отношение T_t(P)/A_t(x).

Если по одному отпечатку остаётся несколько кандидатов,
это не считается однозначным восстановлением точки.

Эксперимент не является решением ECDLP и не демонстрирует
ускорение решения задачи дискретного логарифмирования
на эллиптической кривой.

Все комплексные вычисления выполняются с плавающей точкой.
Поэтому численные фазовые проверки не являются формальным
доказательством точных алгебраических равенств.
"""

readme_path = OUTPUT_DIR / "README_B7.txt"

with open(
    readme_path,
    "w",
    encoding="utf-8"
) as file:
    file.write(readme_text)

print(f"Создан файл: {readme_path}")


# ============================================================
# 13. СОЗДАНИЕ ZIP-АРХИВА
# ============================================================

archive_path = Path("experimentB7_results.zip")

with zipfile.ZipFile(
    archive_path,
    "w",
    compression=zipfile.ZIP_DEFLATED
) as archive:

    for file_path in OUTPUT_DIR.iterdir():

        if file_path.is_file():
            archive.write(
                file_path,
                arcname=file_path.name
            )

    # Добавляем сам скрипт в архив,
    # если он запущен как отдельный файл.
    try:
        script_path = Path(__file__).resolve()

        if script_path.is_file():
            archive.write(
                script_path,
                arcname="experimentB7.py"
            )

    except NameError:
        pass


# ============================================================
# 14. ИТОГОВЫЙ ВЫВОД
# ============================================================

print("\n" + "=" * 72)
print("ЭКСПЕРИМЕНТ B7 ЗАВЕРШЁН")
print("=" * 72)

for frequency, data in SUMMARY.items():

    phase_data = data["проверка_фазовых_отношений"]
    x_data = data["фаза_A"]
    fingerprint_data = data["комплексный_отпечаток_T"]
    single_data = data["один_отпечаток"]

    print(f"\nЧастота t = {frequency}")

    print(
        "Максимальная ошибка отношения для omega:",
        phase_data["максимальная_ошибка_139"]
    )

    print(
        "Максимальная ошибка отношения для omega^2:",
        phase_data["максимальная_ошибка_359"]
    )

    print(
        "Распознавание x по фазе:",
        f'{x_data["правильных_распознаваний"]}/36'
    )

    print(
        "Минимальное угловое расстояние:",
        x_data["минимальное_угловое_расстояние"]
    )

    print(
        "Минимальная дистанция отпечатков T:",
        fingerprint_data["минимальная_дистанция"]
    )

    print(
        "Распознавание T по полной таблице:",
        f'{fingerprint_data["правильных_распознаваний_по_таблице"]}/72'
    )

    print(
        "Исходная точка найдена среди кандидатов:",
        f'{single_data["исходная_точка_найдена_среди_кандидатов"]}/72'
    )

    print(
        "Среднее число кандидатов по одному T:",
        single_data["среднее_число_кандидатов"]
    )

print("\nСоздана папка:", OUTPUT_DIR)
print("Создан ZIP-архив:", archive_path)

print("\nВсе результаты сохранены.")
