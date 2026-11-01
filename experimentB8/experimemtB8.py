
# -*- coding: utf-8 -*-

"""
ЭКСПЕРИМЕНТ B8

Тема:
Восстановление точки эллиптической кривой по одному
комплексному отпечатку с проверкой точности комплексного отношения.

Кривая:
    y^2 = x^3 + 7 (mod 499)

Базовая точка:
    G = (3, 212)

Порядок базовой точки:
    73

Исследуемые частоты:
    t = 1
    t = 174

В отличие от B7, здесь недостаточно округлить фазу.
Для каждого кандидата проверяется полное комплексное равенство:

    T_t / A_t(x) = exp(-2*pi*i*t*y/499)

Проверяется:
1. Восстановление всех 510 аффинных точек кривой.
2. Число ложных кандидатов без проверки комплексной ошибки.
3. Однозначность при разных численных допусках.
4. Минимальное расстояние до ложного кандидата.
5. Устойчивость к комплексному шуму.
6. Вычислительная стоимость восстановления.

Создаваемые файлы:
    B8_восстановление_точек.csv
    B8_устойчивость_к_шуму.csv
    B8_проверка_допусков.csv
    B8_итоги.json
    README_B8.txt
    experimentB8_results.zip

Используются только стандартные библиотеки Python.
"""

import cmath
import csv
import json
import math
import zipfile

from pathlib import Path
from collections import Counter


# ============================================================
# 1. ПАРАМЕТРЫ
# ============================================================

P = 499

B_CURVE = 7

G = (3, 212)

ORDER = 73

FREQUENCIES = [1, 174]

# Численные пороги.
ZERO_EPSILON = 1e-12

TOLERANCES = [
    1e-12,
    1e-10,
    1e-8,
    1e-6,
    1e-4,
    1e-3,
    1e-2
]

NOISE_LEVELS = [
    0,
    1e-8,
    1e-6,
    1e-4,
    1e-3,
    1e-2
]

NOISE_TEST_TOLERANCE = 1e-3

OUTPUT_DIR = Path("experimentB8_results")
OUTPUT_DIR.mkdir(exist_ok=True)

TWO_PI = 2 * math.pi

print("=" * 76)
print("ЭКСПЕРИМЕНТ B8")
print("ТОЧНОСТЬ ВОССТАНОВЛЕНИЯ ТОЧКИ ПО ОДНОМУ ОТПЕЧАТКУ")
print("=" * 76)


# ============================================================
# 2. КУБИЧЕСКИЙ ХАРАКТЕР
# ============================================================

def is_prime(n):
    """Проверка простоты числа."""

    if n < 2:
        return False

    for d in range(2, math.isqrt(n) + 1):
        if n % d == 0:
            return False

    return True


assert is_prime(P)

# Число 7 используется как первообразный корень.
LOG_TABLE = {}

value = 1

for exponent in range(P - 1):
    LOG_TABLE[value] = exponent
    value = (value * 7) % P

assert len(LOG_TABLE) == P - 1

def chi(z):
    """Кубический мультипликативный характер."""

    z %= P

    if z == 0:
        return 0j

    exponent = LOG_TABLE[z] % 3

    return cmath.exp(
        2j * math.pi * exponent / 3
    )


# ============================================================
# 3. АДДИТИВНЫЙ ХАРАКТЕР
# ============================================================

def psi(z):
    """Аддитивный характер."""

    z %= P

    return cmath.exp(
        2j * math.pi * z / P
    )


def psi_frequency(t, z):
    """Аддитивный характер с заданной частотой."""

    return psi(t * z)


# ============================================================
# 4. ЭЛЛИПТИЧЕСКАЯ КРИВАЯ
# ============================================================

def is_on_curve(x, y):
    """Проверка принадлежности точки кривой."""

    return (
        y * y - x * x * x - B_CURVE
    ) % P == 0


ALL_POINTS = []

for x in range(P):

    rhs = (x ** 3 + B_CURVE) % P

    for y in range(P):

        if (y * y) % P == rhs:
            ALL_POINTS.append((x, y))


POINT_SET = set(ALL_POINTS)

assert len(ALL_POINTS) == 510

print("\nКривая:")
print("Уравнение: y^2 = x^3 + 7 mod 499")
print("Аффинных точек:", len(ALL_POINTS))


# ============================================================
# 5. ПРОВЕРКА БАЗОВОЙ ТОЧКИ И ПОДГРУППЫ
# ============================================================

def point_add(P1, P2):
    """Сложение точек эллиптической кривой."""

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

        numerator = 3 * x1 * x1
        denominator = 2 * y1

    else:

        numerator = y2 - y1
        denominator = x2 - x1

    slope = (
        numerator
        * pow(denominator % P, -1, P)
    ) % P

    x3 = (
        slope * slope - x1 - x2
    ) % P

    y3 = (
        slope * (x1 - x3) - y1
    ) % P

    return x3, y3


def scalar_mult(k, point):
    """Умножение точки на скаляр."""

    result = None
    current = point

    while k > 0:

        if k & 1:
            result = point_add(result, current)

        current = point_add(current, current)

        k >>= 1

    return result


assert is_on_curve(*G)

current = None
SUBGROUP = {}

for k in range(ORDER):

    SUBGROUP[k] = current

    current = point_add(current, G)

assert current is None

SUBGROUP_POINTS = {
    point
    for point in SUBGROUP.values()
    if point is not None
}

assert len(SUBGROUP_POINTS) == 72

print("\nПодгруппа:")
print("Порядок G:", ORDER)
print("Ненулевых точек:", len(SUBGROUP_POINTS))
print(
    "Точек подгруппы среди всех аффинных точек:",
    len(SUBGROUP_POINTS)
)


# ============================================================
# 6. ВЫЧИСЛЕНИЕ A_t(x)
# ============================================================

def calculate_A(t):
    """
    Вычисляет:

        A_t(x) = sum chi(u-x) * psi(t*v)

    Суммирование производится по всем 510 аффинным
    точкам эллиптической кривой.

    Возвращает список из 499 комплексных значений.
    """

    result = []

    psi_values = [
        psi_frequency(t, v)
        for v in range(P)
    ]

    for x in range(P):

        total = 0j

        for u, v in ALL_POINTS:

            total += (
                chi(u - x)
                * psi_values[v]
            )

        result.append(total)

    return result


# ============================================================
# 7. ФОРМИРОВАНИЕ ОТПЕЧАТКА
# ============================================================

def calculate_T(t, point, A_values):
    """
    T_t(x,y) = A_t(x) * psi(-t*y)
    """

    x, y = point

    return (
        A_values[x]
        * psi_frequency(-t, y)
    )


# ============================================================
# 8. ПОИСК КАНДИДАТОВ
# ============================================================

def find_candidates(
    T_value,
    t,
    A_values
):
    """
    Ищет все точки кривой, согласующиеся с T_value.

    Для каждого x:

        ratio = T_value / A_t(x)

    Затем находим ближайший корень 499-й степени:

        root = exp(2*pi*i*q/499)

    Получаем:

        y = -q * inverse(t) mod 499

    После чего проверяем:

        y^2 = x^3 + 7 mod 499

    И дополнительно вычисляем полную комплексную ошибку:

        error = abs(ratio - root)

    Возвращает список:

        (x, y, error)
    """

    inverse_t = pow(t, -1, P)

    candidates = []

    for x in range(P):

        A_value = A_values[x]

        if abs(A_value) < ZERO_EPSILON:
            continue

        ratio = T_value / A_value

        angle = cmath.phase(ratio) % TWO_PI

        q_float = angle * P / TWO_PI

        q = int(
            math.floor(q_float + 0.5)
        ) % P

        y = (
            -q * inverse_t
        ) % P

        # Проверка принадлежности кривой.
        if not is_on_curve(x, y):
            continue

        # Корень единицы, соответствующий y.
        expected_ratio = psi(q)

        # Полная комплексная ошибка.
        error = abs(
            ratio - expected_ratio
        )

        candidates.append(
            (x, y, error)
        )

    candidates.sort(
        key=lambda item: item[2]
    )

    return candidates


# ============================================================
# 9. КЛАССИФИКАЦИЯ РЕЗУЛЬТАТА
# ============================================================

def classify_candidates(
    candidates,
    true_point,
    tolerance
):
    """
    Классифицирует кандидатов после применения допуска.

    Возможные результаты:

    unique_correct:
        остался один кандидат, и он правильный.

    ambiguous:
        осталось несколько кандидатов.

    none:
        не осталось кандидатов.

    unique_wrong:
        остался один, но неправильный кандидат.
    """

    accepted = [
        (x, y, error)
        for x, y, error in candidates
        if error <= tolerance
    ]

    if len(accepted) == 0:
        return "none", accepted

    if len(accepted) > 1:
        return "ambiguous", accepted

    candidate = (
        accepted[0][0],
        accepted[0][1]
    )

    if candidate == true_point:
        return "unique_correct", accepted

    return "unique_wrong", accepted


# ============================================================
# 10. ОСНОВНОЙ ЭКСПЕРИМЕНТ
# ============================================================

ALL_RECOVERY_ROWS = []
ALL_TOLERANCE_ROWS = []
ALL_NOISE_ROWS = []

SUMMARY = {}

for t in FREQUENCIES:

    print("\n" + "=" * 76)
    print(f"ИССЛЕДОВАНИЕ ЧАСТОТЫ t = {t}")
    print("=" * 76)

    A_VALUES = calculate_A(t)

    zero_count = sum(
        1
        for z in A_VALUES
        if abs(z) < ZERO_EPSILON
    )

    print("Нулевых значений A_t(x):", zero_count)

    # --------------------------------------------------------
    # 10.1. Восстановление всех 510 точек
    # --------------------------------------------------------

    frequency_rows = []

    tolerance_counters = {
        tolerance: Counter()
        for tolerance in TOLERANCES
    }

    max_true_error = 0.0
    min_false_error = float("inf")

    raw_candidate_counter = Counter()

    for point_index, point in enumerate(ALL_POINTS):

        x_true, y_true = point

        T_value = calculate_T(
            t,
            point,
            A_VALUES
        )

        candidates = find_candidates(
            T_value,
            t,
            A_VALUES
        )

        raw_candidate_counter[
            len(candidates)
        ] += 1

        true_candidates = [
            item
            for item in candidates
            if (item[0], item[1]) == point
        ]

        if len(true_candidates) != 1:
            raise RuntimeError(
                f"Исходная точка не найдена: {point}"
            )

        true_error = true_candidates[0][2]

        max_true_error = max(
            max_true_error,
            true_error
        )

        false_errors = [
            error
            for x, y, error in candidates
            if (x, y) != point
        ]

        if false_errors:

            min_false_error = min(
                min_false_error,
                min(false_errors)
            )

        candidate_text = "; ".join(
            f"{x}:{y}:{error:.12g}"
            for x, y, error in candidates
        )

        row = {
            "частота": t,
            "исходный_x": x_true,
            "исходный_y": y_true,
            "исходная_точка_в_подгруппе": (
                point in SUBGROUP_POINTS
            ),
            "кандидатов_до_допуска": len(candidates),
            "ошибка_правильного_кандидата": true_error,
            "минимальная_ошибка_ложного_кандидата": (
                min(false_errors)
                if false_errors
                else None
            ),
            "кандидаты_x_y_ошибка": candidate_text
        }

        frequency_rows.append(row)
        ALL_RECOVERY_ROWS.append(row)

        # Проверяем все численные допуски.
        for tolerance in TOLERANCES:

            status, accepted = classify_candidates(
                candidates,
                point,
                tolerance
            )

            tolerance_counters[tolerance][status] += 1

    # --------------------------------------------------------
    # 10.2. Итоги по численным допускам
    # --------------------------------------------------------

    print("\nРезультаты по допускам:")

    for tolerance in TOLERANCES:

        counter = tolerance_counters[tolerance]

        row = {
            "частота": t,
            "допуск": tolerance,
            "однозначно_правильно": counter["unique_correct"],
            "неоднозначно": counter["ambiguous"],
            "без_кандидатов": counter["none"],
            "однозначно_неправильно": counter["unique_wrong"]
        }

        ALL_TOLERANCE_ROWS.append(row)

        print(
            f"Допуск {tolerance:g}: "
            f"правильно {counter['unique_correct']}/510, "
            f"неоднозначно {counter['ambiguous']}, "
            f"пусто {counter['none']}"
        )

    # --------------------------------------------------------
    # 10.3. Проверка устойчивости к комплексному шуму
    # --------------------------------------------------------

    print("\nПроверка комплексного шума:")

    noise_results = {}

    for noise_level in NOISE_LEVELS:

        counter = Counter()

        for point_index, point in enumerate(ALL_POINTS):

            T_value = calculate_T(
                t,
                point,
                A_VALUES
            )

            if noise_level > 0:

                # Детерминированное направление шума.
                # Для каждой точки используется свой угол.

                noise_angle = (
                    (point_index + 1)
                    * 2.399963229728653
                ) % TWO_PI

                noise = (
                    noise_level
                    * abs(T_value)
                    * cmath.exp(1j * noise_angle)
                )

                noisy_T = T_value + noise

            else:

                noisy_T = T_value

            candidates = find_candidates(
                noisy_T,
                t,
                A_VALUES
            )

            status, accepted = classify_candidates(
                candidates,
                point,
                NOISE_TEST_TOLERANCE
            )

            counter[status] += 1

        noise_results[str(noise_level)] = {
            "однозначно_правильно": counter["unique_correct"],
            "неоднозначно": counter["ambiguous"],
            "без_кандидатов": counter["none"],
            "однозначно_неправильно": counter["unique_wrong"]
        }

        noise_row = {
            "частота": t,
            "относительный_шум": noise_level,
            "допуск": NOISE_TEST_TOLERANCE,
            "однозначно_правильно": counter["unique_correct"],
            "неоднозначно": counter["ambiguous"],
            "без_кандидатов": counter["none"],
            "однозначно_неправильно": counter["unique_wrong"]
        }

        ALL_NOISE_ROWS.append(noise_row)

        print(
            f"Шум {noise_level:g}: "
            f"правильно {counter['unique_correct']}/510, "
            f"неоднозначно {counter['ambiguous']}, "
            f"пусто {counter['none']}"
        )

    # --------------------------------------------------------
    # 10.4. Сводка частоты
    # --------------------------------------------------------

    SUMMARY[str(t)] = {
        "частота": t,
        "число_точек": len(ALL_POINTS),
        "нулевых_A": zero_count,
        "распределение_кандидатов_до_допуска": {
            str(k): v
            for k, v in sorted(
                raw_candidate_counter.items()
            )
        },
        "максимальная_ошибка_правильного_кандидата": (
            max_true_error
        ),
        "минимальная_ошибка_ложного_кандидата": (
            min_false_error
            if min_false_error != float("inf")
            else None
        ),
        "результаты_по_допускам": {
            str(tolerance): dict(counter)
            for tolerance, counter
            in tolerance_counters.items()
        },
        "результаты_шума": noise_results
    }


# ============================================================
# 11. СОХРАНЕНИЕ CSV
# ============================================================

def save_csv(filename, rows):

    path = OUTPUT_DIR / filename

    if not rows:
        return

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
            delimiter=";"
        )

        writer.writeheader()
        writer.writerows(rows)

    print("Создан файл:", path)


save_csv(
    "B8_восстановление_точек.csv",
    ALL_RECOVERY_ROWS
)

save_csv(
    "B8_проверка_допусков.csv",
    ALL_TOLERANCE_ROWS
)

save_csv(
    "B8_устойчивость_к_шуму.csv",
    ALL_NOISE_ROWS
)


# ============================================================
# 12. ВЫЧИСЛИТЕЛЬНАЯ СЛОЖНОСТЬ
# ============================================================

# Для каждой частоты:
#
# Построение A:
#   499 координат x * 510 точек кривой.
#
# Поиск кандидата:
#   510 исходных точек * 499 возможных x.
#
# Это не сублинейный алгоритм.
#
# Важно также помнить:
# восстановление точки P не означает автоматического
# восстановления скаляра k, для которого P = kG.

operations_A = P * len(ALL_POINTS)

operations_search = (
    len(ALL_POINTS) * P
)

COMPUTATION_INFO = {
    "суммирований_при_построении_A_на_одну_частоту": (
        operations_A
    ),
    "проверок_x_при_восстановлении_всех_точек": (
        operations_search
    ),
    "частот": len(FREQUENCIES),
    "общее_число_суммирований_A": (
        operations_A * len(FREQUENCIES)
    ),
    "общее_число_проверок_x": (
        operations_search * len(FREQUENCIES)
    )
}


# ============================================================
# 13. СОХРАНЕНИЕ JSON
# ============================================================

full_summary = {
    "название": "Эксперимент B8",
    "кривая": "y^2 = x^3 + 7 mod 499",
    "модуль": P,
    "базовая_точка": list(G),
    "порядок_базовой_точки": ORDER,
    "число_аффинных_точек": len(ALL_POINTS),
    "исследуемые_частоты": FREQUENCIES,
    "допуски": TOLERANCES,
    "допуск_шума": NOISE_TEST_TOLERANCE,
    "уровни_шума": NOISE_LEVELS,
    "вычислительная_стоимость": COMPUTATION_INFO,
    "результаты": SUMMARY
}

json_path = OUTPUT_DIR / "B8_итоги.json"

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

print("\nСоздан файл:", json_path)


# ============================================================
# 14. README
# ============================================================

README = """
ЭКСПЕРИМЕНТ B8
===============

Цель
----

Проверить возможность восстановления точки эллиптической
кривой по одному комплексному отпечатку с использованием
полного комплексного равенства.

Кривая
------

y^2 = x^3 + 7 mod 499

Базовая точка:
G = (3, 212)

Порядок:
73

Определения
-----------

Кубический характер:

chi(0) = 0

chi(z) = exp(2*pi*i*e/3),
если z = 7^e mod 499.

Аддитивный характер:

psi(z) = exp(2*pi*i*z/499).

Функция:

A_t(x) = sum chi(u-x)*psi(t*v).

Отпечаток:

T_t(x,y) = A_t(x)*psi(-t*y).

Главная проверка
----------------

Для каждого возможного x вычисляется:

ratio = T_t / A_t(x).

Из фазы ratio определяется предполагаемый y.

Затем проверяется уравнение кривой:

y^2 = x^3 + 7 mod 499.

Но этого недостаточно.

Дополнительно проверяется полная комплексная ошибка:

error = abs(ratio - exp(2*pi*i*q/499)),

где q = -t*y mod 499.

Кандидат принимается только тогда, когда ошибка
не превышает заданный численный допуск.

Содержимое файлов
-----------------

B8_восстановление_точек.csv
    Кандидаты для всех 510 аффинных точек.

B8_проверка_допусков.csv
    Результаты при разных численных допусках.

B8_устойчивость_к_шуму.csv
    Проверка восстановления при комплексном шуме.

B8_итоги.json
    Сводные результаты.

Важные ограничения
------------------

1. Эксперимент использует заранее вычисленную таблицу A_t(x).

2. Для каждой точки перебираются все 499 координат x.

3. Результат относится к восстановлению точки P, а не скаляра k.

4. После восстановления P задача P = kG остаётся отдельной
   задачей дискретного логарифмирования.

5. Все комплексные вычисления выполняются с плавающей точкой.
   Численный допуск не заменяет точное алгебраическое
   доказательство.

6. Проверка шума использует одно детерминированное направление
   комплексного шума для каждой точки и уровня шума.

7. Результаты на поле размера 499 не доказывают аналогичное
   поведение на secp256k1.

Главная цель следующего этапа
-----------------------------

Исследовать, можно ли использовать структуру отпечатка
не только для восстановления точки, но и для получения
информации о скаляре k без полного перебора.
"""

readme_path = OUTPUT_DIR / "README_B8.txt"

with open(
    readme_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(README)

print("Создан файл:", readme_path)


# ============================================================
# 15. СОЗДАНИЕ ZIP-АРХИВА
# ============================================================

archive_path = Path("experimentB8_results.zip")

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

    try:

        script_path = Path(__file__).resolve()

        if script_path.is_file():

            archive.write(
                script_path,
                arcname="experimentB8.py"
            )

    except NameError:
        pass


# ============================================================
# 16. ИТОГОВЫЙ ВЫВОД
# ============================================================

print("\n" + "=" * 76)
print("ЭКСПЕРИМЕНТ B8 ЗАВЕРШЁН")
print("=" * 76)

for frequency, result in SUMMARY.items():

    print(f"\nЧастота t = {frequency}")

    print(
        "Максимальная ошибка правильного кандидата:",
        result[
            "максимальная_ошибка_правильного_кандидата"
        ]
    )

    print(
        "Минимальная ошибка ложного кандидата:",
        result[
            "минимальная_ошибка_ложного_кандидата"
        ]
    )

    print(
        "Распределение кандидатов до допуска:",
        result[
            "распределение_кандидатов_до_допуска"
        ]
    )

    for tolerance, values in (
        result["результаты_по_допускам"].items()
    ):

        print(
            f"Допуск {tolerance}: "
            f"правильно = {values.get('unique_correct', 0)}, "
            f"неоднозначно = {values.get('ambiguous', 0)}, "
            f"без кандидатов = {values.get('none', 0)}"
        )

print("\nВычислительная стоимость:")
print(
    json.dumps(
        COMPUTATION_INFO,
        ensure_ascii=False,
        indent=4
    )
)

print("\nПапка результатов:", OUTPUT_DIR)
print("ZIP-архив:", archive_path)

print("\nВсе результаты сохранены.")
