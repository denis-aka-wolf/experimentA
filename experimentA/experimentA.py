
import csv
import cmath
import math
import zipfile
import json
import os
from collections import defaultdict

# ==========================================
# 1. ПАРАМЕТРЫ ЭЛЛИПТИЧЕСКОЙ КРИВОЙ
# ==========================================

p = 499
a = 0
b = 7

G = (3, 212)
Q = (449, 483)

# ==========================================
# 2. АРИФМЕТИКА ЭЛЛИПТИЧЕСКОЙ КРИВОЙ
# ==========================================

O = None


def inverse(x):
    return pow(x % p, -1, p)


def point_add(P, Q):
    if P is None:
        return Q

    if Q is None:
        return P

    x1, y1 = P
    x2, y2 = Q

    if x1 == x2 and (y1 + y2) % p == 0:
        return None

    if P == Q:
        if y1 == 0:
            return None

        lam = ((3*x1*x1 + a) * inverse(2*y1)) % p

    else:
        lam = ((y2-y1) * inverse(x2-x1)) % p

    x3 = (lam*lam - x1 - x2) % p
    y3 = (lam*(x1-x3) - y1) % p

    return (x3, y3)


def scalar_mult(k, P):
    R = None

    while k > 0:
        if k & 1:
            R = point_add(R, P)

        P = point_add(P, P)
        k >>= 1

    return R


def point_neg(P):
    if P is None:
        return None

    return (P[0], (-P[1]) % p)


# ==========================================
# 3. ПРОВЕРКА ПАРАМЕТРОВ
# ==========================================

def is_on_curve(P):
    if P is None:
        return True

    x, y = P

    return (y*y - x*x*x - a*x - b) % p == 0


assert is_on_curve(G)
assert is_on_curve(Q)

# ==========================================
# 4. ПОСТРОЕНИЕ ПОДГРУППЫ
# ==========================================

points = []

P = None
k = 0

while True:
    points.append({
        "k": k,
        "point": P
    })

    P = point_add(P, G)
    k += 1

    if P is None:
        break

order = len(points)

print("Порядок G:", order)
print("Проверка 371G:", scalar_mult(371, G))
print("Заданный Q:", Q)

assert scalar_mult(371, G) == Q

# ==========================================
# 5. МУЛЬТИПЛИКАТИВНЫЕ ХАРАКТЕРЫ
# ==========================================

def legendre(x):
    x %= p

    if x == 0:
        return 0

    return 1 if pow(x, (p-1)//2, p) == 1 else -1


# Примитивный корень поля F_p

def primitive_root():
    factors = []

    n = p-1
    d = 2

    while d*d <= n:
        if n % d == 0:
            factors.append(d)

            while n % d == 0:
                n //= d

        d += 1

    if n > 1:
        factors.append(n)

    for g in range(2, p):
        if all(
            pow(g, (p-1)//q, p) != 1
            for q in factors
        ):
            return g

    raise ValueError("Primitive root not found")


g = primitive_root()

# Таблица дискретных логарифмов относительно g

log_table = {}

value = 1

for i in range(p-1):
    log_table[value] = i
    value = (value*g) % p


def multiplicative_character(x, m):
    x %= p

    if x == 0:
        return 0j

    exponent = log_table[x] % m

    return cmath.exp(
        2j * math.pi * exponent / m
    )


# ==========================================
# 6. АДДИТИВНЫЙ ХАРАКТЕР
# ==========================================

def additive_character(x, frequency=1):
    return cmath.exp(
        2j * math.pi * (frequency*x % p) / p
    )


# ==========================================
# 7. ХАРАКТЕРНЫЕ СУММЫ
# ==========================================

def character_sums(P):

    if P is None:
        return {
            "S2": 0,
            "S3": 0j,
            "S6": 0j,
            "A1": 0j,
            "A2": 0j
        }

    x0, y0 = P

    S2 = 0
    S3 = 0j
    S6 = 0j

    A1 = 0j
    A2 = 0j

    for x in range(p):

        f = (x**3 + a*x + b) % p

        chi2 = legendre(f)

        S2 += chi2

        S3 += (
            multiplicative_character(x-x0, 3)
            * chi2
        )

        S6 += (
            multiplicative_character(x-x0, 6)
            * chi2
        )

        A1 += (
            additive_character(x-x0, 1)
            * chi2
        )

        A2 += (
            additive_character(x-x0, 2)
            * chi2
        )

    return {
        "S2": S2,
        "S3": S3,
        "S6": S6,
        "A1": A1,
        "A2": A2
    }


# ==========================================
# 8. ВЫЧИСЛЕНИЕ ОТПЕЧАТКОВ
# ==========================================

results = []

for item in points:

    k = item["k"]
    P = item["point"]

    if P is None:
        x = None
        y = None
    else:
        x, y = P

    sums = character_sums(P)

    results.append({
        "k": k,
        "x": x,
        "y": y,
        "S2": sums["S2"],
        "S3_real": sums["S3"].real,
        "S3_imag": sums["S3"].imag,
        "S6_real": sums["S6"].real,
        "S6_imag": sums["S6"].imag,
        "A1_real": sums["A1"].real,
        "A1_imag": sums["A1"].imag,
        "A2_real": sums["A2"].real,
        "A2_imag": sums["A2"].imag
    })


# ==========================================
# 9. ПОИСК КОЛЛИЗИЙ
# ==========================================

def make_fingerprint(row, fields, precision=8):

    return tuple(
        round(float(row[field]), precision)
        for field in fields
    )


fingerprint_sets = {

    "S2": ["S2"],

    "S2_S3": [
        "S2",
        "S3_real",
        "S3_imag"
    ],

    "S2_S3_S6": [
        "S2",
        "S3_real",
        "S3_imag",
        "S6_real",
        "S6_imag"
    ],

    "ALL": [
        "S2",
        "S3_real",
        "S3_imag",
        "S6_real",
        "S6_imag",
        "A1_real",
        "A1_imag",
        "A2_real",
        "A2_imag"
    ]
}


collision_report = {}

for name, fields in fingerprint_sets.items():

    groups = defaultdict(list)

    for row in results:

        fingerprint = make_fingerprint(
            row,
            fields
        )

        groups[fingerprint].append(row["k"])

    collisions = {
        str(key): values
        for key, values in groups.items()
        if len(values) > 1
    }

    collision_report[name] = {
        "unique_fingerprints": len(groups),
        "total_points": len(results),
        "collision_groups": collisions
    }


# ==========================================
# 10. ПРОВЕРКА P И -P
# ==========================================

symmetry_report = []

for item in points:

    P = item["point"]

    if P is None:
        continue

    negP = point_neg(P)

    k_neg = next(
        entry["k"]
        for entry in points
        if entry["point"] == negP
    )

    row1 = results[item["k"]]
    row2 = results[k_neg]

    symmetry_report.append({
        "k": item["k"],
        "k_negative": k_neg,
        "same_x": row1["x"] == row2["x"],
        "same_y": row1["y"] == row2["y"],
        "same_S2": row1["S2"] == row2["S2"],
        "same_fingerprint": (
            make_fingerprint(row1, fingerprint_sets["ALL"])
            ==
            make_fingerprint(row2, fingerprint_sets["ALL"])
        )
    })


# ==========================================
# 11. СОХРАНЕНИЕ CSV
# ==========================================

os.makedirs("experiment_A", exist_ok=True)

with open(
    "experiment_A/character_fingerprints.csv",
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=results[0].keys()
    )

    writer.writeheader()
    writer.writerows(results)


with open(
    "experiment_A/symmetry.csv",
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=symmetry_report[0].keys()
    )

    writer.writeheader()
    writer.writerows(symmetry_report)


with open(
    "experiment_A/collisions.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        collision_report,
        file,
        ensure_ascii=False,
        indent=2
    )


# ==========================================
# 12. ИТОГОВЫЙ ОТЧЁТ
# ==========================================

report = {
    "curve": {
        "p": p,
        "a": a,
        "b": b,
        "G": G,
        "Q": Q,
        "order": order
    },
    "primitive_root": g,
    "collision_report": collision_report,
    "symmetry_checks": len(symmetry_report),
    "points": len(results)
}


with open(
    "experiment_A/report.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        report,
        file,
        ensure_ascii=False,
        indent=2
    )


# ==========================================
# 13. СОЗДАНИЕ АРХИВА
# ==========================================

with zipfile.ZipFile(
    "experiment_A.zip",
    "w",
    zipfile.ZIP_DEFLATED
) as archive:

    for filename in os.listdir("experiment_A"):

        path = os.path.join(
            "experiment_A",
            filename
        )

        archive.write(
            path,
            arcname=filename
        )


# ==========================================
# 14. ВЫВОД РЕЗУЛЬТАТОВ
# ==========================================

print("\n========== EXPERIMENT A ==========")

print("Количество точек:", order)
print("Порядок базовой точки:", order)
print("Примитивный корень:", g)

print("\nКОЛЛИЗИИ:")

for name, data in collision_report.items():

    print("\n", name)

    print(
        "Уникальных отпечатков:",
        data["unique_fingerprints"]
    )

    print(
        "Всего точек:",
        data["total_points"]
    )

    print(
        "Групп коллизий:",
        len(data["collision_groups"])
    )


print("\nПРОВЕРКА СИММЕТРИИ:")

same_fingerprints = sum(
    row["same_fingerprint"]
    for row in symmetry_report
)

print(
    "Совпадающих отпечатков P и -P:",
    same_fingerprints,
    "из",
    len(symmetry_report)
)

print("\nФАЙЛЫ СОХРАНЕНЫ:")
print("experiment_A/character_fingerprints.csv")
print("experiment_A/symmetry.csv")
print("experiment_A/collisions.json")
print("experiment_A/report.json")
print("experiment_A.zip")

print("\nЭКСПЕРИМЕНТ ЗАВЕРШЁН")
