
import csv
import json
import math
import cmath
import zipfile
from pathlib import Path
from collections import defaultdict

import numpy as np


# ============================================================
# EXPERIMENT B3
#
# Curve:
#   y^2 = x^3 + 7 (mod 499)
#
# Base point:
#   G = (3, 212)
#
# Order:
#   n = 73
#
# Main fingerprint:
#   T_t(P) = A_t(x_P) * exp(-2*pi*i*t*y_P/p)
#
#   A_t(x) = sum_R chi3(u-x) * exp(2*pi*i*t*v/p)
#
# Tests:
#   - all 498 nonzero frequencies t
#   - collisions and uniqueness
#   - minimum distances between fingerprints
#   - all 36 pairs P and -P
#   - zero and near-zero A_t(x)
#   - stability under different rounding precisions
#
# Output:
#   experimentB3/
#       fingerprints.csv
#       t_scores.csv
#       at_x_diagnostics.csv
#       pair_distances_t1.csv
#       collisions.json
#       report.json
#       README_results.txt
#       experimentB3.zip
# ============================================================


# ----------------------------
# 1. PARAMETERS
# ----------------------------

P = 499
A_CURVE = 0
B_CURVE = 7

G = (3, 212)
N = 73

PRIMITIVE_ROOT = 7

OUTPUT_DIR = Path("experimentB3")
OUTPUT_DIR.mkdir(exist_ok=True)

ZIP_PATH = OUTPUT_DIR / "experimentB3.zip"

ROUNDING_DIGITS = [8, 10, 12, 14]
ZERO_TOLERANCE = 1e-12

PI2 = 2.0 * math.pi


# ----------------------------
# 2. ELLIPTIC CURVE
# ----------------------------

def inverse_mod(a, p=P):
    a %= p

    if a == 0:
        raise ZeroDivisionError("Inverse of zero")

    return pow(a, -1, p)


def is_on_curve(point):
    if point is None:
        return True

    x, y = point

    return (
        y * y - x * x * x - B_CURVE
    ) % P == 0


def point_neg(point):
    if point is None:
        return None

    x, y = point
    return (x, (-y) % P)


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

        numerator = (3 * x1 * x1 + A_CURVE) % P
        denominator = (2 * y1) % P

    else:
        numerator = (y2 - y1) % P
        denominator = (x2 - x1) % P

    slope = (
        numerator * inverse_mod(denominator)
    ) % P

    x3 = (slope * slope - x1 - x2) % P
    y3 = (slope * (x1 - x3) - y1) % P

    result = (x3, y3)

    if not is_on_curve(result):
        raise ArithmeticError(
            f"Point addition produced invalid point: {result}"
        )

    return result


def scalar_multiply(k, point):
    result = None
    current = point

    while k > 0:
        if k & 1:
            result = point_add(result, current)

        current = point_add(current, current)
        k >>= 1

    return result


# ----------------------------
# 3. BUILD THE FULL CURVE
# ----------------------------

def build_curve_points():
    points = []

    for x in range(P):
        rhs = (x * x * x + B_CURVE) % P

        for y in range(P):
            if (y * y) % P == rhs:
                points.append((x, y))

    return points


# ----------------------------
# 4. BUILD THE SUBGROUP
# ----------------------------

def build_subgroup():
    subgroup = []

    for k in range(N):
        point = scalar_multiply(k, G)
        subgroup.append((k, point))

    return subgroup


# ----------------------------
# 5. CUBIC MULTIPLICATIVE CHARACTER
# ----------------------------

def verify_primitive_root(g, p):
    factors = []

    value = p - 1
    divisor = 2

    while divisor * divisor <= value:
        if value % divisor == 0:
            factors.append(divisor)

            while value % divisor == 0:
                value //= divisor

        divisor += 1

    if value > 1:
        factors.append(value)

    for factor in factors:
        if pow(g, (p - 1) // factor, p) == 1:
            return False

    return True


def build_cubic_character():
    if not verify_primitive_root(PRIMITIVE_ROOT, P):
        raise ValueError(
            f"{PRIMITIVE_ROOT} is not a primitive root modulo {P}"
        )

    exponent = (P - 1) // 3

    omega_mod = pow(PRIMITIVE_ROOT, exponent, P)
    omega2_mod = pow(omega_mod, 2, P)

    if omega_mod == 1:
        raise ValueError("Invalid cubic root of unity")

    if pow(omega_mod, 3, P) != 1:
        raise ValueError("Cubic root verification failed")

    omega = cmath.exp(2j * math.pi / 3)
    omega2 = omega * omega

    mapping = {
        1: 1.0 + 0.0j,
        omega_mod: omega,
        omega2_mod: omega2
    }

    def chi3(z):
        z %= P

        if z == 0:
            return 0.0 + 0.0j

        result = pow(z, exponent, P)

        if result not in mapping:
            raise ArithmeticError(
                f"Unexpected cubic character value: {result}"
            )

        return mapping[result]

    return chi3, omega_mod, omega2_mod


# ----------------------------
# 6. PRECOMPUTE CHARACTER MATRIX
# ----------------------------

def build_character_matrix(chi3):
    """
    Matrix layout:

        character_matrix[x, u] = chi3(u - x)

    Rows correspond to x.
    Columns correspond to u.
    """

    matrix = np.zeros((P, P), dtype=np.complex128)

    for x in range(P):
        for u in range(P):
            matrix[x, u] = chi3(u - x)

    return matrix


# ----------------------------
# 7. GROUP CURVE POINTS BY X
# ----------------------------

def build_x_groups(curve_points):
    """
    For every u, stores all y-values such that
    (u, y) belongs to the curve.
    """

    groups = defaultdict(list)

    for x, y in curve_points:
        groups[x].append(y)

    return groups


# ----------------------------
# 8. CALCULATE A_t(x)
# ----------------------------

def calculate_A_for_t(t, x_groups, character_matrix):
    """
    A_t(x) =
        sum_(u,v) chi3(u-x) * exp(2*pi*i*t*v/P)

    First calculate:

        S_t(u) =
            sum_(v: (u,v) on curve) exp(2*pi*i*t*v/P)

    Then:

        A_t(x) = sum_u chi3(u-x) * S_t(u)

    The vector S_t is real up to floating-point noise,
    because points (u,v) and (u,-v) occur in pairs.
    """

    s_real = np.zeros(P, dtype=np.float64)
    s_imag = np.zeros(P, dtype=np.float64)

    for u, y_values in x_groups.items():
        angles = PI2 * t * np.array(y_values) / P

        s_real[u] = np.sum(np.cos(angles))
        s_imag[u] = np.sum(np.sin(angles))

    s_vector = s_real + 1j * s_imag

    # A_t(x) for all x in the field.
    result = np.sum(
        character_matrix * s_vector[np.newaxis, :],
        axis=1
    )

    return result


# ----------------------------
# 9. FINGERPRINT FUNCTIONS
# ----------------------------

def calculate_fingerprint(A_values, t, point):
    if point is None:
        return None

    x, y = point

    phase = cmath.exp(
        -2j * math.pi * t * y / P
    )

    return complex(A_values[x] * phase)


def fingerprint_key(value, digits):
    return (
        round(value.real, digits),
        round(value.imag, digits)
    )


def find_collision_groups(values, digits):
    """
    values:
        list of (k, point, fingerprint)

    Returns groups with more than one point
    having the same rounded fingerprint.
    """

    groups = defaultdict(list)

    for k, point, value in values:
        key = fingerprint_key(value, digits)
        groups[str(key)].append({
            "k": int(k),
            "point": list(point),
            "real": float(value.real),
            "imag": float(value.imag)
        })

    collisions = []

    for key, group in groups.items():
        if len(group) > 1:
            collisions.append({
                "rounded_key": key,
                "points": group
            })

    return collisions


# ----------------------------
# 10. DISTANCE CALCULATIONS
# ----------------------------

def minimum_pairwise_distance(values):
    """
    Minimum Euclidean distance between all distinct
    entries in a complex-valued vector.
    """

    values = np.asarray(values, dtype=np.complex128)

    if len(values) < 2:
        return None

    distances = np.abs(
        values[:, np.newaxis] - values[np.newaxis, :]
    )

    np.fill_diagonal(distances, np.inf)

    return float(np.min(distances))


def opposite_pair_distances(subgroup, fingerprints):
    """
    Compare P=kG and -P=(73-k)G.

    k = 1..36
    """

    rows = []

    for k in range(1, N // 2 + 1):
        opposite_k = N - k

        value1 = fingerprints[k]
        value2 = fingerprints[opposite_k]

        distance = abs(value1 - value2)

        rows.append({
            "k": k,
            "opposite_k": opposite_k,
            "point_k": subgroup[k][1],
            "point_opposite": subgroup[opposite_k][1],
            "distance": float(distance)
        })

    return rows


# ----------------------------
# 11. MAIN EXPERIMENT
# ----------------------------

def main():

    print("=" * 60)
    print("EXPERIMENT B3")
    print("=" * 60)

    print("\nBuilding curve...")

    curve_points = build_curve_points()

    print("Affine curve points:", len(curve_points))

    if len(curve_points) != 510:
        raise AssertionError(
            f"Expected 510 affine points, got {len(curve_points)}"
        )

    print("\nChecking base point...")

    if not is_on_curve(G):
        raise AssertionError("G is not on the curve")

    subgroup = build_subgroup()

    if subgroup[0][1] is not None:
        raise AssertionError("0G must be the point at infinity")

    if subgroup[N - 1][1] is None:
        raise AssertionError("Unexpected infinity point")

    if scalar_multiply(N, G) is not None:
        raise AssertionError("G does not have order 73")

    if scalar_multiply(N - 1, G) == G:
        raise AssertionError("Unexpected subgroup cycle")

    subgroup_points = [point for k, point in subgroup]

    if len(set(subgroup_points)) != N:
        raise AssertionError("Subgroup contains duplicate points")

    print("Order of G:", N)
    print("Subgroup points:", len(subgroup))

    check_371 = scalar_multiply(371, G)

    print("371G:", check_371)

    if check_371 != (449, 483):
        raise AssertionError("371G verification failed")

    # Verify that all subgroup points are on the curve.
    for k, point in subgroup:
        if not is_on_curve(point):
            raise AssertionError(
                f"Invalid subgroup point at k={k}: {point}"
            )

    print("\nBuilding cubic character...")

    chi3, omega_mod, omega2_mod = build_cubic_character()

    print("Primitive root:", PRIMITIVE_ROOT)
    print("omega mod p:", omega_mod)
    print("omega^2 mod p:", omega2_mod)

    character_matrix = build_character_matrix(chi3)

    x_groups = build_x_groups(curve_points)

    # The subgroup has 36 distinct finite x-coordinates.
    subgroup_x_values = sorted({
        point[0]
        for point in subgroup_points
        if point is not None
    })

    print("\nDistinct subgroup x-values:", len(subgroup_x_values))

    if len(subgroup_x_values) != 36:
        raise AssertionError(
            "Expected 36 distinct finite x-coordinates"
        )

    # --------------------------------------------------------
    # Output collections
    # --------------------------------------------------------

    fingerprint_rows = []
    score_rows = []
    diagnostic_rows = []

    all_collisions = {}

    best_t = None
    best_unique = -1
    best_min_distance = -1.0

    t1_fingerprints = None
    t1_A_values = None

    # --------------------------------------------------------
    # Test every nonzero frequency
    # --------------------------------------------------------

    print("\nTesting all 498 nonzero frequencies...")

    for t in range(1, P):

        if t % 25 == 0 or t == 1 or t == P - 1:
            print(f"  Processing t={t}/{P - 1}")

        A_values = calculate_A_for_t(
            t,
            x_groups,
            character_matrix
        )

        # Fingerprints for the 73 subgroup points.
        fingerprints = [None] * N

        finite_values = []

        for k, point in subgroup:

            if point is None:
                continue

            value = calculate_fingerprint(
                A_values,
                t,
                point
            )

            fingerprints[k] = value

            finite_values.append(value)

            fingerprint_rows.append({
                "t": t,
                "k": k,
                "x": point[0],
                "y": point[1],
                "real": float(value.real),
                "imag": float(value.imag),
                "abs": float(abs(value))
            })

        # The point at infinity gets a special fingerprint.
        # It is not treated as a complex number.
        #
        # This is important: its separation is structural,
        # not a numerical result.

        # Unique counts under several rounding precisions.
        unique_counts = {}

        for digits in ROUNDING_DIGITS:
            keys = {
                fingerprint_key(value, digits)
                for value in finite_values
            }

            # Add one separate identifier for infinity.
            unique_counts[digits] = len(keys) + 1

        # Minimum distance between the 72 finite points.
        min_distance = minimum_pairwise_distance(
            finite_values
        )

        # Opposite point distances.
        pair_rows = opposite_pair_distances(
            subgroup,
            fingerprints
        )

        min_opposite_distance = min(
            row["distance"] for row in pair_rows
        )

        # A_t(x) diagnostics for all distinct x in subgroup.
        subgroup_A = [
            A_values[x]
            for x in subgroup_x_values
        ]

        A_abs_values = np.abs(
            np.asarray(subgroup_A, dtype=np.complex128)
        )

        min_A_abs = float(np.min(A_abs_values))
        max_A_abs = float(np.max(A_abs_values))

        near_zero_count = int(
            np.sum(A_abs_values <= ZERO_TOLERANCE)
        )

        # Record A_t(x) diagnostics.
        for x in subgroup_x_values:
            value = complex(A_values[x])

            diagnostic_rows.append({
                "t": t,
                "x": x,
                "A_real": float(value.real),
                "A_imag": float(value.imag),
                "A_abs": float(abs(value)),
                "near_zero": (
                    abs(value) <= ZERO_TOLERANCE
                )
            })

        # Collision groups at 10 decimal places.
        collision_groups = find_collision_groups(
            [
                (k, point, fingerprints[k])
                for k, point in subgroup
                if point is not None
            ],
            10
        )

        if collision_groups:
            all_collisions[str(t)] = collision_groups

        score_row = {
            "t": t,
            "unique_round_8": unique_counts[8],
            "unique_round_10": unique_counts[10],
            "unique_round_12": unique_counts[12],
            "unique_round_14": unique_counts[14],
            "collisions_round_10": len(collision_groups),
            "min_distance": min_distance,
            "min_opposite_distance": min_opposite_distance,
            "min_A_abs": min_A_abs,
            "max_A_abs": max_A_abs,
            "near_zero_A_count": near_zero_count
        }

        score_rows.append(score_row)

        # Select best t:
        # 1. Maximum uniqueness at 10 decimal places.
        # 2. Maximum minimum pairwise distance.
        unique_10 = unique_counts[10]

        if (
            unique_10 > best_unique
            or (
                unique_10 == best_unique
                and min_distance > best_min_distance
            )
        ):
            best_t = t
            best_unique = unique_10
            best_min_distance = min_distance

        if t == 1:
            t1_fingerprints = fingerprints
            t1_A_values = A_values.copy()

    # --------------------------------------------------------
    # Pair distances for t=1
    # --------------------------------------------------------

    pair_distance_rows = opposite_pair_distances(
        subgroup,
        t1_fingerprints
    )

    # --------------------------------------------------------
    # Global results
    # --------------------------------------------------------

    score_by_t = {
        row["t"]: row
        for row in score_rows
    }

    best_row = score_by_t[best_t]
    t1_row = score_by_t[1]

    # Check whether all 498 frequencies were actually tested.
    if len(score_rows) != P - 1:
        raise AssertionError("Not all 498 frequencies were tested")

    # --------------------------------------------------------
    # CSV: fingerprints.csv
    # --------------------------------------------------------

    fingerprints_path = OUTPUT_DIR / "fingerprints.csv"

    with open(
        fingerprints_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        fieldnames = [
            "t",
            "k",
            "x",
            "y",
            "real",
            "imag",
            "abs"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(fingerprint_rows)

    # --------------------------------------------------------
    # CSV: t_scores.csv
    # --------------------------------------------------------

    scores_path = OUTPUT_DIR / "t_scores.csv"

    with open(
        scores_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        fieldnames = list(score_rows[0].keys())

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(score_rows)

    # --------------------------------------------------------
    # CSV: at_x_diagnostics.csv
    # --------------------------------------------------------

    diagnostics_path = OUTPUT_DIR / "at_x_diagnostics.csv"

    with open(
        diagnostics_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        fieldnames = list(diagnostic_rows[0].keys())

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(diagnostic_rows)

    # --------------------------------------------------------
    # CSV: pair_distances_t1.csv
    # --------------------------------------------------------

    pair_path = OUTPUT_DIR / "pair_distances_t1.csv"

    with open(
        pair_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        fieldnames = [
            "k",
            "opposite_k",
            "point_k",
            "point_opposite",
            "distance"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in pair_distance_rows:
            writer.writerow({
                "k": row["k"],
                "opposite_k": row["opposite_k"],
                "point_k": row["point_k"],
                "point_opposite": row["point_opposite"],
                "distance": row["distance"]
            })

    # --------------------------------------------------------
    # JSON: collisions.json
    # --------------------------------------------------------

    collisions_path = OUTPUT_DIR / "collisions.json"

    with open(
        collisions_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_collisions,
            file,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # JSON: report.json
    # --------------------------------------------------------

    report = {
        "experiment": "B3",
        "curve": {
            "equation": "y^2 = x^3 + 7 mod 499",
            "p": P,
            "a": A_CURVE,
            "b": B_CURVE,
            "affine_points": len(curve_points),
            "total_points_including_infinity": (
                len(curve_points) + 1
            )
        },
        "subgroup": {
            "base_point": list(G),
            "order": N,
            "point_count": len(subgroup),
            "distinct_finite_x_values": len(subgroup_x_values),
            "check_371G": list(check_371)
        },
        "character": {
            "type": "cubic multiplicative character",
            "primitive_root": PRIMITIVE_ROOT,
            "omega_mod_p": omega_mod,
            "omega2_mod_p": omega2_mod
        },
        "fingerprint": {
            "formula": (
                "T_t(P) = A_t(x_P) * "
                "exp(-2*pi*i*t*y_P/p)"
            ),
            "A_formula": (
                "A_t(x) = sum_R chi3(u-x) * "
                "exp(2*pi*i*t*v/p)"
            ),
            "curve_sum": (
                "All 510 affine points; "
                "point at infinity handled separately"
            )
        },
        "frequency_test": {
            "first_t": 1,
            "last_t": P - 1,
            "tested_frequencies": len(score_rows),
            "all_nonzero_frequencies_tested": (
                len(score_rows) == P - 1
            )
        },
        "rounding_digits": ROUNDING_DIGITS,
        "zero_tolerance": ZERO_TOLERANCE,
        "t1": {
            "unique_round_8": t1_row["unique_round_8"],
            "unique_round_10": t1_row["unique_round_10"],
            "unique_round_12": t1_row["unique_round_12"],
            "unique_round_14": t1_row["unique_round_14"],
            "collisions_round_10": (
                t1_row["collisions_round_10"]
            ),
            "min_pairwise_distance": (
                t1_row["min_distance"]
            ),
            "min_opposite_pair_distance": (
                t1_row["min_opposite_distance"]
            ),
            "min_A_abs": t1_row["min_A_abs"],
            "near_zero_A_count": (
                t1_row["near_zero_A_count"]
            )
        },
        "best_t": {
            "t": best_t,
            "unique_round_8": best_row["unique_round_8"],
            "unique_round_10": best_row["unique_round_10"],
            "unique_round_12": best_row["unique_round_12"],
            "unique_round_14": best_row["unique_round_14"],
            "collisions_round_10": (
                best_row["collisions_round_10"]
            ),
            "min_pairwise_distance": (
                best_row["min_distance"]
            ),
            "min_opposite_pair_distance": (
                best_row["min_opposite_distance"]
            ),
            "min_A_abs": best_row["min_A_abs"],
            "near_zero_A_count": (
                best_row["near_zero_A_count"]
            )
        },
        "opposite_pairs_t1": {
            "total_pairs": len(pair_distance_rows),
            "distinct_pairs": sum(
                1 for row in pair_distance_rows
                if row["distance"] > 0
            ),
            "minimum_distance": min(
                row["distance"]
                for row in pair_distance_rows
            ),
            "maximum_distance": max(
                row["distance"]
                for row in pair_distance_rows
            )
        },
        "collision_summary": {
            "frequencies_with_collisions": len(all_collisions),
            "total_collision_groups": sum(
                len(groups)
                for groups in all_collisions.values()
            )
        },
        "files": [
            "fingerprints.csv",
            "t_scores.csv",
            "at_x_diagnostics.csv",
            "pair_distances_t1.csv",
            "collisions.json",
            "report.json",
            "README_results.txt"
        ],
        "important_limitation": (
            "Injectivity on this finite subgroup does not "
            "demonstrate an algorithm for solving ECDLP. "
            "The fingerprint is computed from a known point "
            "and still requires a separate method for recovering k."
        )
    }

    report_path = OUTPUT_DIR / "report.json"

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # README
    # --------------------------------------------------------

    readme_path = OUTPUT_DIR / "README_results.txt"

    readme = f"""
EXPERIMENT B3
=============

Curve:
    y^2 = x^3 + 7 (mod 499)

Base point:
    G = (3, 212)

Order:
    73

Affine points on full curve:
    {len(curve_points)}

Subgroup points:
    {len(subgroup)}

Frequencies tested:
    1 through 498
    Total: {len(score_rows)}

RESULT FOR t=1
--------------

Unique fingerprints at 8 decimal places:
    {t1_row['unique_round_8']}

Unique fingerprints at 10 decimal places:
    {t1_row['unique_round_10']}

Unique fingerprints at 12 decimal places:
    {t1_row['unique_round_12']}

Unique fingerprints at 14 decimal places:
    {t1_row['unique_round_14']}

Collision groups at 10 decimal places:
    {t1_row['collisions_round_10']}

Minimum pairwise distance:
    {t1_row['min_distance']:.16e}

Minimum distance between P and -P:
    {t1_row['min_opposite_distance']:.16e}

Minimum absolute value of A_t(x):
    {t1_row['min_A_abs']:.16e}

Near-zero A_t(x) count:
    {t1_row['near_zero_A_count']}


BEST FREQUENCY
--------------

t:
    {best_t}

Unique fingerprints at 10 decimal places:
    {best_row['unique_round_10']}

Collision groups:
    {best_row['collisions_round_10']}

Minimum pairwise distance:
    {best_row['min_distance']:.16e}


OPPOSITE POINTS
---------------

Pairs tested:
    {len(pair_distance_rows)}

Pairs with nonzero distance:
    {sum(1 for row in pair_distance_rows if row['distance'] > 0)}

Minimum distance:
    {min(row['distance'] for row in pair_distance_rows):.16e}


FILES
-----

fingerprints.csv
    All fingerprints for all 498 frequencies and 73 subgroup points.

t_scores.csv
    Summary statistics for every frequency.

at_x_diagnostics.csv
    Values and absolute values of A_t(x) for subgroup x-coordinates.

pair_distances_t1.csv
    Distances for all 36 opposite-point pairs at t=1.

collisions.json
    Collision groups at 10 decimal places.

report.json
    Machine-readable experiment report.


IMPORTANT
---------

All 498 nonzero frequencies were tested independently.

No assumption was made that t and p-t produce complex-conjugate
fingerprints. The cubic character is complex-valued, so that
conjugacy cannot be assumed without additional analysis.

Numerical uniqueness is tested at several rounding precisions.
The minimum pairwise distance is also reported.

This experiment tests fingerprint separation on one small
subgroup. It does not by itself provide a method for recovering
the discrete logarithm k from a public point Q = kG.
"""

    with open(
        readme_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(readme.strip())

    # --------------------------------------------------------
    # ZIP ARCHIVE
    # --------------------------------------------------------

    files_to_archive = [
        fingerprints_path,
        scores_path,
        diagnostics_path,
        pair_path,
        collisions_path,
        report_path,
        readme_path
    ]

    with zipfile.ZipFile(
        ZIP_PATH,
        "w",
        compression=zipfile.ZIP_DEFLATED
    ) as archive:

        for path in files_to_archive:
            archive.write(
                path,
                arcname=path.name
            )

    # --------------------------------------------------------
    # FINAL CONSOLE REPORT
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("EXPERIMENT B3 RESULTS")
    print("=" * 60)

    print("\nFrequencies tested:", len(score_rows))

    print("\nRESULT FOR t=1")

    print(
        "Unique at 8 decimals:",
        t1_row["unique_round_8"]
    )

    print(
        "Unique at 10 decimals:",
        t1_row["unique_round_10"]
    )

    print(
        "Unique at 12 decimals:",
        t1_row["unique_round_12"]
    )

    print(
        "Unique at 14 decimals:",
        t1_row["unique_round_14"]
    )

    print(
        "Collision groups:",
        t1_row["collisions_round_10"]
    )

    print(
        "Minimum pairwise distance:",
        f"{t1_row['min_distance']:.16e}"
    )

    print(
        "Minimum P/-P distance:",
        f"{t1_row['min_opposite_distance']:.16e}"
    )

    print(
        "Minimum |A_t(x)|:",
        f"{t1_row['min_A_abs']:.16e}"
    )

    print(
        "Near-zero A_t(x) count:",
        t1_row["near_zero_A_count"]
    )

    print("\nBEST FREQUENCY")

    print("t:", best_t)

    print(
        "Unique fingerprints:",
        best_row["unique_round_10"]
    )

    print(
        "Minimum pairwise distance:",
        f"{best_row['min_distance']:.16e}"
    )

    print(
        "Collision groups:",
        best_row["collisions_round_10"]
    )

    print("\nALL FREQUENCIES")

    no_collision_count = sum(
        1
        for row in score_rows
        if row["collisions_round_10"] == 0
    )

    print(
        "Frequencies without collisions:",
        no_collision_count,
        "/",
        len(score_rows)
    )

    print(
        "Frequencies with collisions:",
        len(all_collisions)
    )

    print("\nArchive created:")
    print(ZIP_PATH.resolve())

    print("\nExperiment completed.")


if __name__ == "__main__":
    main()
