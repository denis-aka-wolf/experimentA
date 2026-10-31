EXPERIMENT B3
=============

Curve:
    y^2 = x^3 + 7 (mod 499)

Base point:
    G = (3, 212)

Order:
    73

Affine points on full curve:
    510

Subgroup points:
    73

Frequencies tested:
    1 through 498
    Total: 498

RESULT FOR t=1
--------------

Unique fingerprints at 8 decimal places:
    73

Unique fingerprints at 10 decimal places:
    73

Unique fingerprints at 12 decimal places:
    73

Unique fingerprints at 14 decimal places:
    73

Collision groups at 10 decimal places:
    0

Minimum pairwise distance:
    1.3176708133592083e-01

Minimum distance between P and -P:
    1.3176708133592083e-01

Minimum absolute value of A_t(x):
    4.6224930589073390e+00

Near-zero A_t(x) count:
    0


BEST FREQUENCY
--------------

t:
    174

Unique fingerprints at 10 decimal places:
    73

Collision groups:
    0

Minimum pairwise distance:
    2.4523069591345337e+00


OPPOSITE POINTS
---------------

Pairs tested:
    36

Pairs with nonzero distance:
    36

Minimum distance:
    1.3176708133592083e-01


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