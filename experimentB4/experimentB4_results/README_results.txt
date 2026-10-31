
EXPERIMENT B4

Curve:
    y^2 = x^3 + 7 (mod 499)

Base point:
    G = (3, 212)

Order:
    73

Control:
    371G = (449, 483)

Frequencies:
    (174, 1)


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
