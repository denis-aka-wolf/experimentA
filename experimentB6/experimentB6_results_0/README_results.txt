EXPERIMENT B6 RESULTS
=================================================================

Toy elliptic curve: y^2 = x^3 + 7 mod 499
Base point: G = (3, 212)
Order of G: 73

Fingerprint definition:
T_t(P) = A_t(x_P) * exp(-2*pi*i*t*y_P/499)

This experiment studies algebraic structure of the fingerprints.
It does NOT demonstrate an ECDLP algorithm or a speedup over discrete-logarithm methods.

-----------------------------------------------------------------
FREQUENCY t=1
-----------------------------------------------------------------
Direct formula check: True
Maximum direct formula error: 4.935587579328812e-14
Antipodal y recovery: 36/36
Magnitude lookup: 34/36
Minimum magnitude separation: 0.0
Fingerprint minimum separation: 0.13176708133592083
Recurrence affine RMSE: 22.707694046830902
Recurrence mean absolute error: 19.476990620716503

-----------------------------------------------------------------
FREQUENCY t=174
-----------------------------------------------------------------
Direct formula check: True
Maximum direct formula error: 5.753312400962914e-14
Antipodal y recovery: 36/36
Magnitude lookup: 31/36
Minimum magnitude separation: 0.0
Fingerprint minimum separation: 2.452306959133808
Recurrence affine RMSE: 23.95502912777855
Recurrence mean absolute error: 20.888379031725336


INTERPRETATION NOTES
=================================================================
1. Antipodal ratio recovery uses both T(P) and T(-P).
2. Magnitude lookup is a finite-table classification test, not a general inversion algorithm.
3. The affine recurrence is an empirical fit. A low error would not by itself prove an exact identity.
4. All complex calculations use floating-point arithmetic.
5. Results apply to this toy curve and the tested frequencies only.
