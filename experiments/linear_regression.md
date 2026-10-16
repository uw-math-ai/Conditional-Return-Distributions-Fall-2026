# Linear regression on 100 stocks

Target: 21-day forward return minus the equal-weight universe mean.

- train: 2010-01-04 to 2018-11-28, 224,300 rows
- validation: 2019-01-02 to 2021-12-01, 73,600 rows
- test: 2022-01-03 to 2026-08-31, 116,900 rows

## Validation (fit on train)

```
           R² (%)  IC mean  IC t-stat  IC > 0 (%)  L/S return (%/yr)  L/S Sharpe  months
alpha                                                                                   
0.0         0.227    0.036      1.042      52.778             20.835       1.298    36.0
1000.0      0.231    0.040      1.094      52.778             25.082       1.399    36.0
10000.0     0.215    0.044      1.079      50.000             18.228       0.972    36.0
100000.0    0.119    0.037      0.830      55.556             20.704       0.942    36.0
1000000.0   0.021    0.036      0.773      58.333             22.340       0.953    36.0
```

Chosen alpha: 10000. Final model refit on train + validation.

## Final model

```
            R² (%)  IC mean  IC t-stat  IC > 0 (%)  L/S return (%/yr)  L/S Sharpe  months
train        0.174    0.025      0.897      57.944             12.091       0.540   107.0
validation   0.413    0.052      1.053      55.556             23.605       0.844    36.0
test         0.376    0.066      1.947      60.714             18.701       0.614    56.0
```

Train and validation rows are in-sample for the final model; only test is out-of-sample.

## Coefficients

```
                   coefficient (bp per rank unit)
drawdown_252d                             -33.809
downside_freq_63d                         -16.283
drawdown_21d                              -10.587
skew_126d                                  -8.259
drawdown_63d                               -5.303
mom_5d                                     -0.367
downside_vol_63d                            1.346
vol_63d                                     3.137
reversal_1d                                 5.082
skew_63d                                    6.749
mom_252d                                    9.648
downside_freq_21d                          10.516
downside_vol_21d                           11.909
vol_21d                                    13.786
mom_21d                                    13.931
mom_63d                                    17.684
mom_126d                                   21.351
vol_126d                                   40.865
```
