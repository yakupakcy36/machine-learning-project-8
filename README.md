# Medical Insurance Cost Prediction

Small learning project. The notebook `insurance .ipynb` trains linear regression.

Data: Insurance data (insurance.csv), predicting charges from age, sex, BMI, children, smoker and region. The data files are not included in this repo, so you need to download them yourself.

## Benchmark script

`benchmark.py` compares ridge regression, random forest and HistGradientBoosting with 5-fold cross-validation.
It does not use the real dataset. It runs on synthetic data generated with numpy/scikit-learn (1338 samples, 6 features), so the numbers below only compare the models on that data and say nothing about the real dataset.

## How to run

```
pip install numpy scikit-learn jupyter
jupyter notebook
python benchmark.py
```

## Results (5-fold CV, mean)

| Model | r2 | rmse | mae |
|---|---|---|---|
| Random Forest (200 Trees) | 0.9837 | 1622.79 | 1283.85 |
| HistGradientBoosting | 0.9834 | 1635.8 | 1269.04 |
| Ridge Regression (Additive) | 0.8564 | 4813.19 | 3375.55 |

Full numbers are in `benchmarks/benchmark_results.json`.

## Files

- `insurance .ipynb`: notebook
- `benchmark.py`: benchmark script
- `benchmarks/benchmark_results.json`: results
- `assets/benchmark_chart.svg`: chart of the results
