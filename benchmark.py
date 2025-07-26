#!/usr/bin/env python3
"""
Benchmark Suite: Actuarial Medical Insurance Premium Forecaster
Evaluates Ridge Regression (Linear Baseline), Random Forest Regressor, and
HistGradientBoosting Regressor on actuarial policyholder risk factors (Age, BMI,
Smoker Status, Dependents, Sex, Region) using 5-Fold Cross-Validation.
"""

import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def generate_actuarial_insurance_cohort(n_samples: int = 1338, random_state: int = 42):
    """
    Synthesizes a 1,338-policyholder actuarial medical charges dataset calibrated to the
    US Medical Cost Personal Dataset, including the non-linear smoker x obesity (BMI >= 30)
    synergistic hazard jump.
    """
    rng = np.random.default_rng(random_state)

    age = rng.integers(18, 65, size=n_samples).astype(float)
    sex = rng.binomial(1, 0.505, size=n_samples).astype(float)
    bmi = np.clip(rng.normal(30.66, 6.10, size=n_samples), 16.0, 53.1)
    children = rng.choice([0, 1, 2, 3, 4, 5], size=n_samples, p=[0.429, 0.242, 0.179, 0.117, 0.019, 0.014]).astype(float)
    smoker = rng.binomial(1, 0.205, size=n_samples).astype(float)
    region = rng.integers(0, 4, size=n_samples).astype(float)

    # Actuarial cost function with non-linear age progression and smoker x BMI>=30 interaction
    obese_indicator = (bmi >= 30.0).astype(float)
    base_charge = 2150.0 + 258.5 * age + 1.85 * (age ** 2) + 495.0 * children - 120.0 * region
    smoker_surcharge = smoker * (14800.0 + 19500.0 * obese_indicator + 440.0 * (bmi - 25.0))
    bmi_surcharge = (1.0 - smoker) * np.maximum(0.0, bmi - 25.0) * 145.0
    idiosyncratic_noise = rng.normal(0.0, 1450.0, size=n_samples)

    charges = np.clip(base_charge + smoker_surcharge + bmi_surcharge + idiosyncratic_noise, 1120.0, None)
    X = np.column_stack([age, sex, bmi, children, smoker, region])
    feature_names = ["age", "sex_male", "bmi", "children_dependents", "smoker_status", "region_code"]
    return X, charges, feature_names


def render_svg_chart(results: list, output_path: Path):
    """Generates a dark-mode vector performance chart for actuarial premium forecasting."""
    width, height = 860, 420
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">',
        '  <defs>',
        '    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">',
        '      <stop offset="0%" stop-color="#0b132b"/>',
        '      <stop offset="100%" stop-color="#1e293b"/>',
        '    </linearGradient>',
        '  </defs>',
        f'  <rect width="{width}" height="{height}" rx="12" fill="url(#bgGrad)" stroke="#334155" stroke-width="1.5"/>',
        '  <text x="32" y="42" fill="#f8fafc" font-family="Segoe UI, Inter, sans-serif" font-size="20" font-weight="700">'
        'Actuarial Medical Insurance Premium Forecaster — 5-Fold CV</text>',
        '  <text x="32" y="66" fill="#94a3b8" font-family="Segoe UI, Inter, sans-serif" font-size="13">'
        'Policyholder Cohort (N = 1,338, d = 6 Actuarial Factors) | Capturing Smoker × Obesity Interaction</text>',
        '  <rect x="32" y="84" width="12" height="12" rx="3" fill="#38bdf8"/>',
        '  <text x="50" y="94" fill="#cbd5e1" font-family="Segoe UI, Inter, sans-serif" font-size="12" font-weight="600">R² Variance Explained</text>',
        '  <rect x="235" y="84" width="12" height="12" rx="3" fill="#34d399"/>',
        '  <text x="253" y="94" fill="#cbd5e1" font-family="Segoe UI, Inter, sans-serif" font-size="12" font-weight="600">RMSE ($ USD — Lower is Better)</text>',
        '  <rect x="475" y="84" width="12" height="12" rx="3" fill="#a78bfa"/>',
        '  <text x="493" y="94" fill="#cbd5e1" font-family="Segoe UI, Inter, sans-serif" font-size="12" font-weight="600">MAE ($ USD — Lower is Better)</text>',
    ]

    chart_left = 250
    chart_right = 785
    chart_width = chart_right - chart_left
    top_y = 124
    group_height = 90

    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        gx = chart_left + tick * chart_width
        svg.append(
            f'  <line x1="{gx:.1f}" y1="{top_y - 8}" x2="{gx:.1f}" y2="{top_y + 3 * group_height - 12}" '
            f'stroke="#334155" stroke-dasharray="4,4" stroke-width="1"/>'
        )
        svg.append(
            f'  <text x="{gx:.1f}" y="{top_y + 3 * group_height + 10}" fill="#64748b" '
            f'font-family="Segoe UI, Inter, sans-serif" font-size="11" text-anchor="middle">{tick:.2f}</text>'
        )

    for idx, entry in enumerate(results):
        gy = top_y + idx * group_height
        model_name = entry["model"]
        svg.append(
            f'  <text x="{chart_left - 14}" y="{gy + 32}" fill="#f1f5f9" '
            f'font-family="Segoe UI, Inter, sans-serif" font-size="13" font-weight="600" text-anchor="end">{model_name}</text>'
        )
        r2_w = max(2.0, entry["r2"] * chart_width)
        svg.append(f'  <rect x="{chart_left}" y="{gy}" width="{r2_w:.1f}" height="16" rx="3" fill="#38bdf8"/>')
        svg.append(
            f'  <text x="{chart_left + r2_w + 8:.1f}" y="{gy + 12}" fill="#e2e8f0" '
            f'font-family="Consolas, monospace" font-size="11" font-weight="600">R² = {entry["r2"]:.4f}</text>'
        )

        rmse_w = max(2.0, min(1.0, entry["rmse"] / 6500.0) * chart_width)
        svg.append(f'  <rect x="{chart_left}" y="{gy + 21}" width="{rmse_w:.1f}" height="16" rx="3" fill="#34d399"/>')
        svg.append(
            f'  <text x="{chart_left + rmse_w + 8:.1f}" y="{gy + 33}" fill="#e2e8f0" '
            f'font-family="Consolas, monospace" font-size="11" font-weight="600">RMSE = ${entry["rmse"]:,.1f}</text>'
        )

        mae_w = max(2.0, min(1.0, entry["mae"] / 6500.0) * chart_width)
        svg.append(f'  <rect x="{chart_left}" y="{gy + 42}" width="{mae_w:.1f}" height="16" rx="3" fill="#a78bfa"/>')
        svg.append(
            f'  <text x="{chart_left + mae_w + 8:.1f}" y="{gy + 54}" fill="#e2e8f0" '
            f'font-family="Consolas, monospace" font-size="11" font-weight="600">MAE = ${entry["mae"]:,.1f}</text>'
        )

    svg.append('</svg>')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg), encoding="utf-8")


def main():
    X, y, feature_names = generate_actuarial_insurance_cohort()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=180, learning_rate=0.05, max_depth=5, random_state=42
        ),
        "Random Forest (200 Trees)": RandomForestRegressor(
            n_estimators=200, max_depth=9, min_samples_leaf=4, random_state=42
        ),
        "Ridge Regression (Additive)": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0, random_state=42)),
        ]),
    }

    benchmark_rows = []
    for name, model in models.items():
        oof_preds = np.zeros(len(y))
        for train_idx, val_idx in cv.split(X):
            model.fit(X[train_idx], y[train_idx])
            oof_preds[val_idx] = model.predict(X[val_idx])

        rmse = float(np.sqrt(mean_squared_error(y, oof_preds)))
        mae = float(mean_absolute_error(y, oof_preds))
        r2 = float(r2_score(y, oof_preds))

        benchmark_rows.append({
            "model": name,
            "r2": round(r2, 4),
            "rmse": round(rmse, 2),
            "mae": round(mae, 2),
        })

    benchmark_rows.sort(key=lambda r: r["r2"], reverse=True)

    payload = {
        "project": "Actuarial Medical Insurance Premium Forecaster",
        "dataset": "US Medical Cost Actuarial Cohort (N=1,338, d=6)",
        "validation_scheme": "5-Fold Cross-Validation",
        "features": feature_names,
        "results": benchmark_rows,
    }

    base_dir = Path(__file__).resolve().parent
    json_path = base_dir / "benchmarks" / "benchmark_results.json"
    svg_path = base_dir / "assets" / "benchmark_chart.svg"

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    render_svg_chart(benchmark_rows, svg_path)
    print(f"Saved benchmark results to {json_path}")
    print(f"Saved benchmark chart to {svg_path}")


if __name__ == "__main__":
    main()
