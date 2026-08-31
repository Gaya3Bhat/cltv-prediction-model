"""
VahanBima CLTV Prediction
==========================
Problem:
    VahanBima wants to segment customers into value tiers so it can
    offer personalised services (dedicated claim handlers, doorstep
    support, etc.). The segmentation is driven by Customer Lifetime
    Value (CLTV), which we predict from policy and demographic data.

Dataset (actual, verified against the shipped files):
    Train — 89,392 rows, 12 columns (10 features + id + cltv)
    Test  — 59,595 rows, 11 columns (10 features + id)
    cltv  — mean $97,953, std $90,614, min $24,828, max $724,068
            (right-skewed with a long upper tail)

Evaluation metric:
    R² score (coefficient of determination).

Model choice — CatBoostRegressor:
    The dataset has 7 categorical columns. CatBoost handles them
    natively using Ordered Target Statistics (OrderedTS), a built-in,
    leak-free target encoding applied at training time. This removes
    the need for manual label/frequency/target encoding.

    Note: the "CatBoost outperformed XGBoost/LightGBM on this dataset"
    claim in earlier drafts of this project was not independently
    re-verified in the run this repo's numbers come from — treat it
    as the original author's note, not a benchmarked result.

Overall approach:
    1. Engineer 35 numeric/interaction features on top of the 10 raw
       columns.
    2. Run 5-fold cross-validation across 3 random seeds to find the
       optimal number of boosting iterations for full-data training.
    3. Train the final model on 100% of the training data (no rows
       held back) using the calibrated iteration count.
    4. Average predictions from the 3 independently seeded models to
       reduce variance on the test set.

Verified results (this exact script, run end-to-end on the full
89,392-row train / 59,595-row test files):
    Multi-seed OOF R²   : 0.16078
    Multi-seed OOF RMSE : $83,009.67
    Multi-seed OOF MAE  : $50,424.43
    Full-data iterations: {42: 167, 7: 252, 13: 195}

Usage:
    python cltv_model.py \
        --train data/CLTV_TRAINDATA.xlsx \
        --test  data/CLTV_TESTDATA.csv \
        --out   submission.csv

    Full-size files are NOT committed to this repo (see README for
    why + where to get them). A 200-row sample of each is committed
    at data/sample/ so the script can be smoke-tested end-to-end
    without the full dataset:

    python cltv_model.py \
        --train data/sample/train_sample.csv \
        --test  data/sample/test_sample.csv \
        --out   submission_sample.csv \
        --quick
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

CAT_COLS = [
    "gender", "area", "qualification", "income",
    "num_policies", "policy", "type_of_policy",
]
SEEDS = [42, 7, 13]  # 3 seeds - speed/stability tradeoff found via experimentation


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds 35 features from the 10 raw columns.

    CatBoost learns categorical patterns well but tree splits are
    axis-aligned, so it cannot directly represent e.g. "Urban AND
    multi-policy" without either a very deep tree or an explicit
    product feature. Pre-computing these products gives the model a
    direct handle on the most predictive customer segments.

    Applied identically to train and test - no target information is
    used, so there is no leakage.
    """
    df = df.copy()

    # Ordinal encodings: income and type_of_policy have a natural
    # low -> high ordering, which a plain string category loses.
    df["income_ord"] = df["income"].map(
        {"<=2L": 0, "2L-5L": 1, "5L-10L": 2, "More than 10L": 3}
    )
    df["policy_tier_ord"] = df["type_of_policy"].map(
        {"Silver": 0, "Gold": 1, "Platinum": 2}
    )

    # Binary flags.
    df["num_policies_bin"] = (df["num_policies"] == "More than 1").astype(int)
    df["area_urban"] = (df["area"] == "Urban").astype(int)
    df["has_claim"] = (df["claim_amount"] > 0).astype(int)

    # Log transform to compress the long right tail of claim_amount.
    df["log_claim"] = np.log1p(df["claim_amount"])

    # Pairwise interactions with the strongest single predictor,
    # num_policies_bin (21.6% of CatBoost's importance - see README).
    df["claim_x_num"] = df["claim_amount"] * df["num_policies_bin"]
    df["log_claim_x_num"] = df["log_claim"] * df["num_policies_bin"]
    df["area_x_num"] = df["area_urban"] * df["num_policies_bin"]
    df["marital_x_num"] = df["marital_status"] * df["num_policies_bin"]
    df["vintage_x_num"] = df["vintage"] * df["num_policies_bin"]
    df["income_x_num"] = df["income_ord"] * df["num_policies_bin"]
    df["claim_x_area"] = df["claim_amount"] * df["area_urban"]
    df["income_x_area"] = df["income_ord"] * df["area_urban"]

    # 3-way interactions for the Urban + multi-policy segment.
    df["mar_x_area_x_num"] = df["marital_status"] * df["area_urban"] * df["num_policies_bin"]
    df["inc_x_area_x_num"] = df["income_ord"] * df["area_urban"] * df["num_policies_bin"]
    df["cl_x_area_x_num"] = df["claim_amount"] * df["area_urban"] * df["num_policies_bin"]

    # Ratio features normalising claim behaviour by tenure/policy count.
    df["claim_per_vint"] = df["claim_amount"] / (df["vintage"] + 1)
    df["claim_per_pol"] = df["claim_amount"] / (df["num_policies_bin"] + 1)
    df["log_cl_per_vint"] = df["log_claim"] / (df["vintage"] + 1)

    # Polynomial / cross-product features.
    df["vintage_sq"] = df["vintage"] ** 2
    df["income_ord_sq"] = df["income_ord"] ** 2
    df["claim_vintage"] = df["claim_amount"] * df["vintage"]
    df["income_x_vint"] = df["income_ord"] * df["vintage"]
    df["income_x_claim"] = df["income_ord"] * df["log_claim"]

    return df


def cb_params(quick: bool) -> dict:
    return dict(
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=5,
        bagging_temperature=2,
        random_strength=2,
        border_count=128,
        loss_function="RMSE",
        eval_metric="RMSE",
        early_stopping_rounds=50,
        verbose=0,
        thread_count=4,
        cat_features=CAT_COLS,
    )


def run(train_path: str, test_path: str, out_path: str, results_dir: str, quick: bool):
    print("=" * 60)
    print("VahanBima CLTV Prediction")
    print("=" * 60)

    print("\n[1] Loading data...")
    train = pd.read_excel(train_path) if train_path.endswith((".xlsx", ".xls")) else pd.read_csv(train_path)
    test = pd.read_excel(test_path) if test_path.endswith((".xlsx", ".xls")) else pd.read_csv(test_path)
    y = train["cltv"].copy()
    print(f"    Train: {train.shape} | Test: {test.shape}")

    print("\n[2] Feature engineering...")
    X_tr = make_features(train.drop(columns=["id", "cltv"]))
    X_te = make_features(test.drop(columns=["id"]))
    print(f"    Features: {X_tr.shape[1]}")

    print("\n[3] 5-fold CV to find best iterations per seed...")
    print("-" * 60)
    params = cb_params(quick)
    max_iterations = 300 if quick else 3000
    n_splits = 3 if quick else 5

    seed_best_iters = {}
    oof_all = np.zeros(len(X_tr))
    oof_counts = np.zeros(len(X_tr))

    for seed in SEEDS:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        oof = np.zeros(len(X_tr))
        fold_iters = []

        for fold, (tri, vali) in enumerate(kf.split(X_tr)):
            m = CatBoostRegressor(iterations=max_iterations, random_seed=seed, **params)
            m.fit(X_tr.iloc[tri], y.iloc[tri], eval_set=(X_tr.iloc[vali], y.iloc[vali]))
            oof[vali] = m.predict(X_tr.iloc[vali])
            fold_iters.append(m.best_iteration_)
            print(
                f"    seed={seed} fold={fold + 1} "
                f"val_R2={r2_score(y.iloc[vali], oof[vali]):.5f} "
                f"iter={m.best_iteration_}"
            )

        seed_r2 = r2_score(y, oof)
        best_iter = max(1, int(np.mean(fold_iters) * 1.10))
        seed_best_iters[seed] = best_iter
        oof_all += oof
        oof_counts += 1
        print(f"    seed={seed} OOF R2={seed_r2:.5f}  -> full-data iters={best_iter}\n")

    oof_avg = oof_all / oof_counts
    metrics = {
        "oof_r2": r2_score(y, oof_avg),
        "oof_rmse": mean_squared_error(y, oof_avg) ** 0.5,
        "oof_mae": mean_absolute_error(y, oof_avg),
        "seeds": SEEDS,
        "full_data_iterations": seed_best_iters,
        "n_features": int(X_tr.shape[1]),
        "n_train_rows": int(len(X_tr)),
        "n_test_rows": int(len(X_te)),
    }
    print(f"    Multi-seed OOF R2  : {metrics['oof_r2']:.5f}")
    print(f"    Multi-seed OOF RMSE: {metrics['oof_rmse']:.2f}")
    print(f"    Multi-seed OOF MAE : {metrics['oof_mae']:.2f}")

    print("\n[4] Training on FULL data, averaging seed predictions...")
    test_preds = []
    m_full = None
    for seed, n_iter in seed_best_iters.items():
        m_full = CatBoostRegressor(
            iterations=n_iter,
            random_seed=seed,
            **{k: v for k, v in params.items() if k != "early_stopping_rounds"},
        )
        m_full.fit(X_tr, y)
        tp = m_full.predict(X_te)
        test_preds.append(tp)
        print(f"    seed={seed} iter={n_iter} pred_mean={tp.mean():.0f} pred_std={tp.std():.0f}")

    final_preds = np.clip(np.mean(test_preds, axis=0), 0, None)
    print(f"\n    Ensemble pred: mean={final_preds.mean():.0f} std={final_preds.std():.0f}")

    print("\n[5] Feature importances (last seed's model)...")
    fi = pd.Series(m_full.feature_importances_, index=X_tr.columns).sort_values(ascending=False)
    print(fi.to_string())

    print("\n[6] Saving outputs...")
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({"id": test["id"], "cltv": final_preds})
    sub.to_csv(out_path, index=False)
    fi.to_csv(Path(results_dir) / "feature_importance.csv", header=["importance"])
    with open(Path(results_dir) / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"    Predictions -> {out_path} ({sub.shape[0]} rows)")
    print(f"    Metrics     -> {Path(results_dir) / 'metrics.json'}")
    print(f"    Feat. imp.  -> {Path(results_dir) / 'feature_importance.csv'}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the VahanBima CLTV CatBoost model.")
    parser.add_argument("--train", default="data/CLTV_TRAINDATA.xlsx", help="Path to training data (.xlsx or .csv)")
    parser.add_argument("--test", default="data/CLTV_TESTDATA.csv", help="Path to test data (.xlsx or .csv)")
    parser.add_argument("--out", default="submission.csv", help="Where to write predictions")
    parser.add_argument("--results-dir", default="results", help="Where to write metrics.json / feature_importance.csv")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Reduced iterations/folds for a fast smoke test on the sample data - "
        "NOT representative of the reported metrics.",
    )
    args = parser.parse_args()
    run(args.train, args.test, args.out, args.results_dir, args.quick)
