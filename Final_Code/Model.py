# Model.py - Fixed Version
import json, warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix, accuracy_score, f1_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.impute import SimpleImputer
import joblib

# Optional models
try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False
    print("XGBoost not available")

try:
    from catboost import CatBoostClassifier
    HAVE_CATBOOST = True
except Exception:
    HAVE_CATBOOST = False
    print("CatBoost not available")

# ------------------ Paths / constants ------------------
RAW_DATA_PATH      = "Machine Downtime.csv"
IMPUTED_DATA_PATH  = "results/imputation_results/df_imputed.csv"

RESULTS_DIR  = Path("results")
MODELS_DIR   = RESULTS_DIR / "models"
ANALYSIS_DIR = RESULTS_DIR / "analysis"
PLOTS_DIR    = ANALYSIS_DIR / "plots"
for d in (RESULTS_DIR, MODELS_DIR, ANALYSIS_DIR, PLOTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
CV_FOLDS = 5

# ------------------ Data ------------------
def load_and_prepare_data():
    """Load imputed dataset if present; else raw CSV. Build numeric feature matrix."""
    use_path = IMPUTED_DATA_PATH if Path(IMPUTED_DATA_PATH).exists() else RAW_DATA_PATH
    print(f"Loading data from: {use_path}")
    df = pd.read_csv(use_path, parse_dates=["Date"], dayfirst=True)

    machine_map = {
        "Makino-L1-Unit1-2013": "M1",
        "Makino-L2-Unit1-2015": "M2",
        "Makino-L3-Unit1-2015": "M3",
    }
    df["Machine_ID"] = df["Machine_ID"].map(machine_map).fillna(df["Machine_ID"])
    df["y"] = (df["Downtime"] == "Machine_Failure").astype(int)

    exclude = {"Machine_ID", "Assembly_Line_No", "Date", "Downtime", "y"}
    cand = [c for c in df.columns if c not in exclude]
    X = df[cand].select_dtypes(include=[np.number]).copy()
    y = df["y"].copy()
    groups = df["Machine_ID"].copy()

    meta = {
        "feature_columns": X.columns.tolist(),
        "sensors_used": X.columns.tolist(),
        "machine_mapping": machine_map,
        "total_samples": int(len(X)),
        "failure_rate": float(y.mean()),
        "data_source": use_path,
    }
    (ANALYSIS_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"X shape: {X.shape} | y mean: {y.mean():.3f}")
    print(f"Machines: {sorted(groups.unique().tolist())}")
    return X, y, groups

def split_train_test_with_safety_imputer(X, y, groups):
    """Split by machine (M1+M2 train, M3 test). If NaNs exist, impute using train-only fit."""
    train_mask = groups.isin(["M1", "M2"])
    test_mask  = groups == "M3"
    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    if X_train.isna().any().any() or X_test.isna().any().any():
        imp = SimpleImputer(strategy="median")
        X_train = pd.DataFrame(imp.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
        X_test  = pd.DataFrame(imp.transform(X_test),  index=X_test.index,  columns=X_test.columns)
        print("Note: NaNs detected; applied train-fitted median imputation.")
    return X_train, X_test, y_train, y_test

# ------------------ Leakage guard ------------------
def leakage_guard(X_train, y_train, X_test, auc_threshold=0.995):
    """Remove features that nearly separate the classes on TRAIN (single-feature AUC)."""
    bad = []
    for c in X_train.columns:
        s = X_train[c]
        if s.nunique() < 2:
            continue
        try:
            auc = roc_auc_score(y_train, s)
            auc = max(auc, 1 - auc)
            if auc >= auc_threshold:
                bad.append(c)
        except Exception:
            continue
    if bad:
        print(f"Leakage guard removed {len(bad)} feature(s): {bad}")
        X_train = X_train.drop(columns=bad)
        X_test  = X_test.drop(columns=bad, errors="ignore")
    return X_train, X_test, bad

def report_exact_duplicates_between_splits(X_train, X_test):
    """Detect exact duplicate rows across splits."""
    h_train = pd.util.hash_pandas_object(X_train, index=False)
    h_test  = pd.util.hash_pandas_object(X_test,  index=False)
    dup = np.intersect1d(h_train.values, h_test.values).size
    if dup > 0:
        print(f"Warning: {dup} exact duplicate row(s) appear in both train and test.")

def label_shuffle_sanity_check(model, X_train, y_train):
    """With shuffled labels, CV AUC should be ~0.5."""
    y_perm = y_train.sample(frac=1.0, random_state=RANDOM_STATE).values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    auc = cross_val_score(model, X_train, y_perm, cv=cv, scoring="roc_auc", n_jobs=-1).mean()
    print(f"Sanity (shuffled labels): CV AUC ≈ {auc:.3f}")

# ------------------ Models ------------------
def get_models():
    models = {}
    
    # Random Forest
    models["RandomForest"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_split=5, min_samples_leaf=2,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ))
    ])
    
    # XGBoost
    if HAVE_XGB:
        models["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
            ))
        ])
    
    # CatBoost
    if HAVE_CATBOOST:
        models["CatBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CatBoostClassifier(
                iterations=300, depth=6, learning_rate=0.1,
                loss_function="Logloss", eval_metric="Logloss",
                random_state=RANDOM_STATE, verbose=False
            ))
        ])
    
    return models

# ------------------ Training-history plots ------------------
def plot_training_history(loss_train, loss_val, title, save_path):
    if loss_train is None:
        return
    epochs = np.arange(1, len(loss_train) + 1)
    plt.figure(figsize=(11, 8))
    plt.title(title, fontsize=18, fontweight="bold")
    plt.plot(epochs, loss_train, label="Training Loss", linewidth=3)
    if loss_val is not None:
        plt.plot(epochs, loss_val, label="Validation Loss", linewidth=3)
    plt.xlabel("Epochs"); plt.ylabel("Loss")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches="tight"); plt.close()

def train_with_validation_split(clf, X_train, y_train, model_name):
    """Train model with validation split for history tracking."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE
    )
    
    # Calculate class weights
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    ratio = (neg / max(pos, 1)) if pos > 0 else 1.0
    
    loss_train = loss_val = None
    
    if model_name == "CatBoost" and HAVE_CATBOOST:
        # Set class weights for CatBoost
        clf.set_params(class_weights=[1.0, ratio])
        
        # Train with validation set for history
        try:
            clf.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
            # Get training history if available
            if hasattr(clf, 'get_evals_result'):
                res = clf.get_evals_result()
                loss_train = res.get("learn", {}).get("Logloss", None)
                loss_val = res.get("validation", {}).get("Logloss", None)
        except Exception as e:
            print(f"Warning: CatBoost validation training failed: {e}")
            clf.fit(X_tr, y_tr, verbose=False)
        
        # Retrain on full training data
        clf.fit(X_train, y_train, verbose=False)
        
    elif model_name == "XGBoost" and HAVE_XGB:
        # Set scale_pos_weight for XGBoost
        clf.set_params(scale_pos_weight=ratio)
        
        # Train with validation set for history
        try:
            eval_set = [(X_tr, y_tr), (X_val, y_val)]
            clf.fit(X_tr, y_tr, eval_set=eval_set, verbose=False)
            # Get training history if available
            if hasattr(clf, 'evals_result_'):
                res = clf.evals_result_
                loss_train = res.get("validation_0", {}).get("logloss", None)
                loss_val = res.get("validation_1", {}).get("logloss", None)
        except Exception as e:
            print(f"Warning: XGBoost validation training failed: {e}")
            clf.fit(X_tr, y_tr, verbose=False)
        
        # Retrain on full training data
        clf.fit(X_train, y_train, verbose=False)
        
    else:
        # Standard training for other models
        clf.set_params(class_weight="balanced")
        clf.fit(X_train, y_train)
    
    return loss_train, loss_val

# ------------------ Evaluation ------------------
def evaluate_one(name, pipe, X_train, y_train, X_test, y_test):
    print(f"\nEvaluating {name} ...")
    
    # Extract classifier from pipeline
    clf = pipe.named_steps["clf"]
    
    # Train the classifier with validation tracking
    loss_train, loss_val = train_with_validation_split(clf, X_train, y_train, name)
    
    # Train the full pipeline (including scaler)
    pipe.fit(X_train, y_train)
    
    # Create training history plot
    if loss_train is not None:
        plot_training_history(loss_train, loss_val, f"{name} - Training History",
                              PLOTS_DIR / f"{name.replace(' ', '').lower()}_training_history.png")

    # Predictions
    if hasattr(pipe, "predict_proba"):
        y_proba = pipe.predict_proba(X_test)[:, 1]
    else:
        s = pipe.decision_function(X_test)
        y_proba = (s - s.min()) / (s.max() - s.min() + 1e-9)
    y_pred = (y_proba >= 0.5).astype(int)

    # Metrics
    auc = roc_auc_score(y_test, y_proba)
    ap  = average_precision_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    print(f"  Test: AUC={auc:.3f}  AP={ap:.3f}  Acc={acc:.3f}  F1={f1:.3f}")

    # CV on TRAIN only
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)

    # Per-model evaluation panel
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f"{name} - Performance Analysis", fontsize=16, fontweight="bold")

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[0,0].plot(fpr, tpr, linewidth=2.5, label=f"AUC = {auc:.3f}")
    axes[0,0].plot([0,1], [0,1], "k--", alpha=0.6)
    axes[0,0].set_title("ROC Curve"); axes[0,0].set_xlabel("FPR"); axes[0,0].set_ylabel("TPR")
    axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    axes[0,1].plot(rec, prec, linewidth=2.5, label=f"AP = {ap:.3f}")
    axes[0,1].set_title("Precision–Recall"); axes[0,1].set_xlabel("Recall"); axes[0,1].set_ylabel("Precision")
    axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1,0],
                xticklabels=["No Failure","Failure"], yticklabels=["No Failure","Failure"])
    axes[1,0].set_title("Confusion Matrix"); axes[1,0].set_xlabel("Predicted"); axes[1,0].set_ylabel("Actual")

    axes[1,1].hist(y_proba[y_test==0], bins=20, alpha=0.7, label="No Failure", density=True)
    axes[1,1].hist(y_proba[y_test==1], bins=20, alpha=0.7, label="Failure", density=True)
    axes[1,1].set_title("Predicted Probability Distributions")
    axes[1,1].set_xlabel("Probability"); axes[1,1].set_ylabel("Density")
    axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{name.replace(' ', '').lower()}_evaluation.png", dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "Model": name,
        "Test_AUC": float(auc),
        "Test_AP": float(ap),
        "Test_Accuracy": float(acc),
        "Test_F1": float(f1),
        "CV_AUC_Mean": float(cv_scores.mean()),
        "CV_AUC_Std": float(cv_scores.std()),
        "CV_AUC_Folds": cv_scores.tolist(),
        "Model_Path": str(MODELS_DIR / f"{name.replace(' ', '').lower()}_model.joblib"),
    }

# ------------------ 2×2 comparison dashboard ------------------
def model_comparison_dashboard(results):
    df = pd.DataFrame(results).sort_values("Test_AUC", ascending=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Model Performance Comparison", fontsize=16, fontweight="bold")

    # (1) Bar: Test AUC
    axes[0,0].barh(df["Model"], df["Test_AUC"])
    axes[0,0].set_xlabel("Test AUC"); axes[0,0].invert_yaxis()
    for ytick, val in zip(axes[0,0].get_yticks(), df["Test_AUC"]):
        axes[0,0].text(val + 0.001, ytick, f"{val:.3f}", va="center")

    # (2) Boxplot: CV AUC folds
    box_data, labels = [], []
    for _, r in df.iterrows():
        if isinstance(r["CV_AUC_Folds"], list) and r["CV_AUC_Folds"]:
            box_data.append(r["CV_AUC_Folds"]); labels.append(r["Model"])
    axes[0,1].boxplot(box_data, labels=labels, patch_artist=True)
    axes[0,1].set_ylabel("CV AUC Score"); axes[0,1].set_title("Cross-Validation Score Distribution")

    # (3) Scatter: Test AUC vs AP
    axes[1,0].scatter(df["Test_AUC"], df["Test_AP"], s=80)
    for i, r in df.iterrows():
        axes[1,0].annotate(r["Model"], (r["Test_AUC"], r["Test_AP"]), xytext=(4,4), textcoords="offset points")
    axes[1,0].set_xlabel("Test AUC"); axes[1,0].set_ylabel("Average Precision"); axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_title("AUC vs Average Precision")

    # (4) Summary table
    tbl = df[["Model","Test_AUC","Test_AP","CV_AUC_Mean","CV_AUC_Std"]].copy()
    tbl["CV_AUC"] = (tbl["CV_AUC_Mean"].round(3)).astype(str) + "±" + (tbl["CV_AUC_Std"].round(3)).astype(str)
    tbl = tbl[["Model","Test_AUC","Test_AP","CV_AUC"]].round(3)
    axes[1,1].axis("off")
    axes[1,1].table(cellText=tbl.values, colLabels=tbl.columns, loc="center")
    axes[1,1].set_title("Performance Summary")

    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig(PLOTS_DIR / "model_comparison_dashboard.png", dpi=300, bbox_inches="tight")
    plt.close()

# ------------------ Main ------------------
def main():
    print("🚀 Training CNC Failure Prediction Models")
    print("=" * 60)
    
    X, y, groups = load_and_prepare_data()

    X_train, X_test, y_train, y_test = split_train_test_with_safety_imputer(X, y, groups)
    X_train, X_test, removed = leakage_guard(X_train, y_train, X_test, auc_threshold=0.995)
    report_exact_duplicates_between_splits(X_train, X_test)
    
    # Quick sanity check
    label_shuffle_sanity_check(RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=RANDOM_STATE),
                               X_train, y_train)

    print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
    print(f"Train failure rate: {y_train.mean():.3f} | Test failure rate: {y_test.mean():.3f}")

    models = get_models()
    print(f"\nAvailable models: {list(models.keys())}")
    
    results = []
    for name, pipe in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name}")
        print(f"{'='*50}")
        
        try:
            res = evaluate_one(name, pipe, X_train, y_train, X_test, y_test)
            
            # Save fitted pipeline
            model_path = MODELS_DIR / f"{name.replace(' ', '').lower()}_model.joblib"
            joblib.dump(pipe, model_path)
            print(f"✅ Model saved: {model_path}")
            
            results.append(res)
            
        except Exception as e:
            print(f"❌ Failed to train {name}: {str(e)}")
            continue

    if results:
        pd.DataFrame(results).to_csv(ANALYSIS_DIR / "model_comparison.csv", index=False)
        model_comparison_dashboard(results)

        print("\n" + "="*60)
        print("🎯 TRAINING COMPLETE")
        print("="*60)
        print("\nModel Rankings (by Test AUC):")
        for i, res in enumerate(sorted(results, key=lambda x: x['Test_AUC'], reverse=True), 1):
            print(f"{i}. {res['Model']}: {res['Test_AUC']:.3f}")

        print(f"\n📊 Results saved to: {ANALYSIS_DIR}")
        print(f"🤖 Models saved to: {MODELS_DIR}")
        print(f"📈 Plots saved to: {PLOTS_DIR}")
    else:
        print("❌ No models were successfully trained!")

if __name__ == "__main__":
    main()