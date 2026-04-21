
 # CANCER RISK PREDICTION SYSTEM

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
# Step 2: Upload Lung Cancer Dataset

import pandas as pd
from google.colab import files

# Upload the CSV file from your computer
uploaded = files.upload()

# Load the dataset
data = pd.read_csv(list(uploaded.keys())[0])

# Show first 5 rows
print("Dataset Preview:")
display(data.head())

# Check dataset information
print("\nDataset Information:")
data.info()

# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(df):
    """Split features/target, scale, train/test split."""
    X = df.drop("cancer_risk", axis=1)
    y = df["cancer_risk"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    return X_train, X_test, X_train_sc, X_test_sc, y_train, y_test, scaler, X.columns.tolist()
# ─────────────────────────────────────────────────────────────────────────────
# 3. MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_models(X_train_sc, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, C=1.0),
        "Random Forest":       RandomForestClassifier(n_estimators=150, max_depth=8,
                                                       random_state=42, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, learning_rate=0.1,
                                                           max_depth=4, random_state=42),
    }
    trained = {}
    for name, clf in models.items():
        clf.fit(X_train_sc, y_train)
        trained[name] = clf
    return trained
# ─────────────────────────────────────────────────────────────────────────────
# 4. EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_models(models, X_train_sc, X_test_sc, y_train, y_test):
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, clf in models.items():
        y_pred     = clf.predict(X_test_sc)
        y_prob     = clf.predict_proba(X_test_sc)[:, 1]
        cv_scores  = cross_val_score(clf, X_train_sc, y_train, cv=cv,
                                     scoring="accuracy", n_jobs=-1)
        results[name] = {
            "accuracy":  accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall":    recall_score(y_test, y_pred),
            "f1":        f1_score(y_test, y_pred),
            "roc_auc":   roc_auc_score(y_test, y_prob),
            "cv_mean":   cv_scores.mean(),
            "cv_std":    cv_scores.std(),
            "y_pred":    y_pred,
            "y_prob":    y_prob,
        }
    return results
# ─────────────────────────────────────────────────────────────────────────────
# 5. FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_importance(models, feature_names):
    rf  = models["Random Forest"]
    gb  = models["Gradient Boosting"]
    lr  = models["Logistic Regression"]

    fi_rf  = rf.feature_importances_
    fi_gb  = gb.feature_importances_
    fi_lr  = np.abs(lr.coef_[0]) / np.abs(lr.coef_[0]).sum()

    fi_avg = (fi_rf + fi_gb + fi_lr) / 3
    idx    = np.argsort(fi_avg)[::-1]

    return {
        "names":   [feature_names[i] for i in idx],
        "rf":      fi_rf[idx],
        "gb":      fi_gb[idx],
        "lr":      fi_lr[idx],
        "average": fi_avg[idx],
    }
 #─────────────────────────────────────────────────────────────────────────────
# 6. RISK PREDICTOR FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def predict_risk(model, scaler, patient_data: dict) -> dict:
    """
    Predict cancer risk for a single patient.
    patient_data keys must match training feature columns.
    """
    feature_cols = [
        "age", "bmi", "gender", "smoking_status", "alcohol_use",
        "physical_activity", "diet_pattern", "sleep_hours",
        "stress_level", "family_history", "medical_conditions"
    ]
    row       = pd.DataFrame([patient_data])[feature_cols]
    row_sc    = scaler.transform(row)
    prob      = model.predict_proba(row_sc)[0][1]
    pred      = int(prob >= 0.50)
    risk_pct  = round(prob * 100, 1)

    if risk_pct < 25:
        level, advice = "LOW", "Maintain healthy habits. Annual routine check-up recommended."
    elif risk_pct < 50:
        level, advice = "MODERATE", "Consider lifestyle improvements and biennial cancer screening."
    elif risk_pct < 75:
        level, advice = "HIGH", "Consult a physician promptly. Targeted cancer screening advised."
    else:
        level, advice = "VERY HIGH", "Urgent medical consultation required. Comprehensive screening needed."

    return {
        "risk_probability": risk_pct,
        "risk_level":       level,
        "prediction":       pred,
        "advice":           advice,
    }
# ─────────────────────────────────────────────────────────────────────────────
# 7. VISUALISATION DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

PALETTE  = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12", "#9b59b6", "#1abc9c"]
BG_DARK  = "#0d1117"
GRID_CLR = "#21262d"
TEXT_CLR = "#c9d1d9"
ACCENT   = "#238636"

plt.rcParams.update({
    "figure.facecolor":  BG_DARK,
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    GRID_CLR,
    "axes.labelcolor":   TEXT_CLR,
    "xtick.color":       TEXT_CLR,
    "ytick.color":       TEXT_CLR,
    "text.color":        TEXT_CLR,
    "grid.color":        GRID_CLR,
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

def plot_dashboard(results, fi, df, y_test, output_path):
    fig = plt.figure(figsize=(22, 26), facecolor=BG_DARK)
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.38,
                             left=0.07, right=0.96, top=0.94, bottom=0.04)

    model_names = list(results.keys())
    colors_m    = ["#3498db", "#2ecc71", "#e67e22"]

    # ── Title ────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.97, "Cancer Risk Prediction System — ML Dashboard",
             ha="center", va="top", fontsize=20, fontweight="bold", color="#e6edf3")
    fig.text(0.5, 0.955, "B.Tech Minor Project | Artificial Intelligence & Data Science",
             ha="center", va="top", fontsize=11, color="#8b949e")

    # ── 1. Metrics Comparison Bar Chart ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    metrics    = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    mlabels    = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    x          = np.arange(len(metrics))
    bw         = 0.25
    for i, (nm, clr) in enumerate(zip(model_names, colors_m)):
        vals = [results[nm][m] for m in metrics]
        bars = ax1.bar(x + i*bw, vals, bw, label=nm, color=clr, alpha=0.88,
                       edgecolor="#0d1117", linewidth=0.6, zorder=3)
        for bar, val in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=7.5,
                     color=TEXT_CLR, fontweight="bold")
    ax1.set_xticks(x + bw)
    ax1.set_xticklabels(mlabels, fontsize=9)
    ax1.set_ylim(0, 1.10)
    ax1.set_ylabel("Score", fontsize=10)
    ax1.set_title("Model Performance Comparison", fontsize=12, fontweight="bold", pad=10)
    ax1.legend(fontsize=9, framealpha=0.2)
    ax1.grid(axis="y", zorder=0)

    # ── 2. Cross-Validation ───────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    cv_means = [results[nm]["cv_mean"] for nm in model_names]
    cv_stds  = [results[nm]["cv_std"]  for nm in model_names]
    ax2.barh(model_names, cv_means, xerr=cv_stds, color=colors_m,
             alpha=0.85, edgecolor="#0d1117", capsize=4, zorder=3)
    for i, (m, s) in enumerate(zip(cv_means, cv_stds)):
        ax2.text(m + s + 0.002, i, f"{m:.3f}±{s:.3f}", va="center", fontsize=8, color=TEXT_CLR)
    ax2.set_xlim(0.6, 1.05)
    ax2.set_xlabel("CV Accuracy (5-Fold)", fontsize=9)
    ax2.set_title("Cross-Validation Scores", fontsize=12, fontweight="bold", pad=10)
    ax2.grid(axis="x", zorder=0)

    # ── 3. ROC Curves ────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    for nm, clr in zip(model_names, colors_m):
        fpr, tpr, _ = roc_curve(y_test, results[nm]["y_prob"])
        auc_val     = results[nm]["roc_auc"]
        ax3.plot(fpr, tpr, color=clr, lw=2.0, label=f"{nm} (AUC={auc_val:.3f})")
        ax3.fill_between(fpr, tpr, alpha=0.06, color=clr)
    ax3.plot([0,1],[0,1], "w--", lw=1, alpha=0.4, label="Random (AUC=0.5)")
    ax3.set_xlabel("False Positive Rate", fontsize=9)
    ax3.set_ylabel("True Positive Rate", fontsize=9)
    ax3.set_title("ROC-AUC Curves", fontsize=12, fontweight="bold", pad=10)
    ax3.legend(fontsize=7.5, framealpha=0.2)
    ax3.grid(True)

    # ── 4. Confusion Matrix (Best Model = GB) ────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    cm        = confusion_matrix(y_test, results[best_name]["y_pred"])
    cm_norm   = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im        = ax4.imshow(cm_norm, cmap="Greens", vmin=0, vmax=1)
    labels    = ["Low Risk", "High Risk"]
    ax4.set_xticks([0,1]); ax4.set_yticks([0,1])
    ax4.set_xticklabels(labels, fontsize=9); ax4.set_yticklabels(labels, fontsize=9)
    ax4.set_xlabel("Predicted", fontsize=9); ax4.set_ylabel("Actual", fontsize=9)
    ax4.set_title(f"Confusion Matrix — {best_name}", fontsize=11, fontweight="bold", pad=10)
    for i in range(2):
        for j in range(2):
            ax4.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.1%})",
                     ha="center", va="center", fontsize=12,
                     color="white" if cm_norm[i,j] > 0.5 else TEXT_CLR, fontweight="bold")
    plt.colorbar(im, ax=ax4, fraction=0.046)

    # ── 5. Feature Importance (Avg) ───────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    feat_display = {
        "age":"Age","bmi":"BMI","gender":"Gender",
        "smoking_status":"Smoking","alcohol_use":"Alcohol",
        "physical_activity":"Activity","diet_pattern":"Diet",
        "sleep_hours":"Sleep","stress_level":"Stress",
        "family_history":"Family Hist.","medical_conditions":"Med. History"
    }
    names_disp = [feat_display.get(n, n) for n in fi["names"]]
    bar_clr    = [PALETTE[i % len(PALETTE)] for i in range(len(fi["names"]))]
    bars = ax5.barh(names_disp[::-1], fi["average"][::-1], color=bar_clr[::-1],
                    alpha=0.88, edgecolor="#0d1117", zorder=3)
    for bar, val in zip(bars, fi["average"][::-1]):
        ax5.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center", fontsize=7.5, color=TEXT_CLR)
    ax5.set_xlabel("Importance Score", fontsize=9)
    ax5.set_title("Feature Importance (Average)", fontsize=12, fontweight="bold", pad=10)
    ax5.grid(axis="x", zorder=0)

    # ── 6. Age Distribution by Risk ──────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.hist(df[df.cancer_risk==0]["age"], bins=25, color="#2ecc71", alpha=0.7, label="Low Risk", edgecolor="#0d1117")
    ax6.hist(df[df.cancer_risk==1]["age"], bins=25, color="#e74c3c", alpha=0.7, label="High Risk", edgecolor="#0d1117")
    ax6.set_xlabel("Age (years)", fontsize=9); ax6.set_ylabel("Count", fontsize=9)
    ax6.set_title("Age Distribution by Risk Group", fontsize=12, fontweight="bold", pad=10)
    ax6.legend(fontsize=9, framealpha=0.2); ax6.grid(axis="y")

    # ── 7. BMI vs Age scatter ────────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 1])
    low  = df[df.cancer_risk==0]
    high = df[df.cancer_risk==1]
    ax7.scatter(low["age"],  low["bmi"],  c="#2ecc71", alpha=0.35, s=12, label="Low Risk")
    ax7.scatter(high["age"], high["bmi"], c="#e74c3c", alpha=0.35, s=12, label="High Risk")
    ax7.axhline(30, color="#f39c12", ls="--", lw=1, alpha=0.6, label="BMI 30 (Obese)")
    ax7.set_xlabel("Age", fontsize=9); ax7.set_ylabel("BMI", fontsize=9)
    ax7.set_title("BMI vs Age — Risk Distribution", fontsize=12, fontweight="bold", pad=10)
    ax7.legend(fontsize=9, framealpha=0.2); ax7.grid(True)

    # ── 8. Smoking status risk proportion ───────────────────────────────────
    ax8 = fig.add_subplot(gs[2, 2])
    smoke_labels = ["Never", "Former", "Light\nSmoker", "Heavy\nSmoker"]
    smoke_risks  = []
    for s in range(4):
        subset = df[df.smoking_status==s]
        smoke_risks.append(subset.cancer_risk.mean() * 100)
    smoke_clr = ["#2ecc71", "#f39c12", "#e67e22", "#e74c3c"]
    bars = ax8.bar(smoke_labels, smoke_risks, color=smoke_clr, alpha=0.88, edgecolor="#0d1117", zorder=3)
    for bar, val in zip(bars, smoke_risks):
        ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=9, color=TEXT_CLR, fontweight="bold")
    ax8.set_ylabel("High Risk % in Group", fontsize=9)
    ax8.set_title("Cancer Risk Rate by Smoking Status", fontsize=12, fontweight="bold", pad=10)
    ax8.set_ylim(0, 100); ax8.grid(axis="y", zorder=0)

    # ── 9. Risk distribution ─────────────────────────────────────────────────
    ax9 = fig.add_subplot(gs[3, 0])
    risk_counts = df.cancer_risk.value_counts()
    wedges, texts, autotexts = ax9.pie(
        risk_counts, labels=["Low Risk", "High Risk"],
        colors=["#2ecc71", "#e74c3c"], autopct="%1.1f%%",
        startangle=90, wedgeprops={"edgecolor":"#0d1117","linewidth":1.5},
        textprops={"color": TEXT_CLR, "fontsize": 10}
    )
    for at in autotexts: at.set_fontweight("bold")
    ax9.set_title("Dataset Risk Distribution", fontsize=12, fontweight="bold", pad=10)

    # ── 10. Activity level vs Risk ───────────────────────────────────────────
    ax10 = fig.add_subplot(gs[3, 1])
    act_labels = ["Sedentary", "Light", "Moderate", "Active", "Very\nActive"]
    act_risks  = [df[df.physical_activity==a].cancer_risk.mean()*100 for a in range(5)]
    act_clrs   = ["#e74c3c","#e67e22","#f39c12","#2ecc71","#1abc9c"]
    bars = ax10.bar(act_labels, act_risks, color=act_clrs, alpha=0.88, edgecolor="#0d1117", zorder=3)
    for bar, val in zip(bars, act_risks):
        ax10.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                  f"{val:.1f}%", ha="center", va="bottom", fontsize=9, color=TEXT_CLR, fontweight="bold")
    ax10.set_ylabel("High Risk % in Group", fontsize=9)
    ax10.set_title("Risk Rate by Physical Activity Level", fontsize=12, fontweight="bold", pad=10)
    ax10.set_ylim(0, 100); ax10.grid(axis="y", zorder=0)

    # ── 11. Stress level vs Risk ─────────────────────────────────────────────
    ax11 = fig.add_subplot(gs[3, 2])
    stress_risks = [df[df.stress_level==s].cancer_risk.mean()*100 for s in range(1,11)]
    ax11.plot(range(1,11), stress_risks, color="#9b59b6", lw=2.5, marker="o",
              markersize=7, markerfacecolor="#e67e22", zorder=3)
    ax11.fill_between(range(1,11), stress_risks, alpha=0.15, color="#9b59b6")
    ax11.set_xlabel("Stress Level (1–10)", fontsize=9)
    ax11.set_ylabel("High Risk % in Group", fontsize=9)
    ax11.set_title("Stress Level vs Cancer Risk Rate", fontsize=12, fontweight="bold", pad=10)
    ax11.set_xticks(range(1,11)); ax11.grid(True)

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
    plt.close()
    print(f"[✔] Dashboard saved → {output_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. PRINT REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_report(results, fi, sample_pred):
    SEP = "═" * 65
    print(f"\n{SEP}")
    print("  CANCER RISK PREDICTION — MODEL EVALUATION REPORT")
    print(SEP)

    for nm, r in results.items():
        print(f"\n  ▶  {nm}")
        print(f"     Accuracy  : {r['accuracy']:.4f}  ({r['accuracy']*100:.2f}%)")
        print(f"     Precision : {r['precision']:.4f}")
        print(f"     Recall    : {r['recall']:.4f}")
        print(f"     F1-Score  : {r['f1']:.4f}")
        print(f"     ROC-AUC   : {r['roc_auc']:.4f}")
        print(f"     CV Acc    : {r['cv_mean']:.4f} ± {r['cv_std']:.4f}")

    print(f"\n{SEP}")
    print("  FEATURE IMPORTANCE (TOP 5 — Average across models)")
    print(SEP)
    feat_display = {
        "age":"Age","bmi":"BMI","gender":"Gender",
        "smoking_status":"Smoking Status","alcohol_use":"Alcohol Use",
        "physical_activity":"Physical Activity","diet_pattern":"Diet Pattern",
        "sleep_hours":"Sleep Hours","stress_level":"Stress Level",
        "family_history":"Family History","medical_conditions":"Medical History"
    }
    for i in range(5):
        n = feat_display.get(fi["names"][i], fi["names"][i])
        print(f"  {i+1}. {n:<25}  Importance: {fi['average'][i]:.4f}")

    print(f"\n{SEP}")
    print("  SAMPLE PATIENT — RISK PREDICTION (Gradient Boosting)")
    print(SEP)
    p = sample_pred
    bar_len = int(p["risk_probability"] / 2)
    bar     = "█" * bar_len + "░" * (50 - bar_len)
    print(f"\n  Risk Probability : {p['risk_probability']}%")
    print(f"  [{bar}]")
    print(f"\n  Risk Level  : {p['risk_level']}")
    print(f"  Prediction  : {'High Risk ⚠' if p['prediction'] else 'Low Risk ✓'}")
    print(f"  Advice      : {p['advice']}")
    print(f"\n{SEP}\n")
    import joblib
# ─────────────────────────────────────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    OUT_IMG = "/content/cancer_risk_dashboard.png"

    print("\n[1/5] Generating synthetic healthcare dataset …")
    df = generate_dataset(n_samples=1500)
    print(f"      Dataset shape : {df.shape}")
    print(f"      Risk ratio    : {df.cancer_risk.value_counts().to_dict()}")

    print("\n[2/5] Preprocessing data …")
    X_tr, X_te, X_tr_sc, X_te_sc, y_tr, y_te, scaler, feat_names = preprocess(df)
    print(f"      Train : {X_tr.shape}  |  Test : {X_te.shape}")

    print("\n[3/5] Training models (Logistic Regression, Random Forest, Gradient Boosting) …")
    models = train_models(X_tr_sc, y_tr)
    print("      All models trained ✔")

    print("\n[4/5] Evaluating models …")
    results = evaluate_models(models, X_tr_sc, X_te_sc, y_tr, y_te)
    fi      = get_feature_importance(models, feat_names)

    # Sample patient prediction
    sample_patient = {
        "age": 58, "bmi": 31.2, "gender": 1,
        "smoking_status": 2,       # Current Light Smoker
        "alcohol_use": 1,          # Occasional
        "physical_activity": 1,    # Light
        "diet_pattern": 1,         # High Fat
        "sleep_hours": 5.5,
        "stress_level": 7,
        "family_history": 1,       # Yes
        "medical_conditions": 2,
    }
    gb_model    = models["Gradient Boosting"]
    sample_pred = predict_risk(gb_model, scaler, sample_patient)

    print_report(results, fi, sample_pred)

    print("[5/5] Generating visualisation dashboard …")
    plot_dashboard(results, fi, df, y_te, OUT_IMG)
    print("\n[✔] All done!\n")

    # Save model and scaler
    best_model = models["Gradient Boosting"]   # or whichever you want
    joblib.dump(best_model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("✔ Model and scaler saved successfully!")


if __name__ == "__main__":
    main()
