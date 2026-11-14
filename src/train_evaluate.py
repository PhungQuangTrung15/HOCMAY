import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import (
    build_preprocessor_flow_a,
    build_preprocessor_flow_b,
    load_and_clean_data,
    split_train_test,
)
from src.utils import (
    ERROR_CSV_PATH,
    FIGURES_DIR,
    MODEL_PATH,
    RESULTS_CSV_PATH,
    TUNING_CSV_PATH,
    ensure_directories,
    get_risk_tier,
    set_seed,
)


def _generate_eda_figures(df: pd.DataFrame) -> None:
    # Hình 1: Phân phối nhãn
    plt.figure(figsize=(6, 4))
    labels = df["Dropout"].map({0: "Graduate (0)", 1: "Dropout (1)"})
    sns.countplot(x=labels, palette=["#2ecc71", "#e74c3c"])
    plt.title("Phân phối nhãn Graduate vs Dropout")
    plt.xlabel("Trạng thái")
    plt.ylabel("Số lượng sinh viên")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "hinh1_phan_phoi_nhan.png", dpi=150)
    plt.close()

    # Hình 2: Tương quan tài chính
    fin_cols = [c for c in ["Tuition fees up to date", "Debtor", "Scholarship holder", "Dropout"] if c in df.columns]
    plt.figure(figsize=(6, 4.5))
    sns.heatmap(df[fin_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Ma trận tương quan các yếu tố tài chính & Bỏ học")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "hinh2_tai_chinh_tuong_quan.png", dpi=150)
    plt.close()

    # Hình 3: Boxplot điểm số / số tín chỉ đạt
    grade_col = next(
        (c for c in ["Curricular units 2nd sem (grade)", "Curricular units 1st sem (grade)"] if c in df.columns),
        df.select_dtypes(include=[np.number]).columns[0],
    )
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df["Dropout"].map({0: "Graduate", 1: "Dropout"}), y=df[grade_col], palette=["#3498db", "#e67e22"])
    plt.title(f"Phân phối {grade_col} theo nhóm sinh viên")
    plt.xlabel("Trạng thái")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "hinh3_boxplot_hoc_tap.png", dpi=150)
    plt.close()


def run_experiment_pipeline() -> pd.DataFrame:
    set_seed(42)
    ensure_directories()

    df = load_and_clean_data()
    _generate_eda_figures(df)

    X_train, X_test, y_train, y_test = split_train_test(df, test_size=0.20, random_state=42)

    flows = {
        "Flow_A": lambda: build_preprocessor_flow_a(X_train),
        "Flow_B": lambda: build_preprocessor_flow_b(X_train),
    }

    models_spec = {
        "KNN": (
            KNeighborsClassifier(),
            {"clf__n_neighbors": [5, 11], "clf__weights": ["uniform", "distance"]},
        ),
        "NaiveBayes": (
            GaussianNB(),
            {"clf__var_smoothing": [1e-9, 1e-7]},
        ),
        "DecisionTree": (
            DecisionTreeClassifier(random_state=42),
            {"clf__max_depth": [5, 10, None], "clf__min_samples_split": [2, 10]},
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=42, n_jobs=1),
            {"clf__n_estimators": [100, 200], "clf__max_depth": [10, None]},
        ),
    }

    results_rows = []
    tuning_rows = []
    roc_curves = {}

    best_f1 = -1.0
    best_model = None
    best_name = ""

    for flow_name, preproc_fn in flows.items():
        for model_name, (estimator, param_grid) in models_spec.items():
            exp_name = f"{model_name}-{flow_name}"
            pipe = Pipeline([
                ("preproc", preproc_fn()),
                ("clf", estimator),
            ])

            grid = GridSearchCV(
                pipe,
                param_grid=param_grid,
                cv=3,
                scoring="f1",
                n_jobs=1,
            )
            grid.fit(X_train, y_train)
            best_est = grid.best_estimator_

            y_pred = best_est.predict(X_test)
            y_prob = best_est.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_prob)

            results_rows.append({
                "Model": model_name,
                "Flow": flow_name,
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1_Score": round(f1, 4),
                "ROC_AUC": round(auc, 4),
            })

            tuning_rows.append({
                "Experiment": exp_name,
                "Best_Params": str(grid.best_params_),
                "CV_Best_F1": round(grid.best_score_, 4),
                "Test_F1": round(f1, 4),
            })

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_curves[exp_name] = (fpr, tpr, auc)

            if f1 > best_f1:
                best_f1 = f1
                best_model = best_est
                best_name = exp_name

    results_df = pd.DataFrame(results_rows).sort_values(by="F1_Score", ascending=False).reset_index(drop=True)
    tuning_df = pd.DataFrame(tuning_rows)

    results_df.to_csv(RESULTS_CSV_PATH, index=False)
    tuning_df.to_csv(TUNING_CSV_PATH, index=False)
    joblib.dump(best_model, MODEL_PATH)

    # Hình 4: So sánh Luồng A vs B (F1-score)
    plt.figure(figsize=(8, 4.5))
    sns.barplot(data=results_df, x="Model", y="F1_Score", hue="Flow", palette="viridis")
    plt.title("So sánh F1-Score giữa Luồng A (Baseline) và Luồng B (Advanced)")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "hinh4_so_sanh_luong_A_B.png", dpi=150)
    plt.close()

    # Hình 5: Đường cong ROC-AUC
    plt.figure(figsize=(7, 5))
    for exp_name, (fpr, tpr, auc) in roc_curves.items():
        if "Flow_B" in exp_name or exp_name == best_name:
            plt.plot(fpr, tpr, label=f"{exp_name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Đường cong ROC-AUC các mô hình")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "hinh5_roc_auc_curve.png", dpi=150)
    plt.close()

    # Đánh giá chi tiết mô hình tốt nhất -> Confusion Matrix, Feature Importance & Error Analysis
    best_pred = best_model.predict(X_test)
    best_prob = best_model.predict_proba(X_test)[:, 1]

    # Hình 6: Confusion Matrix
    cm = confusion_matrix(y_test, best_pred)
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Graduate", "Dropout"], yticklabels=["Graduate", "Dropout"])
    plt.xlabel("Dự đoán")
    plt.ylabel("Thực tế")
    plt.title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "hinh6_confusion_matrix.png", dpi=150)
    plt.close()

    # Hình 7: Feature Importance
    clf_step = best_model.named_steps["clf"]
    preproc_step = best_model.named_steps["preproc"]
    try:
        if isinstance(preproc_step, Pipeline):
            feat_names = preproc_step.named_steps["col_trans"].get_feature_names_out()
        else:
            feat_names = preproc_step.get_feature_names_out()
    except Exception:
        feat_names = [f"Feature_{i}" for i in range(best_model.named_steps["preproc"].transform(X_test).shape[1])]

    if hasattr(clf_step, "feature_importances_"):
        importances = clf_step.feature_importances_
        top_idx = np.argsort(importances)[-10:]
        plt.figure(figsize=(8, 4.5))
        plt.barh([feat_names[i].split("__")[-1] for i in top_idx], importances[top_idx], color="#9b59b6")
        plt.xlabel("Mức độ quan trọng")
        plt.title(f"Top 10 đặc trưng quan trọng nhất ({best_name})")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "hinh7_feature_importance.png", dpi=150)
        plt.close()

    # Xuất file Error Analysis
    error_df = X_test.copy().reset_index(drop=True)
    error_df.insert(0, "Test_Student_Index", range(len(error_df)))
    error_df["Actual_Dropout"] = y_test.values
    error_df["Predicted_Dropout"] = best_pred
    error_df["Dropout_Probability"] = np.round(best_prob, 4)
    error_df["Risk_Tier"] = [get_risk_tier(p) for p in best_prob]

    def _classify_error(row):
        if row["Actual_Dropout"] == 1 and row["Predicted_Dropout"] == 0:
            return "FN (False Negative)"
        if row["Actual_Dropout"] == 0 and row["Predicted_Dropout"] == 1:
            return "FP (False Positive)"
        return "Correct"

    error_df["Error_Type"] = error_df.apply(_classify_error, axis=1)
    error_df.to_csv(ERROR_CSV_PATH, index=False)

    return results_df