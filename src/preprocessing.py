from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from src.utils import get_first_csv_path


class DomainFeatureEngineer(BaseEstimator, TransformerMixin):
    """Tạo biến phái sinh tri thức miền cho Luồng B (Advanced)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        # Tỷ lệ qua môn kỳ 1 và kỳ 2
        if "Curricular units 1st sem (enrolled)" in df.columns and "Curricular units 1st sem (approved)" in df.columns:
            denom1 = df["Curricular units 1st sem (enrolled)"].replace(0, np.nan)
            df["pass_rate_sem1"] = (df["Curricular units 1st sem (approved)"] / denom1).fillna(0.0)

        if "Curricular units 2nd sem (enrolled)" in df.columns and "Curricular units 2nd sem (approved)" in df.columns:
            denom2 = df["Curricular units 2nd sem (enrolled)"].replace(0, np.nan)
            df["pass_rate_sem2"] = (df["Curricular units 2nd sem (approved)"] / denom2).fillna(0.0)

        # Chênh lệch điểm trung bình 2 kỳ
        if "Curricular units 1st sem (grade)" in df.columns and "Curricular units 2nd sem (grade)" in df.columns:
            df["grade_diff"] = df["Curricular units 2nd sem (grade)"] - df["Curricular units 1st sem (grade)"]

        # Chỉ số rủi ro tài chính
        if "Tuition fees up to date" in df.columns and "Debtor" in df.columns:
            df["financial_risk"] = ((df["Tuition fees up to date"] == 0) | (df["Debtor"] == 1)).astype(int)

        return df


def load_and_clean_data(csv_path: Path | None = None) -> pd.DataFrame:
    """Đọc CSV tự nhận dấu phân cách, lọc bỏ Enrolled và chuẩn hóa cột mục tiêu Dropout."""
    if csv_path is None or not Path(csv_path).exists():
        csv_path = get_first_csv_path()

    # Tự động nhận diện dấu phân cách (; hoặc ,)
    with open(csv_path, "r", encoding="utf-8-sig", errors="replace") as f:
        sample = f.read(4096)
    sep = ";" if sample.count(";") > sample.count(",") else ","

    df = pd.read_csv(csv_path, sep=sep, encoding="utf-8-sig")
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]

    # Xóa cột ID nếu có
    id_cols = [c for c in df.columns if c.lower() in {"id", "student_id", "student id"}]
    if id_cols:
        df = df.drop(columns=id_cols)

    df = df.drop_duplicates().reset_index(drop=True)

    # Chuẩn hóa nhãn mục tiêu về nhị phân: 1 = Dropout, 0 = Graduate
    if "Target" in df.columns:
        df["Target"] = df["Target"].astype(str).str.strip()
        df = df[df["Target"].isin(["Dropout", "Graduate"])].copy()
        df["Dropout"] = (df["Target"] == "Dropout").astype(int)
        df = df.drop(columns=["Target"])
    elif "Dropout" in df.columns:
        df["Dropout"] = pd.to_numeric(df["Dropout"], errors="coerce").dropna().astype(int)
    else:
        raise ValueError("Dataset phải có cột 'Target' ('Dropout'/'Graduate') hoặc cột 'Dropout' (0/1).")

    return df.reset_index(drop=True)


def print_eda_overview(df: pd.DataFrame) -> None:
    print(f"Kích thước dữ liệu (sau khi lọc Enrolled): {df.shape[0]} dòng x {df.shape[1]} cột")
    print(f"Tổng số giá trị thiếu (Missing values): {int(df.isna().sum().sum())}")
    counts = df["Dropout"].value_counts()
    print(f"Phân phối lớp: Graduate (0) = {counts.get(0, 0)} | Dropout (1) = {counts.get(1, 0)}")


def split_train_test(df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42):
    X = df.drop(columns=["Dropout"])
    y = df["Dropout"]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def build_preprocessor_flow_a(X_sample: pd.DataFrame) -> ColumnTransformer:
    """Luồng A (Baseline): SimpleImputer + StandardScaler / OneHotEncoder."""
    num_cols = X_sample.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_sample.select_dtypes(exclude=[np.number]).columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )


def build_preprocessor_flow_b(X_sample: pd.DataFrame) -> Pipeline:
    """Luồng B (Advanced): Feature Engineering + RobustScaler / OneHotEncoder."""
    engineered = DomainFeatureEngineer().fit_transform(X_sample)
    num_cols = engineered.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = engineered.select_dtypes(exclude=[np.number]).columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    ct = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )

    return Pipeline([
        ("feat_eng", DomainFeatureEngineer()),
        ("col_trans", ct),
    ])