import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import utils.loggers
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from utils.loggers import setup_logger

# 📁 Setup
logger = setup_logger("features")
FEATURES_DIR = "data/features"
DATA_PATH = "data/processed/prostate_cancer_prediction.csv"
os.makedirs(FEATURES_DIR, exist_ok=True)

# 🔍 Load Data
def load_data(path):
    logger.info(f"📥 Loading data from {path}")
    return pd.read_csv(path)

# 🧬 Domain-Specific Features
def create_domain_features(df):
    logger.info("🧬 Creating domain-specific features")
    if "PSA" in df.columns and "Prostate_Volume" in df.columns:
        df["PSA_Density"] = df["PSA"] / (df["Prostate_Volume"] + 1e-5)
    return df

# 🔐 Encode Target
def encode_target(df, label_col="Biopsy_Result"):
    logger.info("🔐 Encoding target")
    le = LabelEncoder()
    df["target"] = le.fit_transform(df[label_col])
    df.drop(columns=[label_col], inplace=True)
    joblib.dump(le, os.path.join(FEATURES_DIR, "label_encoder.pkl"))
    return df

# 🔠 Encode Categorical Features
def encode_categoricals(df):
    cat_cols = df.select_dtypes(include="object").columns
    if len(cat_cols) > 0:
        df = pd.get_dummies(df, columns=cat_cols)
        logger.info(f"🔠 One-hot encoded: {list(cat_cols)}")
    return df

# 📊 Scale Numerical Features
def scale_numericals(df):
    num_cols = df.select_dtypes(include="number").columns.drop("target")
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    joblib.dump(scaler, os.path.join(FEATURES_DIR, "scaler.pkl"))
    logger.info(f"📊 Scaled numerical columns: {list(num_cols)}")
    return df

# 📈 Plot Correlation Matrix
def plot_correlation_matrix(df, save_path):
    logger.info("📈 Plotting correlation matrix")
    num_df = df.select_dtypes(include="number").drop(columns=["target"], errors="ignore")
    corr = num_df.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar=True)
    plt.title("Feature Correlation Matrix", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"🖼️ Correlation matrix saved to {save_path}")

# 🔍 Drop Highly Correlated Features
def drop_correlated_features(df, threshold=0.9):
    logger.info(f"🔍 Dropping features with correlation > {threshold}")
    corr_matrix = df.drop(columns=["target"]).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    df.drop(columns=to_drop, inplace=True)
    logger.info(f"🧹 Dropped correlated features: {to_drop}")
    return df

# 📌 Feature Selection via L1 Regularization
def select_features_l1(df):
    logger.info("🧮 Selecting features via L1 regularization")
    X = df.drop(columns=["target"])
    y = df["target"]
    model = LogisticRegression(penalty="l1", solver="liblinear", random_state=42)
    selector = SelectFromModel(model)
    selector.fit(X, y)
    selected_cols = X.columns[selector.get_support()]
    df = df[selected_cols.tolist() + ["target"]]
    logger.info(f"📌 Selected features: {list(selected_cols)}")
    return df

# 📉 Dimensionality Reduction with PCA
def reduce_dimensionality(df, n_components=0.95):
    logger.info("📉 Applying PCA for dimensionality reduction")
    X = df.drop(columns=["target"])
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    joblib.dump(pca, os.path.join(FEATURES_DIR, "pca.pkl"))
    df_pca = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
    df_pca["target"] = df["target"].values
    logger.info(f"✅ Reduced to {X_pca.shape[1]} principal components")
    return df_pca

# ✂️ Split and Save
def split_and_save(df):
    logger.info("✂️ Splitting and saving data")
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    X_train.to_csv(os.path.join(FEATURES_DIR, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(FEATURES_DIR, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(FEATURES_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(FEATURES_DIR, "y_test.csv"), index=False)
    logger.info("✅ Feature splits saved successfully")

# 🚀 Main Pipeline
def main():
    df = load_data(DATA_PATH)
    df = create_domain_features(df)
    df = encode_target(df)
    df = encode_categoricals(df)
    df = scale_numericals(df)
    plot_correlation_matrix(df, os.path.join(FEATURES_DIR, "correlation_matrix.png"))
    df = drop_correlated_features(df)
    df = select_features_l1(df)
    df = reduce_dimensionality(df)
    split_and_save(df)

if __name__ == "__main__":
    main()