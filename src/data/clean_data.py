import os
import pandas as pd
import utils.loggers

from utils.loggers import setup_logger

# 📁 Setup paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)

# 🧾 Initialize logger
logger = setup_logger(PROCESSED_DIR, log_file='cleaning.log')

def clean_dataframe(df):
    """Clean dataframe: handle missing values, detect and treat outliers, strip strings, drop duplicates."""
    try:
        logger.info("🧼 Cleaning dataframe")
        # Handle missing values
        missing_before = df.isnull().sum().sum()
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                median = df[col].median()
                df[col] = df[col].fillna(median)
            elif df[col].dtype == 'object':
                mode = df[col].mode()
                if not mode.empty:
                    df[col] = df[col].fillna(mode[0])
        missing_after = df.isnull().sum().sum()
        logger.info(f"🩹 Missing values handled: {missing_before} → {missing_after}")

        # Strip whitespace from string columns
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip()

        # Detect and treat outliers (IQR method)
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            if outliers > 0:
                logger.info(f"🚨 Outliers detected in '{col}': {outliers} (capped)")
                df[col] = df[col].clip(lower, upper)

        # Drop duplicates
        before_dupes = len(df)
        df = df.drop_duplicates()
        after_dupes = len(df)
        logger.info(f"🗑️ Duplicates dropped: {before_dupes - after_dupes}")

        logger.info("✅ Dataframe cleaned")
        return df
    except Exception as e:
        logger.error(f"❌ Error during cleaning: {e}")
        raise

def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    logger.info("🚀 Starting data cleaning pipeline")

    try:
        for filename in os.listdir(RAW_DIR):
            if filename.endswith('.csv'):
                raw_path = os.path.join(RAW_DIR, filename)
                processed_path = os.path.join(PROCESSED_DIR, filename)

                logger.info(f"📂 Reading file: {filename}")
                df = pd.read_csv(raw_path)

                df_clean = clean_dataframe(df)
                df_clean.to_csv(processed_path, index=False)
                logger.info(f"💾 Saved cleaned data to: {processed_path}")

        logger.info("🏁 Data cleaning completed")

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise

if __name__ == '__main__':
    main()