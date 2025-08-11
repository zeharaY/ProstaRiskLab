import sys
import os
# Ensure src is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import yaml
from data.clean_data import main as clean_main
from features.build_features import main as feat_main
from utils.loggers import setup_logger
from models.models import run_all_models

logger = setup_logger("pipeline")


def main():
    logger.info("🚀 Starting full pipeline")

    # Phase 1: Clean raw data
    logger.info("🧹 Cleaning raw data...")
    clean_main()

    # Phase 2: Feature engineering
    logger.info("🛠️ Feature engineering...")
    feat_main()

    # Phase 3: Model training + stacking
    logger.info("🤖 Model stacking...")
    # Load features and target
    X = pd.read_csv("data/features/X_train.csv").values
    y = pd.read_csv("data/features/y_train.csv").values.ravel()
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    run_all_models(X, y, config)

    logger.info("🏁 Pipeline completed successfully")

if __name__ == "__main__":
    main()