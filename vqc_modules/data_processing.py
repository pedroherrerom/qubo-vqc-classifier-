"""data_processing.py — Data preprocessing utilities.

All transformers are fitted on the training split and applied to the test
split to prevent information leakage.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

if TYPE_CHECKING:
    pass

logger = logging.getLogger("vqc")

# Angle-encoding requires features in [0, π]
_ANGLE_MIN = 0.0
_ANGLE_MAX = float(np.pi)

# Max unique values allowed for high-cardinality object columns
_MAX_CARDINALITY = 30


def _drop_irrelevant(
    df: pd.DataFrame, id_cols: list[str]
) -> pd.DataFrame:
    """Drop explicitly listed ID columns if present."""
    to_drop = [c for c in id_cols if c in df.columns]
    if to_drop:
        logger.debug("Dropping id_cols: %s", to_drop)
    return df.drop(columns=to_drop)


def _drop_zero_variance(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns with zero variance (constant features)."""
    numeric = df.select_dtypes(include="number")
    zero_var = numeric.columns[numeric.std() == 0].tolist()
    if zero_var:
        logger.debug("Dropping zero-variance columns: %s", zero_var)
    return df.drop(columns=zero_var)


def _drop_high_cardinality(df: pd.DataFrame) -> pd.DataFrame:
    """Drop object columns with more than _MAX_CARDINALITY unique values."""
    obj_cols = df.select_dtypes(include="object").columns
    high_card = [c for c in obj_cols if df[c].nunique() > _MAX_CARDINALITY]
    if high_card:
        logger.debug("Dropping high-cardinality object columns: %s", high_card)
    return df.drop(columns=high_card)


def _label_encode(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Label-encode remaining categorical columns (fitted on train)."""
    obj_cols = train.select_dtypes(include="object").columns.tolist()
    for col in obj_cols:
        le = LabelEncoder()
        train = train.copy()
        test = test.copy()
        train[col] = le.fit_transform(train[col].astype(str))
        # Handle unseen labels in test set
        mapping = {label: idx for idx, label in enumerate(le.classes_)}
        test[col] = test[col].astype(str).map(mapping).fillna(-1).astype(int)
    return train, test


class DataPreprocessor:
    """
    Fit-transform pipeline for the QUBO-VQC dataset.

    Parameters
    ----------
    id_cols : list[str]
        Columns to drop unconditionally (IDs, SMILES, etc.).
    target : str
        Name of the binary label column.
    """

    def __init__(self, id_cols: list[str], target: str) -> None:
        self.id_cols = id_cols
        self.target = target
        self._imputer: SimpleImputer | None = None
        self._scaler: MinMaxScaler | None = None
        self._feature_cols: list[str] = []

    # ------------------------------------------------------------------
    def fit_transform(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """
        Preprocess train and test dataframes.

        Returns
        -------
        X_train, X_test : np.ndarray  (n_samples, n_features), float64
        y_train, y_test : np.ndarray  (n_samples,), int
        feature_names   : list[str]
        """
        # Separate labels
        y_train = train_df[self.target].values.astype(int)
        y_test = test_df[self.target].values.astype(int)
        train_df = train_df.drop(columns=[self.target])
        test_df = test_df.drop(columns=[self.target])

        # Structural cleanup
        train_df = _drop_irrelevant(train_df, self.id_cols)
        test_df = _drop_irrelevant(test_df, self.id_cols)

        train_df = _drop_zero_variance(train_df)
        # Align test to train columns after zero-var drop
        test_df = test_df.reindex(columns=train_df.columns)

        train_df = _drop_high_cardinality(train_df)
        test_df = test_df.reindex(columns=train_df.columns)

        train_df, test_df = _label_encode(train_df, test_df)

        self._feature_cols = train_df.columns.tolist()
        logger.info("Features after cleaning: %d", len(self._feature_cols))

        X_train = train_df.values.astype(float)
        X_test = test_df.values.astype(float)

        # Median imputation (fit on train)
        self._imputer = SimpleImputer(strategy="median")
        X_train = self._imputer.fit_transform(X_train)
        X_test = self._imputer.transform(X_test)

        # MinMax scaling to [0, π]
        self._scaler = MinMaxScaler(feature_range=(_ANGLE_MIN, _ANGLE_MAX))
        X_train = self._scaler.fit_transform(X_train)
        X_test = self._scaler.transform(X_test)

        logger.info(
            "X_train shape: %s  X_test shape: %s", X_train.shape, X_test.shape
        )
        return X_train, X_test, y_train, y_test, self._feature_cols

    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply fitted pipeline to a new dataframe (no label column)."""
        if self._imputer is None or self._scaler is None:
            raise RuntimeError("Preprocessor has not been fitted yet.")
        df = _drop_irrelevant(df, self.id_cols)
        df = df.reindex(columns=self._feature_cols).copy()
        X = df.values.astype(float)
        X = self._imputer.transform(X)
        return self._scaler.transform(X)


def load_datasets(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train/test DataFrames according to the pipeline config.

    Supports two modes:
    - Dual-CSV  : ``train_path`` + ``test_path``
    - Single-CSV: ``csv_path`` split by ``test_size``
    """
    if config.get("train_path") and config.get("test_path"):
        logger.info("Dual-CSV mode: %s / %s", config["train_path"], config["test_path"])
        train_df = pd.read_csv(config["train_path"])
        test_df = pd.read_csv(config["test_path"])
    elif config.get("csv_path"):
        from sklearn.model_selection import train_test_split  # noqa: PLC0415

        logger.info("Single-CSV mode: %s", config["csv_path"])
        df = pd.read_csv(config["csv_path"])
        train_df, test_df = train_test_split(
            df,
            test_size=config.get("test_size", 0.2),
            stratify=df[config["target"]],
            random_state=config.get("seed", 42),
        )
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
    else:
        raise ValueError("Provide 'train_path'+'test_path' or 'csv_path' in config.")

    logger.info(
        "Raw shapes — train: %s  test: %s", train_df.shape, test_df.shape
    )
    return train_df, test_df
