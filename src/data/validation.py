import pandas as pd
from sklearn.exceptions import NotFittedError

def validate_features(df: pd.DataFrame, allowed_features: list) -> None:
    """Validate dataframe against feature constraints"""
    missing = [f for f in allowed_features if f not in df.columns]
    extra = [f for f in df.columns if f not in allowed_features]
    
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    if extra:
        raise ValueError(f"Disallowed features detected: {extra}")

def validate_model(model) -> None:
    """Verify model is properly trained"""
    if not hasattr(model, "feature_names_in_"):
        raise NotFittedError("Model appears untrained - missing feature names")