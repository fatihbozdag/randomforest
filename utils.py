"""
Utility functions for Water Quality Analysis System
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import json
import pickle
from pathlib import Path

def setup_logging(level: str = "INFO", log_file: str = "water_quality_analysis.log") -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def classify_water_quality(score: float) -> str:
    """
    Classify water quality based on fuzzy score using official BCG thresholds
    
    Args:
        score: Water quality score (0-1)
    
    Returns:
        Quality classification string
    """
    if score >= 0.80:
        return "High (BCG1)"
    elif score >= 0.60:
        return "Good (BCG2)"
    elif score >= 0.40:
        return "Moderate (BCG3)"
    elif score >= 0.20:
        return "Poor (BCG4)"
    else:
        return "Bad (BCG5)"

def classify_water_quality_batch(scores: np.ndarray) -> List[str]:
    """Classify water quality for multiple scores"""
    return [classify_water_quality(score) for score in scores]

def calculate_statistics(data: pd.DataFrame, target_col: str = None) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics for dataset
    
    Args:
        data: Input dataframe
        target_col: Target column name (optional)
    
    Returns:
        Dictionary with statistics
    """
    stats = {
        "shape": data.shape,
        "columns": list(data.columns),
        "dtypes": data.dtypes.to_dict(),
        "missing_values": data.isnull().sum().to_dict(),
        "missing_percentage": (data.isnull().sum() / len(data) * 100).to_dict(),
        "numeric_columns": data.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_columns": data.select_dtypes(include=['object']).columns.tolist()
    }
    
    # Numeric statistics
    if len(stats["numeric_columns"]) > 0:
        numeric_data = data[stats["numeric_columns"]]
        stats["numeric_stats"] = {
            "mean": numeric_data.mean().to_dict(),
            "std": numeric_data.std().to_dict(),
            "min": numeric_data.min().to_dict(),
            "max": numeric_data.max().to_dict(),
            "median": numeric_data.median().to_dict(),
            "skewness": numeric_data.skew().to_dict(),
            "kurtosis": numeric_data.kurtosis().to_dict()
        }
    
    # Target variable statistics
    if target_col and target_col in data.columns:
        target_data = data[target_col].dropna()
        stats["target_stats"] = {
            "mean": target_data.mean(),
            "std": target_data.std(),
            "min": target_data.min(),
            "max": target_data.max(),
            "median": target_data.median(),
            "unique_values": target_data.nunique(),
            "value_counts": target_data.value_counts().to_dict()
        }
    
    return stats

def detect_outliers(data: pd.Series, method: str = "iqr", threshold: float = 1.5) -> np.ndarray:
    """
    Detect outliers in data series
    
    Args:
        data: Input data series
        method: Outlier detection method ('iqr', 'zscore', 'isolation_forest')
        threshold: Threshold for outlier detection
    
    Returns:
        Boolean array indicating outliers
    """
    if method == "iqr":
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (data < lower_bound) | (data > upper_bound)
    
    elif method == "zscore":
        z_scores = np.abs((data - data.mean()) / data.std())
        return z_scores > threshold
    
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")

def create_output_directories(output_paths: Dict[str, str]) -> None:
    """Create output directories if they don't exist"""
    for path in output_paths.values():
        os.makedirs(path, exist_ok=True)

def save_model(model: Any, filepath: str) -> None:
    """Save model to file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def load_model(filepath: str) -> Any:
    """Load model from file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def export_results(results: Dict[str, Any], output_path: str, format: str = "json") -> None:
    """
    Export results to file
    
    Args:
        results: Results dictionary
        output_path: Output file path
        format: Export format ('json', 'csv', 'excel')
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if format == "json":
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    elif format == "csv":
        if isinstance(results, dict):
            # Convert dict to DataFrame if possible
            if all(isinstance(v, (int, float, str)) for v in results.values()):
                pd.DataFrame([results]).to_csv(output_path, index=False)
            else:
                # Handle nested dictionaries
                flattened = flatten_dict(results)
                pd.DataFrame([flattened]).to_csv(output_path, index=False)
        else:
            pd.DataFrame(results).to_csv(output_path, index=False)
    
    elif format == "excel":
        if isinstance(results, dict):
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                for sheet_name, data in results.items():
                    if isinstance(data, pd.DataFrame):
                        data.to_excel(writer, sheet_name=sheet_name, index=False)
                    else:
                        pd.DataFrame([data]).to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            pd.DataFrame(results).to_excel(output_path, index=False)

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """Flatten nested dictionary"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def calculate_prediction_intervals(predictions: np.ndarray, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate prediction intervals using bootstrap method
    
    Args:
        predictions: Array of predictions
        confidence: Confidence level (0-1)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(predictions, lower_percentile, axis=0)
    upper_bound = np.percentile(predictions, upper_percentile, axis=0)
    
    return lower_bound, upper_bound

def calculate_confidence_intervals(mean: float, std: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence intervals for mean
    
    Args:
        mean: Sample mean
        std: Sample standard deviation
        n: Sample size
        confidence: Confidence level (0-1)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    from scipy import stats
    
    alpha = 1 - confidence
    t_value = stats.t.ppf(1 - alpha/2, df=n-1)
    margin_of_error = t_value * (std / np.sqrt(n))
    
    return mean - margin_of_error, mean + margin_of_error

def format_time_duration(seconds: float) -> str:
    """Format time duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"

def validate_data_quality(data: pd.DataFrame, required_columns: List[str] = None) -> Dict[str, Any]:
    """
    Validate data quality and return issues
    
    Args:
        data: Input dataframe
        required_columns: List of required columns
    
    Returns:
        Dictionary with validation results
    """
    issues = {
        "missing_columns": [],
        "empty_dataframe": False,
        "all_null_columns": [],
        "duplicate_rows": 0,
        "data_types": {},
        "range_issues": {}
    }
    
    # Check if dataframe is empty
    if data.empty:
        issues["empty_dataframe"] = True
        return issues
    
    # Check required columns
    if required_columns:
        missing_cols = [col for col in required_columns if col not in data.columns]
        issues["missing_columns"] = missing_cols
    
    # Check for all-null columns
    null_columns = data.columns[data.isnull().all()].tolist()
    issues["all_null_columns"] = null_columns
    
    # Check for duplicate rows
    issues["duplicate_rows"] = data.duplicated().sum()
    
    # Check data types
    for col in data.columns:
        issues["data_types"][col] = str(data[col].dtype)
    
    # Check for range issues in numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        col_data = data[col].dropna()
        if len(col_data) > 0:
            if col_data.min() < 0 or col_data.max() > 1:
                issues["range_issues"][col] = {
                    "min": col_data.min(),
                    "max": col_data.max(),
                    "expected_range": "0-1"
                }
    
    return issues

def generate_timestamp() -> str:
    """Generate timestamp string for file naming"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def safe_filename(filename: str) -> str:
    """Convert filename to safe format"""
    import re
    # Remove or replace unsafe characters
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    return safe_name

def print_progress_bar(iteration: int, total: int, prefix: str = '', suffix: str = '', length: int = 50) -> None:
    """Print progress bar to console"""
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='')
    if iteration == total:
        print()

def calculate_feature_correlations(data: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
    """
    Calculate feature correlations
    
    Args:
        data: Input dataframe
        target_col: Target column name (optional)
    
    Returns:
        Correlation matrix
    """
    if target_col and target_col in data.columns:
        correlations = data.corr()[target_col].sort_values(ascending=False)
        return correlations
    else:
        return data.corr()

def check_multicollinearity(data: pd.DataFrame, threshold: float = 0.95) -> Dict[str, List[str]]:
    """
    Check for multicollinearity in features
    
    Args:
        data: Input dataframe
        threshold: Correlation threshold for multicollinearity
    
    Returns:
        Dictionary with highly correlated feature pairs
    """
    corr_matrix = data.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    high_corr_pairs = {}
    for column in upper_tri.columns:
        high_corr = upper_tri[column][upper_tri[column] > threshold].index.tolist()
        if high_corr:
            high_corr_pairs[column] = high_corr 