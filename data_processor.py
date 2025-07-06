"""
Data processing module for Water Quality Analysis System
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import warnings
import logging

from config import config
from utils import (
    setup_logging, calculate_statistics, detect_outliers, 
    validate_data_quality, check_multicollinearity
)

class WaterQualityDataProcessor:
    """
    Comprehensive data processing class for water quality analysis
    """
    
    def __init__(self, random_seed: int = None, log_level: str = "INFO"):
        """
        Initialize data processor
        
        Args:
            random_seed: Random seed for reproducibility
            log_level: Logging level
        """
        self.random_seed = random_seed or config.get("data.random_seed", 42)
        self.logger = setup_logging(log_level)
        
        # Data storage
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_val = None
        self.y_val = None
        
        # Preprocessing components
        self.scaler = None
        self.imputer = None
        self.feature_selector = None
        self.pca = None
        self.preprocessing_pipeline = None
        
        # Data statistics
        self.feature_statistics = {}
        self.data_quality_report = {}
        self.feature_names = []
        self.target_column = None
        
        # Configuration
        self.normalize_features = config.get("data.normalize_features", True)
        self.handle_outliers = config.get("data.handle_outliers", True)
        self.outlier_threshold = config.get("data.outlier_threshold", 3.0)
        self.test_size = config.get("data.test_size", 0.2)
        self.validation_size = config.get("data.validation_size", 0.1)
        
        # Feature engineering settings
        self.polynomial_degree = config.get("feature_engineering.polynomial_degree", 2)
        self.interaction_terms = config.get("feature_engineering.interaction_terms", True)
        self.pca_components = config.get("feature_engineering.pca_components", None)
        self.feature_selection = config.get("feature_engineering.feature_selection", True)
        self.correlation_threshold = config.get("feature_engineering.correlation_threshold", 0.95)
        
        np.random.seed(self.random_seed)
        self.logger.info(f"Data processor initialized with random seed: {self.random_seed}")
    
    def load_data(self, train_path: str, test_path: str) -> 'WaterQualityDataProcessor':
        """
        Load training and test datasets
        
        Args:
            train_path: Path to training data file
            test_path: Path to test data file
            
        Returns:
            Self for method chaining
        """
        self.logger.info("Loading datasets...")
        
        try:
            # Load training data
            self.train_data = pd.read_excel(train_path)
            self.logger.info(f"Training data loaded: {self.train_data.shape}")
            
            # Load test data
            self.test_data = pd.read_excel(test_path)
            self.logger.info(f"Test data loaded: {self.test_data.shape}")
            
            # Store original column names
            self.original_train_columns = list(self.train_data.columns)
            self.original_test_columns = list(self.test_data.columns)
            
            # Identify target column
            self._identify_target_column()
            
            # Validate data quality
            self._validate_data_quality()
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
        
        return self
    
    def _identify_target_column(self) -> None:
        """Identify target column in training data"""
        # Look for common target column names
        target_candidates = ['FUZZY', 'fuzzy', 'target', 'Target', 'TARGET', 'quality', 'Quality']
        
        for col in target_candidates:
            if col in self.train_data.columns:
                self.target_column = col
                self.logger.info(f"Target column identified: {self.target_column}")
                break
        
        if self.target_column is None:
            self.logger.warning("No target column found. Assuming last column is target.")
            self.target_column = self.train_data.columns[-1]
    
    def _validate_data_quality(self) -> None:
        """Validate data quality and generate report"""
        self.logger.info("Validating data quality...")
        
        # Validate training data
        train_required = ['TP', 'EC', 'DO', 'TIT', 'tıt', self.target_column]
        train_issues = validate_data_quality(
            self.train_data, 
            required_columns=train_required
        )
        
        # Validate test data (accept raw or normalized columns)
        test_required = ['TP_norm', 'EC_norm', 'DO_norm', 'tıt_norm', 'TP', 'EC', 'DO', 'TIT', 'tıt']
        test_issues = validate_data_quality(
            self.test_data,
            required_columns=test_required
        )
        
        self.data_quality_report = {
            'train_data': train_issues,
            'test_data': test_issues,
            'validation_passed': not any([
                train_issues['empty_dataframe'],
                test_issues['empty_dataframe'],
                len(train_issues['missing_columns']) > 0,
                len(test_issues['missing_columns']) > 0
            ])
        }
        
        if not self.data_quality_report['validation_passed']:
            self.logger.warning("Data quality issues detected:")
            self.logger.warning(f"Train data issues: {train_issues}")
            self.logger.warning(f"Test data issues: {test_issues}")
        else:
            self.logger.info("Data quality validation passed")
    
    def preprocess_data(self) -> 'WaterQualityDataProcessor':
        """
        Preprocess the loaded data
        
        Returns:
            Self for method chaining
        """
        self.logger.info("Starting data preprocessing...")
        
        # Standardize column names
        self._standardize_column_names()
        
        # Handle missing values
        self._handle_missing_values()
        
        # Handle outliers
        if self.handle_outliers:
            self._handle_outliers()
        
        # Feature engineering
        self._engineer_features()
        
        # Split data
        self._split_data()
        
        # Create preprocessing pipeline
        self._create_preprocessing_pipeline()
        
        # Apply preprocessing
        self._apply_preprocessing()
        
        # Generate feature statistics
        self._generate_feature_statistics()
        
        self.logger.info("Data preprocessing completed")
        return self
    
    def _standardize_column_names(self) -> None:
        """Standardize column names across datasets"""
        # Standardize training data column names
        train_rename_map = {
            'İSTASYON': 'station',
            'TP': 'tp',
            'EC': 'ec',
            'DO': 'do',
            'TIT': 'tit',
            'tıt': 'tit',  # handle lowercase Turkish dotless i
            'FUZZY': 'fuzzy'
        }
        
        # Standardize test data column names
        test_rename_map = {
            'istasyon': 'station',
            'TP_norm': 'tp',
            'EC_norm': 'ec',
            'DO_norm': 'do',
            'tıt_norm': 'tit',
            'TP': 'tp',       # allow raw (unnormalized) test columns
            'EC': 'ec',
            'DO': 'do',
            'TIT': 'tit',
            'tıt': 'tit',
            'Fuzzy': 'fuzzy'
        }
        
        self.train_data = self.train_data.rename(columns=train_rename_map)
        self.test_data = self.test_data.rename(columns=test_rename_map)
        
        # Update target column name after standardization
        if self.target_column in train_rename_map:
            self.target_column = train_rename_map[self.target_column]
        
        self.logger.info("Column names standardized")
    
    def _handle_missing_values(self) -> None:
        """Handle missing values in the datasets"""
        # Check for missing values
        train_missing = self.train_data.isnull().sum()
        test_missing = self.test_data.isnull().sum()
        
        if train_missing.sum() > 0:
            self.logger.info(f"Missing values in training data: {train_missing[train_missing > 0].to_dict()}")
        
        if test_missing.sum() > 0:
            self.logger.info(f"Missing values in test data: {test_missing[test_missing > 0].to_dict()}")
        
        # For now, we'll use simple imputation (mean for numeric, mode for categorical)
        # In a production system, you might want more sophisticated imputation
        numeric_columns = self.train_data.select_dtypes(include=[np.number]).columns
        categorical_columns = self.train_data.select_dtypes(include=['object']).columns
        
        # Exclude target column from imputation in test data (since test data shouldn't have target values)
        test_numeric_columns = [col for col in numeric_columns if col != self.target_column]
        test_categorical_columns = [col for col in categorical_columns if col != self.target_column]
        
        # Impute numeric columns with mean
        if len(numeric_columns) > 0:
            self.train_data[numeric_columns] = self.train_data[numeric_columns].fillna(
                self.train_data[numeric_columns].mean()
            )
            # Only impute feature columns in test data, not target column
            if len(test_numeric_columns) > 0:
                self.test_data[test_numeric_columns] = self.test_data[test_numeric_columns].fillna(
                    self.train_data[test_numeric_columns].mean()
                )
        
        # Impute categorical columns with mode
        if len(categorical_columns) > 0:
            for col in categorical_columns:
                if col in self.train_data.columns and not self.train_data[col].empty:
                    mode_value = self.train_data[col].mode()[0] if len(self.train_data[col].mode()) > 0 else 'unknown'
                    self.train_data[col] = self.train_data[col].fillna(mode_value)
                    if col in test_categorical_columns and col in self.test_data.columns:
                        self.test_data[col] = self.test_data[col].fillna(mode_value)
        
        self.logger.info("Missing values handled")
    
    def _handle_outliers(self) -> None:
        """Handle outliers in numeric features"""
        numeric_columns = self.train_data.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != self.target_column]
        
        outlier_counts = {}
        
        for col in numeric_columns:
            outliers = detect_outliers(
                self.train_data[col], 
                method="iqr", 
                threshold=self.outlier_threshold
            )
            outlier_count = outliers.sum()
            outlier_counts[col] = outlier_count
            
            if outlier_count > 0:
                # Replace outliers with median
                median_value = self.train_data[col].median()
                self.train_data.loc[outliers, col] = median_value
                self.logger.info(f"Replaced {outlier_count} outliers in {col} with median")
        
        if any(outlier_counts.values()):
            self.logger.info(f"Outlier handling summary: {outlier_counts}")
        else:
            self.logger.info("No outliers detected")
    
    def _engineer_features(self) -> None:
        """Engineer additional features"""
        # Create polynomial features
        if self.polynomial_degree > 1:
            self._create_polynomial_features()
        
        # Create interaction terms
        if self.interaction_terms:
            self._create_interaction_terms()
        
        # Create station-based features
        self._create_station_features()
        
        self.logger.info("Feature engineering completed")
    
    def _create_polynomial_features(self) -> None:
        """Create polynomial features"""
        from sklearn.preprocessing import PolynomialFeatures
        
        numeric_columns = self.train_data.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != self.target_column]
        
        if len(numeric_columns) > 0:
            poly = PolynomialFeatures(degree=self.polynomial_degree, include_bias=False)
            
            # Apply to training data
            poly_features_train = poly.fit_transform(self.train_data[numeric_columns])
            poly_feature_names = poly.get_feature_names_out(numeric_columns)
            
            # Add polynomial features to training data
            for i, name in enumerate(poly_feature_names):
                if name not in self.train_data.columns:  # Avoid duplicates
                    self.train_data[f'poly_{name}'] = poly_features_train[:, i]
            
            # Apply to test data
            poly_features_test = poly.transform(self.test_data[numeric_columns])
            for i, name in enumerate(poly_feature_names):
                if name not in self.test_data.columns:  # Avoid duplicates
                    self.test_data[f'poly_{name}'] = poly_features_test[:, i]
    
    def _create_interaction_terms(self) -> None:
        """Create interaction terms between features"""
        numeric_columns = self.train_data.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != self.target_column]
        
        if len(numeric_columns) >= 2:
            for i in range(len(numeric_columns)):
                for j in range(i + 1, len(numeric_columns)):
                    col1, col2 = numeric_columns[i], numeric_columns[j]
                    interaction_name = f'interaction_{col1}_{col2}'
                    
                    self.train_data[interaction_name] = self.train_data[col1] * self.train_data[col2]
                    self.test_data[interaction_name] = self.test_data[col1] * self.test_data[col2]
    
    def _create_station_features(self) -> None:
        """Create station-based features"""
        if 'station' in self.train_data.columns:
            # Create station encoding
            from sklearn.preprocessing import LabelEncoder
            
            le = LabelEncoder()
            self.train_data['station_encoded'] = le.fit_transform(self.train_data['station'])
            
            # Handle unseen labels in test data
            test_stations = self.test_data['station'].values
            test_encoded = []
            for station in test_stations:
                if station in le.classes_:
                    test_encoded.append(le.transform([station])[0])
                else:
                    # Assign a default value for unseen stations (use -1 or max+1)
                    test_encoded.append(-1)
            self.test_data['station_encoded'] = test_encoded
            
            # Create station statistics
            station_stats = self.train_data.groupby('station')[self.target_column].agg(['mean', 'std']).reset_index()
            station_stats.columns = ['station', 'station_mean_quality', 'station_std_quality']
            
            self.train_data = self.train_data.merge(station_stats, on='station', how='left')
            self.test_data = self.test_data.merge(station_stats, on='station', how='left')
            
            # Fill NaN values in test data with overall statistics
            overall_mean = self.train_data[self.target_column].mean()
            overall_std = self.train_data[self.target_column].std()
            
            self.test_data['station_mean_quality'] = self.test_data['station_mean_quality'].fillna(overall_mean)
            self.test_data['station_std_quality'] = self.test_data['station_std_quality'].fillna(overall_std)
    
    def _split_data(self) -> None:
        """Split data into train, validation, and test sets"""
        # Prepare features and target
        feature_columns = [col for col in self.train_data.columns 
                          if col not in [self.target_column, 'station']]
        
        X = self.train_data[feature_columns]
        y = self.train_data[self.target_column]
        
        # Split into train and temporary test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_seed, stratify=None
        )
        
        # Split temporary test into validation and final test
        val_size_adjusted = self.validation_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=self.random_seed, stratify=None
        )
        
        # Store splits
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.feature_names = feature_columns
        
        self.logger.info(f"Data split completed:")
        self.logger.info(f"  Training set: {X_train.shape}")
        self.logger.info(f"  Validation set: {X_val.shape}")
        self.logger.info(f"  Test set: {X_test.shape}")
    
    def _create_preprocessing_pipeline(self) -> None:
        """Create preprocessing pipeline"""
        # Define numeric features
        numeric_features = self.X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        # Create preprocessing steps
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler() if self.normalize_features else 'passthrough')
        ])
        
        # Create column transformer
        self.preprocessing_pipeline = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features)
            ],
            remainder='passthrough'
        )
        
        self.logger.info("Preprocessing pipeline created")
    
    def _apply_preprocessing(self) -> None:
        """Apply preprocessing to the data"""
        # Fit and transform training data
        self.X_train_processed = self.preprocessing_pipeline.fit_transform(self.X_train)
        self.X_val_processed = self.preprocessing_pipeline.transform(self.X_val)
        self.X_test_processed = self.preprocessing_pipeline.transform(self.X_test)
        
        # Convert to DataFrame for easier handling
        feature_names_processed = self.preprocessing_pipeline.get_feature_names_out()
        self.X_train_processed = pd.DataFrame(self.X_train_processed, columns=feature_names_processed)
        self.X_val_processed = pd.DataFrame(self.X_val_processed, columns=feature_names_processed)
        self.X_test_processed = pd.DataFrame(self.X_test_processed, columns=feature_names_processed)
        
        # Update feature names
        self.feature_names_processed = feature_names_processed.tolist()
        
        self.logger.info("Preprocessing applied to all datasets")
    
    def _generate_feature_statistics(self) -> None:
        """Generate comprehensive feature statistics"""
        self.feature_statistics = {
            'original_features': calculate_statistics(self.X_train),
            'processed_features': calculate_statistics(self.X_train_processed),
            'target_statistics': calculate_statistics(pd.DataFrame({'target': self.y_train})),
            'correlations': self.X_train_processed.corrwith(self.y_train).to_dict() if len(self.X_train_processed) > 0 else {},
            'multicollinearity': check_multicollinearity(self.X_train_processed, self.correlation_threshold) if len(self.X_train_processed) > 0 else {}
        }
        
        self.logger.info("Feature statistics generated")
    
    def get_feature_statistics(self) -> Dict[str, Any]:
        """Get comprehensive feature statistics"""
        return self.feature_statistics
    
    def get_data_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, 
                                      pd.Series, pd.Series, pd.Series]:
        """Get processed data splits"""
        return (self.X_train_processed, self.X_val_processed, self.X_test_processed,
                self.y_train, self.y_val, self.y_test)
    
    def get_test_data_processed(self) -> pd.DataFrame:
        """Get processed test data for predictions"""
        # Process the original test data
        feature_columns = [col for col in self.test_data.columns if col != 'station']
        X_test_original = self.test_data[feature_columns]
        
        # Apply preprocessing
        X_test_processed = self.preprocessing_pipeline.transform(X_test_original)
        feature_names_processed = self.preprocessing_pipeline.get_feature_names_out()
        
        return pd.DataFrame(X_test_processed, columns=feature_names_processed)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names"""
        return self.feature_names_processed
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get comprehensive data summary"""
        return {
            'train_shape': self.train_data.shape,
            'test_shape': self.test_data.shape,
            'feature_names': self.feature_names_processed,
            'target_column': self.target_column,
            'data_quality_report': self.data_quality_report,
            'feature_statistics': self.feature_statistics,
            'preprocessing_applied': True
        } 