import pandas as pd
import numpy as np
import io
import json
import base64
from typing import Union, Dict, Any
import streamlit as st

class FileHandler:
    """
    Utility class for handling file uploads and data loading.
    """
    
    @staticmethod
    def load_file(uploaded_file) -> pd.DataFrame:
        """
        Load data from uploaded file (CSV or Excel).
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            pd.DataFrame: Loaded data
            
        Raises:
            Exception: If file cannot be loaded or is in unsupported format
        """
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                # Try different encodings and separators
                try:
                    # First try with UTF-8
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    # Try with latin-1 encoding
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='latin-1')
                except Exception:
                    # Try with different separator
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8')
                    
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Basic data validation
            if df.empty:
                raise ValueError("The uploaded file is empty")
            
            if df.shape[1] == 0:
                raise ValueError("No columns found in the file")
            
            # Clean column names (remove extra spaces, special characters)
            df.columns = df.columns.str.strip()
            
            # Try to infer datetime columns
            df = FileHandler._infer_datetime_columns(df)
            
            return df
            
        except Exception as e:
            raise Exception(f"Error loading file: {str(e)}")
    
    @staticmethod
    def _infer_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Attempt to automatically detect and convert datetime columns.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with datetime columns converted
        """
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to datetime if it looks like a date
                sample_values = df[col].dropna().head(100)
                if len(sample_values) > 0:
                    # Check if values look like dates
                    try:
                        # Try parsing a sample
                        pd.to_datetime(sample_values.iloc[0])
                        # If successful, convert the entire column
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    except:
                        continue
        
        return df
    
    @staticmethod
    def validate_data(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate uploaded data and return validation results.
        
        Args:
            df (pd.DataFrame): Dataframe to validate
            
        Returns:
            dict: Validation results
        """
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'info': []
        }
        
        # Check for minimum requirements
        if df.shape[0] < 2:
            validation_results['errors'].append("Dataset must have at least 2 rows")
            validation_results['is_valid'] = False
        
        if df.shape[1] < 1:
            validation_results['errors'].append("Dataset must have at least 1 column")
            validation_results['is_valid'] = False
        
        # Check for common issues
        if df.isnull().all().any():
            empty_cols = df.columns[df.isnull().all()].tolist()
            validation_results['warnings'].append(f"Empty columns found: {empty_cols}")
        
        missing_percentage = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        if missing_percentage > 50:
            validation_results['warnings'].append(f"High percentage of missing values: {missing_percentage:.1f}%")
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            validation_results['warnings'].append(f"Found {duplicate_count} duplicate rows")
        
        # Info about data types
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        categorical_cols = len(df.select_dtypes(include=['object']).columns)
        datetime_cols = len(df.select_dtypes(include=['datetime']).columns)
        
        validation_results['info'].append(f"Data types: {numeric_cols} numeric, {categorical_cols} categorical, {datetime_cols} datetime")
        
        return validation_results

class ExportHandler:
    """
    Utility class for exporting data and insights.
    """
    
    @staticmethod
    def to_csv(df: pd.DataFrame) -> str:
        """
        Convert dataframe to CSV string.
        
        Args:
            df (pd.DataFrame): Dataframe to export
            
        Returns:
            str: CSV string
        """
        return df.to_csv(index=False)
    
    @staticmethod
    def to_excel(df: pd.DataFrame) -> bytes:
        """
        Convert dataframe to Excel bytes.
        
        Args:
            df (pd.DataFrame): Dataframe to export
            
        Returns:
            bytes: Excel file bytes
        """
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Data')
        return output.getvalue()
    
    @staticmethod
    def to_json(df: pd.DataFrame) -> str:
        """
        Convert dataframe to JSON string.
        
        Args:
            df (pd.DataFrame): Dataframe to export
            
        Returns:
            str: JSON string
        """
        return df.to_json(orient='records', date_format='iso')
    
    @staticmethod
    def format_insights_for_export(insights: Dict[str, Any]) -> str:
        """
        Format AI insights for text export.
        
        Args:
            insights (dict): AI insights dictionary
            
        Returns:
            str: Formatted insights text
        """
        formatted_text = "AI Data Analysis Report\n"
        formatted_text += "=" * 50 + "\n\n"
        
        for insight_type, content in insights.items():
            formatted_text += f"{insight_type.upper()}\n"
            formatted_text += "-" * len(insight_type) + "\n"
            
            if isinstance(content, dict):
                for key, value in content.items():
                    formatted_text += f"\n{key}:\n{value}\n"
            else:
                formatted_text += f"\n{content}\n"
            
            formatted_text += "\n" + "=" * 30 + "\n\n"
        
        return formatted_text

class DataProfiler:
    """
    Utility class for comprehensive data profiling.
    """
    
    @staticmethod
    def generate_profile_report(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a comprehensive profile report of the dataset.
        
        Args:
            df (pd.DataFrame): Dataset to profile
            
        Returns:
            dict: Comprehensive profile report
        """
        profile = {
            'overview': DataProfiler._get_overview(df),
            'columns': DataProfiler._get_column_profiles(df),
            'correlations': DataProfiler._get_correlations(df),
            'missing_data': DataProfiler._get_missing_data_analysis(df),
            'duplicates': DataProfiler._get_duplicate_analysis(df)
        }
        
        return profile
    
    @staticmethod
    def _get_overview(df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic overview of the dataset."""
        return {
            'rows': len(df),
            'columns': len(df.columns),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object']).columns),
            'datetime_columns': len(df.select_dtypes(include=['datetime']).columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'missing_cells': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        }
    
    @staticmethod
    def _get_column_profiles(df: pd.DataFrame) -> Dict[str, Any]:
        """Get detailed profiles for each column."""
        profiles = {}
        
        for col in df.columns:
            col_data = df[col]
            profile = {
                'dtype': str(col_data.dtype),
                'non_null_count': col_data.count(),
                'null_count': col_data.isnull().sum(),
                'null_percentage': (col_data.isnull().sum() / len(df)) * 100,
                'unique_count': col_data.nunique(),
                'unique_percentage': (col_data.nunique() / len(df)) * 100
            }
            
            if col_data.dtype in ['int64', 'float64']:
                profile.update({
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'mean': col_data.mean(),
                    'median': col_data.median(),
                    'std': col_data.std(),
                    'skewness': col_data.skew(),
                    'kurtosis': col_data.kurtosis()
                })
            elif col_data.dtype == 'object':
                value_counts = col_data.value_counts()
                profile.update({
                    'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                    'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                    'least_frequent': value_counts.index[-1] if len(value_counts) > 0 else None,
                    'least_frequent_count': value_counts.iloc[-1] if len(value_counts) > 0 else 0
                })
            
            profiles[col] = profile
        
        return profiles
    
    @staticmethod
    def _get_correlations(df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate correlations between numeric columns."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return {'message': 'Not enough numeric columns for correlation analysis'}
        
        corr_matrix = numeric_df.corr()
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:  # Strong correlation threshold
                    strong_correlations.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'strong_correlations': strong_correlations
        }
    
    @staticmethod
    def _get_missing_data_analysis(df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing data patterns."""
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        return {
            'columns_with_missing': missing_counts[missing_counts > 0].to_dict(),
            'missing_percentages': missing_percentages[missing_percentages > 0].to_dict(),
            'rows_with_missing': df.isnull().any(axis=1).sum(),
            'complete_rows': (~df.isnull().any(axis=1)).sum()
        }
    
    @staticmethod
    def _get_duplicate_analysis(df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze duplicate rows."""
        duplicate_mask = df.duplicated()
        
        return {
            'duplicate_count': duplicate_mask.sum(),
            'duplicate_percentage': (duplicate_mask.sum() / len(df)) * 100,
            'unique_rows': len(df) - duplicate_mask.sum()
        }

class StatisticalTests:
    """
    Utility class for performing statistical tests.
    """
    
    @staticmethod
    def normality_test(series: pd.Series) -> Dict[str, Any]:
        """
        Perform normality tests on a numeric series.
        
        Args:
            series (pd.Series): Numeric series to test
            
        Returns:
            dict: Test results
        """
        from scipy.stats import shapiro, normaltest
        
        clean_series = series.dropna()
        
        if len(clean_series) < 3:
            return {'error': 'Insufficient data for normality testing'}
        
        results = {}
        
        # Shapiro-Wilk test (for small samples)
        if len(clean_series) <= 5000:
            stat, p_value = shapiro(clean_series)
            results['shapiro_wilk'] = {
                'statistic': stat,
                'p_value': p_value,
                'is_normal': p_value > 0.05
            }
        
        # D'Agostino's normality test
        if len(clean_series) >= 8:
            stat, p_value = normaltest(clean_series)
            results['dagostino'] = {
                'statistic': stat,
                'p_value': p_value,
                'is_normal': p_value > 0.05
            }
        
        return results
    
    @staticmethod
    def correlation_significance(df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
        """
        Test the significance of correlation between two columns.
        
        Args:
            df (pd.DataFrame): Dataset
            col1 (str): First column
            col2 (str): Second column
            
        Returns:
            dict: Correlation test results
        """
        from scipy.stats import pearsonr, spearmanr
        
        data1 = df[col1].dropna()
        data2 = df[col2].dropna()
        
        # Align the data (remove rows where either value is missing)
        combined = pd.concat([data1, data2], axis=1).dropna()
        
        if len(combined) < 3:
            return {'error': 'Insufficient data for correlation testing'}
        
        x, y = combined.iloc[:, 0], combined.iloc[:, 1]
        
        # Pearson correlation
        pearson_corr, pearson_p = pearsonr(x, y)
        
        # Spearman correlation (rank-based)
        spearman_corr, spearman_p = spearmanr(x, y)
        
        return {
            'pearson': {
                'correlation': pearson_corr,
                'p_value': pearson_p,
                'is_significant': pearson_p < 0.05
            },
            'spearman': {
                'correlation': spearman_corr,
                'p_value': spearman_p,
                'is_significant': spearman_p < 0.05
            },
            'sample_size': len(combined)
        }
