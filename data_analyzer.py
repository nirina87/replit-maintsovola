import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class DataAnalyzer:
    """
    A comprehensive data analysis class that provides statistical analysis,
    data profiling, and trend detection capabilities.
    """
    
    def __init__(self, data):
        """
        Initialize the DataAnalyzer with a pandas DataFrame.
        
        Args:
            data (pd.DataFrame): The dataset to analyze
        """
        self.data = data.copy()
        self.numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
        self.datetime_columns = data.select_dtypes(include=['datetime']).columns.tolist()
    
    def get_basic_statistics(self):
        """
        Generate basic statistical summary of the dataset.
        
        Returns:
            dict: Basic statistics including shape, types, missing values
        """
        stats = {
            'shape': self.data.shape,
            'columns': {
                'total': len(self.data.columns),
                'numeric': len(self.numeric_columns),
                'categorical': len(self.categorical_columns),
                'datetime': len(self.datetime_columns)
            },
            'missing_values': {
                'total': self.data.isnull().sum().sum(),
                'percentage': (self.data.isnull().sum().sum() / (self.data.shape[0] * self.data.shape[1])) * 100
            },
            'duplicate_rows': self.data.duplicated().sum(),
            'memory_usage': self.data.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        }
        
        return stats
    
    def get_data_quality_metrics(self):
        """
        Calculate data quality metrics.
        
        Returns:
            dict: Data quality metrics
        """
        total_cells = self.data.shape[0] * self.data.shape[1]
        missing_cells = self.data.isnull().sum().sum()
        
        metrics = {
            'Completeness': ((total_cells - missing_cells) / total_cells) * 100,
            'Uniqueness': (len(self.data.drop_duplicates()) / len(self.data)) * 100,
            'Consistency': self._calculate_consistency_score()
        }
        
        return metrics
    
    def _calculate_consistency_score(self):
        """
        Calculate a consistency score based on data types and patterns.
        
        Returns:
            float: Consistency score as percentage
        """
        consistency_scores = []
        
        for col in self.categorical_columns:
            # Check for consistent formatting in categorical columns
            unique_values = self.data[col].dropna().unique()
            if len(unique_values) > 0:
                # Simple consistency check - could be enhanced
                consistency_scores.append(90.0)  # Placeholder for actual consistency logic
        
        return np.mean(consistency_scores) if consistency_scores else 95.0
    
    def detect_outliers(self, method='iqr'):
        """
        Detect outliers in numeric columns using specified method.
        
        Args:
            method (str): Method to use ('iqr', 'zscore')
            
        Returns:
            dict: Outliers information for each numeric column
        """
        outliers = {}
        
        for col in self.numeric_columns:
            if method == 'iqr':
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (self.data[col] < lower_bound) | (self.data[col] > upper_bound)
                outliers[col] = {
                    'count': outlier_mask.sum(),
                    'percentage': (outlier_mask.sum() / len(self.data)) * 100,
                    'bounds': {'lower': lower_bound, 'upper': upper_bound}
                }
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(self.data[col].dropna()))
                outlier_mask = z_scores > 3
                outliers[col] = {
                    'count': outlier_mask.sum(),
                    'percentage': (outlier_mask.sum() / len(self.data)) * 100,
                    'threshold': 3
                }
        
        return outliers
    
    def calculate_correlations(self):
        """
        Calculate correlations between numeric variables.
        
        Returns:
            pd.DataFrame: Correlation matrix
        """
        if len(self.numeric_columns) < 2:
            return pd.DataFrame()
        
        correlation_matrix = self.data[self.numeric_columns].corr()
        return correlation_matrix
    
    def detect_trends(self):
        """
        Detect trends in numeric data over time or sequence.
        
        Returns:
            dict: Trend information for each numeric column
        """
        trends = {}
        
        for col in self.numeric_columns:
            series = self.data[col].dropna()
            if len(series) < 3:
                continue
                
            # Calculate trend using linear regression
            x = np.arange(len(series))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
            
            # Determine trend direction
            if p_value < 0.05:  # Significant trend
                if slope > 0:
                    trend_direction = 'increasing'
                else:
                    trend_direction = 'decreasing'
            else:
                trend_direction = 'no clear trend'
            
            trends[col] = {
                'direction': trend_direction,
                'slope': slope,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'strength': self._categorize_trend_strength(abs(r_value))
            }
        
        return trends
    
    def _categorize_trend_strength(self, r_value):
        """
        Categorize the strength of correlation/trend.
        
        Args:
            r_value (float): Absolute correlation coefficient
            
        Returns:
            str: Strength category
        """
        if r_value >= 0.7:
            return 'strong'
        elif r_value >= 0.3:
            return 'moderate'
        else:
            return 'weak'
    
    def perform_clustering(self, n_clusters=3):
        """
        Perform K-means clustering on numeric data.
        
        Args:
            n_clusters (int): Number of clusters
            
        Returns:
            dict: Clustering results
        """
        if len(self.numeric_columns) < 2:
            return {'error': 'Insufficient numeric columns for clustering'}
        
        # Prepare data for clustering
        clustering_data = self.data[self.numeric_columns].dropna()
        
        if len(clustering_data) < n_clusters:
            return {'error': 'Insufficient data points for clustering'}
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(clustering_data)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        # Calculate cluster statistics
        cluster_stats = {}
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            cluster_data = clustering_data[cluster_mask]
            
            cluster_stats[f'Cluster_{i}'] = {
                'size': len(cluster_data),
                'percentage': (len(cluster_data) / len(clustering_data)) * 100,
                'centroid': cluster_data.mean().to_dict()
            }
        
        return {
            'cluster_labels': cluster_labels,
            'cluster_centers': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_,
            'cluster_stats': cluster_stats
        }
    
    def perform_pca(self, n_components=2):
        """
        Perform Principal Component Analysis on numeric data.
        
        Args:
            n_components (int): Number of principal components
            
        Returns:
            dict: PCA results
        """
        if len(self.numeric_columns) < 2:
            return {'error': 'Insufficient numeric columns for PCA'}
        
        # Prepare data for PCA
        pca_data = self.data[self.numeric_columns].dropna()
        
        if len(pca_data) < 2:
            return {'error': 'Insufficient data points for PCA'}
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(pca_data)
        
        # Perform PCA
        pca = PCA(n_components=min(n_components, len(self.numeric_columns)))
        transformed_data = pca.fit_transform(scaled_data)
        
        return {
            'transformed_data': transformed_data,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_),
            'components': pca.components_,
            'feature_names': self.numeric_columns
        }
    
    def get_column_profiles(self):
        """
        Generate detailed profiles for each column.
        
        Returns:
            dict: Column profiles with statistics and characteristics
        """
        profiles = {}
        
        for col in self.data.columns:
            col_data = self.data[col]
            
            profile = {
                'dtype': str(col_data.dtype),
                'non_null_count': col_data.count(),
                'null_count': col_data.isnull().sum(),
                'unique_count': col_data.nunique(),
                'memory_usage': col_data.memory_usage(deep=True)
            }
            
            if col in self.numeric_columns:
                profile.update({
                    'mean': col_data.mean(),
                    'median': col_data.median(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'skewness': col_data.skew(),
                    'kurtosis': col_data.kurtosis()
                })
            
            elif col in self.categorical_columns:
                value_counts = col_data.value_counts()
                profile.update({
                    'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                    'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                    'top_values': value_counts.head(5).to_dict()
                })
            
            profiles[col] = profile
        
        return profiles
    
    def detect_anomalies(self):
        """
        Detect various types of anomalies in the dataset.
        
        Returns:
            dict: Anomaly detection results
        """
        anomalies = {
            'statistical_outliers': self.detect_outliers('iqr'),
            'data_quality_issues': self._detect_data_quality_anomalies(),
            'pattern_anomalies': self._detect_pattern_anomalies()
        }
        
        return anomalies
    
    def _detect_data_quality_anomalies(self):
        """
        Detect data quality related anomalies.
        
        Returns:
            dict: Data quality anomalies
        """
        issues = {}
        
        # Check for columns with too many missing values
        missing_threshold = 0.5  # 50%
        for col in self.data.columns:
            missing_pct = self.data[col].isnull().sum() / len(self.data)
            if missing_pct > missing_threshold:
                issues[f'{col}_high_missing'] = {
                    'type': 'high_missing_values',
                    'percentage': missing_pct * 100
                }
        
        # Check for columns with low cardinality
        for col in self.categorical_columns:
            unique_ratio = self.data[col].nunique() / len(self.data)
            if unique_ratio < 0.01 and self.data[col].nunique() > 1:  # Less than 1% unique
                issues[f'{col}_low_cardinality'] = {
                    'type': 'low_cardinality',
                    'unique_ratio': unique_ratio
                }
        
        return issues
    
    def _detect_pattern_anomalies(self):
        """
        Detect pattern-based anomalies.
        
        Returns:
            dict: Pattern anomalies
        """
        patterns = {}
        
        # Check for sudden changes in numeric columns
        for col in self.numeric_columns:
            series = self.data[col].dropna()
            if len(series) > 10:
                # Calculate rolling mean and identify significant deviations
                rolling_mean = series.rolling(window=5).mean()
                deviations = np.abs(series - rolling_mean) > (2 * series.std())
                
                if deviations.sum() > 0:
                    patterns[f'{col}_sudden_changes'] = {
                        'type': 'sudden_changes',
                        'count': deviations.sum(),
                        'positions': deviations[deviations].index.tolist()
                    }
        
        return patterns
