import json
import os
import pandas as pd
import numpy as np
from openai import OpenAI

class AIInsights:
    """
    AI-powered insights generator using OpenAI for intelligent data analysis.
    """
    
    def __init__(self):
        """
        Initialize the AIInsights class with OpenAI client.
        """
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        self.model = "gpt-4o"
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def generate_comprehensive_insights(self, data, insight_types):
        """
        Generate comprehensive insights about the dataset.
        
        Args:
            data (pd.DataFrame): The dataset to analyze
            insight_types (list): List of insight types to generate
            
        Returns:
            dict: Generated insights
        """
        insights = {}
        
        # Prepare data summary for AI analysis
        data_summary = self._prepare_data_summary(data)
        
        try:
            for insight_type in insight_types:
                if insight_type == "Statistical Summary":
                    insights[insight_type] = self._generate_statistical_insights(data_summary)
                elif insight_type == "Trend Analysis":
                    insights[insight_type] = self._generate_trend_insights(data_summary)
                elif insight_type == "Anomaly Detection":
                    insights[insight_type] = self._generate_anomaly_insights(data_summary)
                elif insight_type == "Correlations":
                    insights[insight_type] = self._generate_correlation_insights(data_summary)
                elif insight_type == "Recommendations":
                    insights[insight_type] = self._generate_recommendations(data_summary)
                    
        except Exception as e:
            insights["Error"] = f"Failed to generate insights: {str(e)}"
        
        return insights
    
    def _prepare_data_summary(self, data):
        """
        Prepare a comprehensive data summary for AI analysis.
        
        Args:
            data (pd.DataFrame): The dataset
            
        Returns:
            dict: Data summary
        """
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
        
        summary = {
            "shape": data.shape,
            "columns": {
                "numeric": numeric_columns,
                "categorical": categorical_columns,
                "total": len(data.columns)
            },
            "missing_values": data.isnull().sum().to_dict(),
            "basic_stats": {},
            "sample_data": data.head(5).to_dict('records')
        }
        
        # Add basic statistics for numeric columns
        if numeric_columns:
            summary["basic_stats"] = data[numeric_columns].describe().to_dict()
        
        # Add categorical value counts for categorical columns
        if categorical_columns:
            summary["categorical_stats"] = {}
            for col in categorical_columns[:5]:  # Limit to first 5 categorical columns
                summary["categorical_stats"][col] = data[col].value_counts().head(10).to_dict()
        
        return summary
    
    def _generate_statistical_insights(self, data_summary):
        """
        Generate statistical insights using AI.
        
        Args:
            data_summary (dict): Summary of the dataset
            
        Returns:
            dict: Statistical insights
        """
        prompt = f"""
        Analyze this dataset and provide key statistical insights:
        
        Dataset Summary:
        - Shape: {data_summary['shape'][0]} rows, {data_summary['shape'][1]} columns
        - Numeric columns: {data_summary['columns']['numeric']}
        - Categorical columns: {data_summary['columns']['categorical']}
        - Missing values: {data_summary['missing_values']}
        - Basic statistics: {data_summary.get('basic_stats', {})}
        
        Provide insights about:
        1. Dataset size and structure
        2. Data quality (missing values, completeness)
        3. Key statistical patterns
        4. Notable characteristics
        
        Return as JSON with keys: dataset_overview, data_quality, key_patterns, notable_features
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a data analysis expert. Provide clear, actionable insights about datasets."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    
    def _generate_trend_insights(self, data_summary):
        """
        Generate trend analysis insights using AI.
        
        Args:
            data_summary (dict): Summary of the dataset
            
        Returns:
            dict: Trend insights
        """
        prompt = f"""
        Analyze potential trends and patterns in this dataset:
        
        Numeric columns: {data_summary['columns']['numeric']}
        Basic statistics: {data_summary.get('basic_stats', {})}
        Sample data: {data_summary['sample_data'][:3]}
        
        Identify and analyze:
        1. Potential time-based trends (if applicable)
        2. Relationships between variables
        3. Distribution patterns
        4. Growth or decline patterns
        
        Return as JSON with keys: time_trends, variable_relationships, distributions, growth_patterns
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a trend analysis expert. Identify meaningful patterns and trends in data."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    
    def _generate_anomaly_insights(self, data_summary):
        """
        Generate anomaly detection insights using AI.
        
        Args:
            data_summary (dict): Summary of the dataset
            
        Returns:
            dict: Anomaly insights
        """
        prompt = f"""
        Identify potential anomalies and outliers in this dataset:
        
        Basic statistics: {data_summary.get('basic_stats', {})}
        Missing values: {data_summary['missing_values']}
        Sample data: {data_summary['sample_data'][:5]}
        
        Look for:
        1. Statistical outliers (extreme values)
        2. Data quality issues
        3. Unusual patterns or inconsistencies
        4. Potential data entry errors
        
        Return as JSON with keys: statistical_outliers, data_quality_issues, unusual_patterns, recommendations
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an anomaly detection expert. Identify unusual patterns and potential issues in data."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    
    def _generate_correlation_insights(self, data_summary):
        """
        Generate correlation analysis insights using AI.
        
        Args:
            data_summary (dict): Summary of the dataset
            
        Returns:
            dict: Correlation insights
        """
        prompt = f"""
        Analyze potential correlations and relationships in this dataset:
        
        Numeric columns: {data_summary['columns']['numeric']}
        Categorical columns: {data_summary['columns']['categorical']}
        Basic statistics: {data_summary.get('basic_stats', {})}
        
        Identify:
        1. Strong potential correlations between numeric variables
        2. Relationships between categorical and numeric variables
        3. Multi-variable relationships
        4. Causal vs correlational insights
        
        Return as JSON with keys: numeric_correlations, categorical_relationships, multi_variable, causal_insights
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a correlation analysis expert. Identify meaningful relationships between variables."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    
    def _generate_recommendations(self, data_summary):
        """
        Generate actionable recommendations using AI.
        
        Args:
            data_summary (dict): Summary of the dataset
            
        Returns:
            dict: Recommendations
        """
        prompt = f"""
        Based on this dataset analysis, provide actionable recommendations:
        
        Dataset overview:
        - Shape: {data_summary['shape']}
        - Columns: {data_summary['columns']}
        - Missing values: {data_summary['missing_values']}
        - Statistics: {data_summary.get('basic_stats', {})}
        
        Provide recommendations for:
        1. Data cleaning and preprocessing
        2. Further analysis opportunities
        3. Visualization suggestions
        4. Business insights and actions
        
        Return as JSON with keys: data_cleaning, analysis_opportunities, visualization_suggestions, business_insights
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a data strategy expert. Provide actionable recommendations for data analysis and business decisions."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    
    def answer_natural_language_query(self, data, query):
        """
        Answer natural language queries about the data.
        
        Args:
            data (pd.DataFrame): The dataset
            query (str): Natural language question
            
        Returns:
            str: AI response to the query
        """
        # Prepare context about the data
        data_context = self._prepare_data_summary(data)
        
        # Add correlation information if numeric columns exist
        correlation_info = ""
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            corr_matrix = data[numeric_cols].corr()
            # Get strongest correlations
            correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.3:  # Only include moderate to strong correlations
                        correlations.append(f"{corr_matrix.columns[i]} - {corr_matrix.columns[j]}: {corr_val:.3f}")
            
            if correlations:
                correlation_info = f"Strong correlations found: {'; '.join(correlations[:5])}"
        
        prompt = f"""
        You are a data analyst assistant. Answer the following question about this dataset:
        
        Question: {query}
        
        Dataset Context:
        - Shape: {data_context['shape'][0]} rows, {data_context['shape'][1]} columns
        - Numeric columns: {data_context['columns']['numeric']}
        - Categorical columns: {data_context['columns']['categorical']}
        - Missing values summary: {dict(list(data_context['missing_values'].items())[:5])}
        - Sample statistics: {data_context.get('basic_stats', {})}
        - {correlation_info}
        
        Provide a clear, informative answer based on the data characteristics. If specific calculations are needed that you cannot perform, explain what analysis would be required.
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful data analyst. Provide clear, accurate insights based on dataset characteristics."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content
    
    def generate_executive_summary(self, data):
        """
        Generate an executive summary of the dataset.
        
        Args:
            data (pd.DataFrame): The dataset
            
        Returns:
            str: Executive summary
        """
        data_summary = self._prepare_data_summary(data)
        
        prompt = f"""
        Create an executive summary of this dataset for business stakeholders:
        
        Dataset Overview:
        - Size: {data_summary['shape'][0]} records, {data_summary['shape'][1]} variables
        - Data types: {len(data_summary['columns']['numeric'])} numeric, {len(data_summary['columns']['categorical'])} categorical
        - Data quality: {sum(data_summary['missing_values'].values())} missing values total
        - Key variables: {data_summary['columns']['numeric'][:5]} (numeric), {data_summary['columns']['categorical'][:5]} (categorical)
        
        Basic statistics: {data_summary.get('basic_stats', {})}
        
        Provide a 3-4 paragraph executive summary covering:
        1. What this dataset contains and its business relevance
        2. Data quality and completeness assessment
        3. Key opportunities for analysis and insights
        4. Recommended next steps
        
        Write in business-friendly language, avoiding technical jargon.
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a business analyst creating executive summaries. Write clearly for business stakeholders."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content
