import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns

class DataVisualizer:
    """
    A comprehensive data visualization class using Plotly for interactive charts.
    """
    
    def __init__(self, data):
        """
        Initialize the DataVisualizer with a pandas DataFrame.
        
        Args:
            data (pd.DataFrame): The dataset to visualize
        """
        self.data = data.copy()
        self.numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
        self.datetime_columns = data.select_dtypes(include=['datetime']).columns.tolist()
        
        # Default color palette
        self.color_palette = px.colors.qualitative.Set3
    
    def create_correlation_heatmap(self):
        """
        Create an interactive correlation heatmap for numeric columns.
        
        Returns:
            plotly.graph_objects.Figure: Correlation heatmap
        """
        if len(self.numeric_columns) < 2:
            # Create empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="Not enough numeric columns for correlation analysis",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            fig.update_layout(
                title="Correlation Heatmap",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False)
            )
            return fig
        
        # Calculate correlation matrix
        corr_matrix = self.data[self.numeric_columns].corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.around(corr_matrix.values, decimals=2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Correlation Heatmap",
            xaxis_title="Variables",
            yaxis_title="Variables",
            width=600,
            height=600
        )
        
        return fig
    
    def create_distribution_plot(self, column, group_by=None):
        """
        Create distribution plot for a numeric column.
        
        Args:
            column (str): Column name to plot
            group_by (str, optional): Column to group by
            
        Returns:
            plotly.graph_objects.Figure: Distribution plot
        """
        if column not in self.numeric_columns:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Column '{column}' is not numeric",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            return fig
        
        if group_by and group_by in self.categorical_columns:
            # Create grouped distribution
            fig = px.histogram(
                self.data, 
                x=column, 
                color=group_by,
                marginal="box",
                nbins=30,
                title=f"Distribution of {column} by {group_by}"
            )
        else:
            # Create single distribution
            fig = px.histogram(
                self.data, 
                x=column,
                marginal="box",
                nbins=30,
                title=f"Distribution of {column}"
            )
        
        fig.update_layout(
            xaxis_title=column,
            yaxis_title="Frequency",
            showlegend=bool(group_by)
        )
        
        return fig
    
    def create_scatter_plot(self, x_col, y_col, color_col=None):
        """
        Create scatter plot for two numeric columns.
        
        Args:
            x_col (str): X-axis column
            y_col (str): Y-axis column
            color_col (str, optional): Column for color coding
            
        Returns:
            plotly.graph_objects.Figure: Scatter plot
        """
        if x_col not in self.numeric_columns or y_col not in self.numeric_columns:
            fig = go.Figure()
            fig.add_annotation(
                text="Both X and Y columns must be numeric",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            return fig
        
        # Create scatter plot
        if color_col:
            fig = px.scatter(
                self.data,
                x=x_col,
                y=y_col,
                color=color_col,
                title=f"{y_col} vs {x_col}",
                trendline="ols"  # Add trend line
            )
        else:
            fig = px.scatter(
                self.data,
                x=x_col,
                y=y_col,
                title=f"{y_col} vs {x_col}",
                trendline="ols"
            )
        
        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_col
        )
        
        return fig
    
    def create_time_series_plot(self, time_col, value_col):
        """
        Create time series plot.
        
        Args:
            time_col (str): Time column
            value_col (str): Value column
            
        Returns:
            plotly.graph_objects.Figure: Time series plot
        """
        if time_col not in self.datetime_columns:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Column '{time_col}' is not a datetime column",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            return fig
        
        if value_col not in self.numeric_columns:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Column '{value_col}' is not numeric",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            return fig
        
        # Sort by time column
        sorted_data = self.data.sort_values(time_col)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sorted_data[time_col],
            y=sorted_data[value_col],
            mode='lines+markers',
            name=value_col,
            line=dict(width=2),
            marker=dict(size=4)
        ))
        
        fig.update_layout(
            title=f"Time Series: {value_col} over {time_col}",
            xaxis_title=time_col,
            yaxis_title=value_col,
            hovermode='x unified'
        )
        
        return fig
    
    def create_box_plot(self, column, group_by=None):
        """
        Create box plot for a numeric column.
        
        Args:
            column (str): Column to plot
            group_by (str, optional): Column to group by
            
        Returns:
            plotly.graph_objects.Figure: Box plot
        """
        if column not in self.numeric_columns:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Column '{column}' is not numeric",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            return fig
        
        if group_by and group_by in self.categorical_columns:
            fig = px.box(
                self.data,
                x=group_by,
                y=column,
                title=f"Box Plot of {column} by {group_by}"
            )
        else:
            fig = px.box(
                self.data,
                y=column,
                title=f"Box Plot of {column}"
            )
        
        fig.update_layout(
            yaxis_title=column
        )
        
        return fig
    
    def create_bar_chart(self, category_col, value_col):
        """
        Create bar chart for categorical data.
        
        Args:
            category_col (str): Category column
            value_col (str): Value column
            
        Returns:
            plotly.graph_objects.Figure: Bar chart
        """
        if category_col not in self.categorical_columns:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Column '{category_col}' is not categorical",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            return fig
        
        if value_col not in self.numeric_columns:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Column '{value_col}' is not numeric",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            return fig
        
        # Aggregate data
        agg_data = self.data.groupby(category_col)[value_col].sum().reset_index()
        agg_data = agg_data.sort_values(value_col, ascending=False)
        
        fig = px.bar(
            agg_data,
            x=category_col,
            y=value_col,
            title=f"{value_col} by {category_col}"
        )
        
        fig.update_layout(
            xaxis_title=category_col,
            yaxis_title=value_col,
            xaxis_tickangle=-45
        )
        
        return fig
    
    def create_multi_variable_plot(self, variables, plot_type="pair"):
        """
        Create multi-variable visualization.
        
        Args:
            variables (list): List of variables to plot
            plot_type (str): Type of plot ('pair', 'parallel')
            
        Returns:
            plotly.graph_objects.Figure: Multi-variable plot
        """
        numeric_vars = [var for var in variables if var in self.numeric_columns]
        
        if len(numeric_vars) < 2:
            fig = go.Figure()
            fig.add_annotation(
                text="Need at least 2 numeric variables",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            return fig
        
        if plot_type == "parallel":
            fig = px.parallel_coordinates(
                self.data[numeric_vars].dropna(),
                title="Parallel Coordinates Plot"
            )
        else:  # pair plot
            # Create correlation matrix subplot
            n_vars = len(numeric_vars)
            fig = make_subplots(
                rows=n_vars, cols=n_vars,
                subplot_titles=[f"{var1} vs {var2}" for var1 in numeric_vars for var2 in numeric_vars]
            )
            
            for i, var1 in enumerate(numeric_vars):
                for j, var2 in enumerate(numeric_vars):
                    if i == j:
                        # Diagonal: histogram
                        fig.add_trace(
                            go.Histogram(x=self.data[var1], name=var1),
                            row=i+1, col=j+1
                        )
                    else:
                        # Off-diagonal: scatter plot
                        fig.add_trace(
                            go.Scatter(
                                x=self.data[var2], 
                                y=self.data[var1],
                                mode='markers',
                                name=f"{var1} vs {var2}",
                                showlegend=False
                            ),
                            row=i+1, col=j+1
                        )
            
            fig.update_layout(title="Pair Plot Matrix")
        
        return fig
    
    def create_summary_dashboard(self):
        """
        Create a summary dashboard with multiple visualizations.
        
        Returns:
            plotly.graph_objects.Figure: Dashboard figure
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Data Overview", 
                "Missing Values", 
                "Numeric Distributions", 
                "Categorical Distributions"
            ],
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "histogram"}, {"type": "bar"}]
            ]
        )
        
        # 1. Data Overview (column types)
        col_types = {
            'Numeric': len(self.numeric_columns),
            'Categorical': len(self.categorical_columns),
            'DateTime': len(self.datetime_columns)
        }
        
        fig.add_trace(
            go.Bar(x=list(col_types.keys()), y=list(col_types.values()), name="Column Types"),
            row=1, col=1
        )
        
        # 2. Missing Values
        missing_data = self.data.isnull().sum()
        top_missing = missing_data[missing_data > 0].head(10)
        
        if len(top_missing) > 0:
            fig.add_trace(
                go.Bar(x=top_missing.index, y=top_missing.values, name="Missing Values"),
                row=1, col=2
            )
        
        # 3. Numeric Distributions (first numeric column)
        if self.numeric_columns:
            first_numeric = self.numeric_columns[0]
            fig.add_trace(
                go.Histogram(x=self.data[first_numeric], name=first_numeric),
                row=2, col=1
            )
        
        # 4. Categorical Distributions (first categorical column)
        if self.categorical_columns:
            first_categorical = self.categorical_columns[0]
            cat_counts = self.data[first_categorical].value_counts().head(10)
            fig.add_trace(
                go.Bar(x=cat_counts.index, y=cat_counts.values, name=first_categorical),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Data Summary Dashboard",
            height=800,
            showlegend=False
        )
        
        return fig
