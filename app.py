import streamlit as st
import pandas as pd
import numpy as np
from data_analyzer import DataAnalyzer
from visualization import DataVisualizer
from ai_insights import AIInsights
from utils import FileHandler, ExportHandler
import traceback

# Page configuration
st.set_page_config(
    page_title="AI Data Analysis Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'insights' not in st.session_state:
    st.session_state.insights = None

def main():
    # Page d'accueil avec message principal
    st.markdown("""
    <div style="text-align: center; padding: 3rem 0;">
        <h1 style="color: #2E8B57; font-size: 3.5rem; font-weight: bold; margin: 0;">
            L'agritech intelligente au service de la rentabilité durable
        </h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Bloc de statistiques
    st.markdown("""
    <div style="margin: 3rem 0;">
        <div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 2rem;">
            <div style="background: linear-gradient(135deg, #2E8B57, #228B22); color: white; padding: 2rem; border-radius: 15px; text-align: center; min-width: 200px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <h3 style="margin: 0; font-size: 2.5rem; font-weight: bold;">15</h3>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">Utilisateurs actifs</p>
            </div>
            <div style="background: linear-gradient(135deg, #4682B4, #1E90FF); color: white; padding: 2rem; border-radius: 15px; text-align: center; min-width: 200px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <h3 style="margin: 0; font-size: 2.5rem; font-weight: bold;">31</h3>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">Projets en financement</p>
            </div>
            <div style="background: linear-gradient(135deg, #FF8C00, #FF6347); color: white; padding: 2rem; border-radius: 15px; text-align: center; min-width: 200px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <h3 style="margin: 0; font-size: 2.5rem; font-weight: bold;">1 510,44</h3>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">Hectares cultivés</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Message de présentation Maintso Vola en bas
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; margin-top: 2rem;">
        <p style="color: #666; font-size: 1rem; max-width: 900px; margin: 0 auto; line-height: 1.6;">
            Chez Maintso Vola, nous connectons la finance et la technologie pour révolutionner l'agriculture. 
            Grâce à la data, à des outils de suivi en temps réel et à une infrastructure optimisée, 
            chaque investissement devient traçable, performant et à fort impact. 
            📊 Investissez dans une nouvelle génération de projets agricoles pilotés par la tech.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for file upload and configuration
    with st.sidebar:
        st.header("📁 Data Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your dataset to begin analysis"
        )
        
        if uploaded_file is not None:
            try:
                # Load data
                with st.spinner("Loading data..."):
                    file_handler = FileHandler()
                    data = file_handler.load_file(uploaded_file)
                    st.session_state.data = data
                    st.session_state.analyzer = DataAnalyzer(data)
                
                st.success(f"✅ Data loaded successfully!")
                st.info(f"Shape: {data.shape[0]} rows × {data.shape[1]} columns")
                
                # Display data types
                st.subheader("📋 Data Overview")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Rows", data.shape[0])
                with col2:
                    st.metric("Columns", data.shape[1])
                
                # Show column types
                with st.expander("Column Information"):
                    for col in data.columns:
                        dtype = str(data[col].dtype)
                        null_count = data[col].isnull().sum()
                        st.text(f"{col}: {dtype} ({null_count} nulls)")
                        
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
                st.session_state.data = None
                st.session_state.analyzer = None
    
    # Main content area
    if st.session_state.data is not None:
        data = st.session_state.data
        analyzer = st.session_state.analyzer
        
        # Create tabs for different functionalities
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Data Preview", 
            "🤖 AI Insights", 
            "📈 Visualizations", 
            "💬 Natural Language Query", 
            "📄 Export"
        ])
        
        with tab1:
            display_data_preview(data, analyzer)
        
        with tab2:
            display_ai_insights(data)
        
        with tab3:
            display_visualizations(data)
        
        with tab4:
            display_natural_language_query(data)
        
        with tab5:
            display_export_options(data)
    
    else:
        st.info("👆 Veuillez télécharger un fichier CSV ou Excel pour commencer l'analyse")

def display_data_preview(data, analyzer):
    st.header("📊 Data Preview & Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Sample Data")
        st.dataframe(data.head(100), use_container_width=True)
    
    with col2:
        st.subheader("Quick Stats")
        
        # Basic statistics
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.metric("Numeric Columns", len(numeric_cols))
            st.metric("Text Columns", len(data.columns) - len(numeric_cols))
            st.metric("Missing Values", data.isnull().sum().sum())
            
            # Show basic statistics for numeric columns
            with st.expander("Numeric Column Statistics"):
                st.dataframe(data[numeric_cols].describe())
        
        # Data quality assessment
        st.subheader("Data Quality")
        quality_metrics = analyzer.get_data_quality_metrics()
        for metric, value in quality_metrics.items():
            st.metric(metric, f"{value:.1f}%")

def display_ai_insights(data):
    st.header("🤖 AI-Powered Insights")
    
    ai_insights = AIInsights()
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("Analysis Options")
        
        insight_types = st.multiselect(
            "Select insight types to generate:",
            ["Statistical Summary", "Trend Analysis", "Anomaly Detection", "Correlations", "Recommendations"],
            default=["Statistical Summary", "Trend Analysis"]
        )
        
        if st.button("🔍 Generate AI Insights", type="primary"):
            with st.spinner("AI is analyzing your data..."):
                try:
                    insights = ai_insights.generate_comprehensive_insights(data, insight_types)
                    st.session_state.insights = insights
                except Exception as e:
                    st.error(f"Error generating insights: {str(e)}")
                    st.error("Please ensure your OpenAI API key is set correctly.")
    
    with col1:
        if st.session_state.insights:
            insights = st.session_state.insights
            
            for insight_type, content in insights.items():
                with st.expander(f"📋 {insight_type}", expanded=True):
                    if isinstance(content, dict):
                        for key, value in content.items():
                            st.markdown(f"**{key}:** {value}")
                    else:
                        st.markdown(content)
        else:
            st.info("Click 'Generate AI Insights' to analyze your data with AI")

def display_visualizations(data):
    st.header("📈 Interactive Visualizations")
    
    visualizer = DataVisualizer(data)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Visualization Options")
        
        chart_type = st.selectbox(
            "Select chart type:",
            ["Correlation Heatmap", "Distribution Plot", "Scatter Plot", "Time Series", "Box Plot", "Bar Chart"]
        )
        
        # Column selection based on chart type
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        datetime_cols = data.select_dtypes(include=['datetime']).columns.tolist()
        
        if chart_type == "Scatter Plot":
            x_col = st.selectbox("X-axis column:", numeric_cols)
            y_col = st.selectbox("Y-axis column:", numeric_cols)
            color_col = st.selectbox("Color by (optional):", [None] + categorical_cols + numeric_cols)
        elif chart_type == "Time Series":
            if datetime_cols:
                time_col = st.selectbox("Time column:", datetime_cols)
                value_col = st.selectbox("Value column:", numeric_cols)
            else:
                st.warning("No datetime columns found. Please ensure your data has datetime columns for time series plots.")
                return
        elif chart_type in ["Distribution Plot", "Box Plot"]:
            selected_col = st.selectbox("Select column:", numeric_cols)
            group_by = st.selectbox("Group by (optional):", [None] + categorical_cols)
        elif chart_type == "Bar Chart":
            cat_col = st.selectbox("Category column:", categorical_cols)
            value_col = st.selectbox("Value column:", numeric_cols)
        
    with col2:
        try:
            if chart_type == "Correlation Heatmap":
                fig = visualizer.create_correlation_heatmap()
            elif chart_type == "Distribution Plot":
                fig = visualizer.create_distribution_plot(selected_col, group_by)
            elif chart_type == "Scatter Plot":
                fig = visualizer.create_scatter_plot(x_col, y_col, color_col)
            elif chart_type == "Time Series" and datetime_cols:
                fig = visualizer.create_time_series_plot(time_col, value_col)
            elif chart_type == "Box Plot":
                fig = visualizer.create_box_plot(selected_col, group_by)
            elif chart_type == "Bar Chart":
                fig = visualizer.create_bar_chart(cat_col, value_col)
            
            if 'fig' in locals():
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")

def display_natural_language_query(data):
    st.header("💬 Natural Language Data Query")
    
    ai_insights = AIInsights()
    
    st.markdown("Ask questions about your data in plain English!")
    
    # Sample questions
    with st.expander("💡 Example Questions"):
        st.markdown("""
        - "What are the main trends in this dataset?"
        - "Which columns have the strongest correlations?"
        - "Are there any outliers or anomalies?"
        - "What insights can you provide about the sales data?"
        - "Show me the distribution of values in the age column"
        """)
    
    query = st.text_area(
        "Enter your question:",
        placeholder="e.g., What are the key patterns in my data?",
        height=100
    )
    
    if st.button("🔍 Ask AI", type="primary") and query:
        with st.spinner("AI is analyzing your question..."):
            try:
                response = ai_insights.answer_natural_language_query(data, query)
                
                st.subheader("🤖 AI Response:")
                st.markdown(response)
                
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                st.error("Please ensure your OpenAI API key is set correctly.")

def display_export_options(data):
    st.header("📄 Export Data & Insights")
    
    export_handler = ExportHandler()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Export Data")
        
        # Data export options
        export_format = st.selectbox(
            "Select export format:",
            ["CSV", "Excel", "JSON"]
        )
        
        # Filter options
        st.subheader("Filter Options")
        
        # Column selection
        selected_columns = st.multiselect(
            "Select columns to export:",
            data.columns.tolist(),
            default=data.columns.tolist()
        )
        
        # Row filtering
        max_rows = st.number_input(
            "Maximum rows to export:",
            min_value=1,
            max_value=len(data),
            value=min(1000, len(data))
        )
        
        if st.button("📥 Export Data"):
            try:
                filtered_data = data[selected_columns].head(max_rows)
                
                if export_format == "CSV":
                    csv_data = export_handler.to_csv(filtered_data)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name="exported_data.csv",
                        mime="text/csv"
                    )
                elif export_format == "Excel":
                    excel_data = export_handler.to_excel(filtered_data)
                    st.download_button(
                        label="Download Excel",
                        data=excel_data,
                        file_name="exported_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                elif export_format == "JSON":
                    json_data = export_handler.to_json(filtered_data)
                    st.download_button(
                        label="Download JSON",
                        data=json_data,
                        file_name="exported_data.json",
                        mime="application/json"
                    )
                    
            except Exception as e:
                st.error(f"Error exporting data: {str(e)}")
    
    with col2:
        st.subheader("📋 Export Insights")
        
        if st.session_state.insights:
            insights_text = export_handler.format_insights_for_export(st.session_state.insights)
            
            st.download_button(
                label="📄 Download Insights Report",
                data=insights_text,
                file_name="ai_insights_report.txt",
                mime="text/plain"
            )
            
            st.text_area(
                "Insights Preview:",
                insights_text[:500] + "..." if len(insights_text) > 500 else insights_text,
                height=300,
                disabled=True
            )
        else:
            st.info("Generate AI insights first to export them")

if __name__ == "__main__":
    main()
