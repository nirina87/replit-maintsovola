import streamlit as st
import pandas as pd
import numpy as np
from data_analyzer import DataAnalyzer
from visualization import DataVisualizer
from ai_insights import AIInsights
from utils import FileHandler, ExportHandler
from auth import AuthManager
import traceback

# Page configuration optimisée pour mobile PWA
st.set_page_config(
    page_title="Maintso Vola - Agritech",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Maintso Vola - L'agritech intelligente au service de la rentabilité durable"
    }
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'insights' not in st.session_state:
    st.session_state.insights = None
if 'page' not in st.session_state:
    st.session_state.page = 'accueil'
if 'user_logged_in' not in st.session_state:
    st.session_state.user_logged_in = False
if 'user_data' not in st.session_state:
    st.session_state.user_data = None

# Initialize auth manager
auth_manager = AuthManager()

def main():
    # Injection PWA dans le head HTML
    st.markdown("""
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="default">
    <meta name="theme-color" content="#2E8B57">
    <link rel="manifest" href="/static/manifest.json">
    <link rel="apple-touch-icon" href="/static/icon-192.png">
    
    <script>
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('/static/sw.js')
        .then(function(registration) {
            console.log('SW registered: ', registration);
        })
        .catch(function(registrationError) {
            console.log('SW registration failed: ', registrationError);
        });
    }
    </script>
    
    <style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }
    
    /* Style mobile-first */
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        
        .stButton > button {
            width: 100%;
            font-size: 18px;
            padding: 12px;
        }
        
        h1 {
            font-size: 24px !important;
        }
    }
    
    /* Masquer les éléments Streamlit non nécessaires sur mobile */
    .css-1d391kg, .css-1v0mbdj {
        display: none;
    }
    
    /* Style pour les cartes statistiques */
    .stat-card {
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Navigation entre les pages
    if st.session_state.page == 'accueil':
        show_accueil()
    elif st.session_state.page == 'connexion':
        show_connexion()
    elif st.session_state.page == 'inscription':
        show_inscription()
    else:
        show_dashboard()

def show_accueil():
    # Page d'accueil avec message principal
    st.markdown("""
    <div style="text-align: center; padding: 3rem 0;">
        <h1 style="color: #2E8B57; font-size: 35px; font-weight: bold; margin: 0;">
            L'agritech intelligente au service de la rentabilité durable
        </h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Récupérer les statistiques réelles de la base de données
    user_stats = auth_manager.get_user_stats()
    
    # Bloc de statistiques optimisé mobile
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card" style="background: linear-gradient(135deg, #2E8B57, #228B22); color: white;">
            <h2 style="margin: 0; font-size: 2rem; font-weight: bold;">{user_stats['total_users']}</h2>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Utilisateurs</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card" style="background: linear-gradient(135deg, #4682B4, #1E90FF); color: white;">
            <h2 style="margin: 0; font-size: 2rem; font-weight: bold;">31</h2>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Projets</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-card" style="background: linear-gradient(135deg, #FF8C00, #FF6347); color: white;">
            <h2 style="margin: 0; font-size: 2rem; font-weight: bold;">1,510</h2>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Hectares</p>
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
    
    # Bouton de connexion/inscription optimisé mobile
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🔐 Se connecter / S'inscrire", type="primary", use_container_width=True):
        st.session_state.page = 'connexion'
        st.rerun()
    
    # Afficher un message d'installation PWA
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem; padding: 1rem; background: #f0f8f5; border-radius: 10px; border-left: 4px solid #2E8B57;">
        <p style="margin: 0; color: #2E8B57; font-weight: bold;">📱 Installez l'app sur votre téléphone !</p>
        <p style="margin: 0.5rem 0 0 0; color: #666; font-size: 0.9rem;">
            Sur mobile: Menu → "Ajouter à l'écran d'accueil"<br>
            Accès hors ligne et notifications inclus !
        </p>
    </div>
    """, unsafe_allow_html=True)

def show_connexion():
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="color: #2E8B57; font-size: 28px; font-weight: bold;">
            Connexion à Maintso Vola
        </h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Onglets pour connexion et inscription
    tab1, tab2 = st.tabs(["🔑 Connexion", "📝 Inscription"])
    
    with tab1:
        st.markdown("### Se connecter")
        with st.form("login_form"):
            username = st.text_input("Nom d'utilisateur")
            password = st.text_input("Mot de passe", type="password")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                login_button = st.form_submit_button("Se connecter", type="primary", use_container_width=True)
            
            if login_button:
                if username and password:
                    # Tenter la connexion avec la base de données
                    result = auth_manager.login_user(username, password)
                    if result["success"]:
                        st.success(result["message"])
                        st.session_state.user_logged_in = True
                        st.session_state.user_data = result["user_data"]
                        st.session_state.page = 'dashboard'
                        st.rerun()
                    else:
                        st.error(result["message"])
                else:
                    st.error("Veuillez remplir tous les champs")
    
    with tab2:
        st.markdown("### S'inscrire")
        with st.form("register_form"):
            nom = st.text_input("Nom *")
            prenoms = st.text_input("Prénoms")
            email = st.text_input("Email *")
            telephone = st.text_input("Téléphone", placeholder="Format: 032XXXXXXX ou 033XXXXXXX")
            
            password = st.text_input("Mot de passe *", type="password")
            confirm_password = st.text_input("Confirmer mot de passe *", type="password")
            
            st.markdown("**Pourquoi rejoindre Maintso Vola ?**")
            investir = st.checkbox("Je souhaite investir dans l'agriculture")
            cherche_investisseurs = st.checkbox("Je cherche des investisseurs pour mon projet agricole")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                register_button = st.form_submit_button("S'inscrire", type="primary", use_container_width=True)
            
            if register_button:
                if nom and email and password and confirm_password:
                    if password == confirm_password:
                        # Validation du téléphone si fourni
                        if telephone and not (telephone.startswith('032') or telephone.startswith('033')):
                            st.error("Format de téléphone invalide. Utilisez 032XXXXXXX ou 033XXXXXXX")
                        else:
                            # Enregistrer l'utilisateur dans la base de données
                            result = auth_manager.register_user(
                                nom=nom,
                                prenoms=prenoms,
                                email=email,
                                telephone=telephone,
                                password=password,
                                investir=investir,
                                cherche_investisseurs=cherche_investisseurs
                            )
                            
                            if result["success"]:
                                st.success(result["message"])
                                st.info("Vous pouvez maintenant vous connecter avec votre email.")
                            else:
                                st.error(result["message"])
                    else:
                        st.error("Les mots de passe ne correspondent pas")
                else:
                    st.error("Veuillez remplir tous les champs obligatoires (*)")
    
    # Bouton retour
    if st.button("← Retour à l'accueil"):
        st.session_state.page = 'accueil'
        st.rerun()

def show_inscription():
    # Cette fonction n'est plus utilisée car l'inscription est dans les onglets
    pass

def show_dashboard():
    # Affichage personnalisé avec les données utilisateur
    if st.session_state.user_data:
        user_data = st.session_state.user_data
        st.title(f"🌱 Dashboard Maintso Vola - Bienvenue {user_data['prenoms'] or user_data['nom']}")
        
        # Afficher le profil utilisateur
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("### Votre espace d'analyse de données agricoles")
        with col2:
            st.markdown(f"**Email:** {user_data['email']}")
            if user_data['investir']:
                st.success("👨‍💼 Investisseur")
            if user_data['cherche_investisseurs']:
                st.info("🌱 Porteur de projet")
    else:
        st.title("🌱 Dashboard Maintso Vola")
        st.markdown("Bienvenue dans votre espace d'analyse de données agricoles !")
    
    # Sidebar for file upload and configuration
    with st.sidebar:
        st.header("📁 Téléchargement de données")
        
        # Bouton de déconnexion
        if st.button("🚪 Se déconnecter"):
            st.session_state.user_logged_in = False
            st.session_state.user_data = None
            st.session_state.page = 'accueil'
            st.rerun()
        
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
