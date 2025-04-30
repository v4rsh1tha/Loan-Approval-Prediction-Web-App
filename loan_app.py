import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="💸",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .main {
        background: linear-gradient(135deg, #F6F8FA 0%, #E8ECEF 100%);
        padding: 2rem;
        font-family: 'Inter', sans-serif;
        min-height: 100vh;
    }

    .title {
        font-size: 36px !important;
        font-weight: 700;
        color: #1F2A44;
        margin-bottom: 16px;
        background: linear-gradient(to right, #1F2A44, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: fadeIn 1s ease-in;
    }

    .subtitle {
        font-size: 20px;
        color: #4B5563;
        margin-bottom: 24px;
        animation: slideIn 1s ease-in;
    }

    .gh-card {
        background: #FFFFFF;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .gh-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
    }

    .gh-card-header {
        font-size: 18px;
        font-weight: 600;
        color: #1F2A44;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 2px solid #3B82F6;
    }

    .success {
        background: linear-gradient(to right, #10B981, #34D399);
        padding: 20px;
        border-radius: 12px;
        color: #FFFFFF;
        margin-bottom: 20px;
        animation: popIn 0.5s ease-in;
    }

    .danger {
        background: linear-gradient(to right, #EF4444, #F87171);
        padding: 20px;
        border-radius: 12px;
        color: #FFFFFF;
        margin-bottom: 20px;
        animation: popIn 0.5s ease-in;
    }

    .info-box {
        background: #EFF6FF;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid #DBEAFE;
    }

    .stButton > button {
        background: linear-gradient(to right, #3B82F6, #2563EB);
        color: #FFFFFF;
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 16px;
        font-weight: 600;
        border: none;
        transition: background 0.3s ease, transform 0.2s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(to right, #2563EB, #1D4ED8);
        transform: scale(1.05);
    }

    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 2px solid #DBEAFE;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 12px 16px;
        color: #6B7280;
        font-size: 16px;
        font-weight: 500;
        transition: color 0.3s ease;
    }

    .stTabs [aria-selected="true"] {
        color: #1F2A44;
        border-bottom: 3px solid #3B82F6;
        font-weight: 700;
    }

    .stMetric {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }

    .stMetric [data-testid="stMetricLabel"] {
        font-size: 16px;
        color: #4B5563;
    }

    .stMetric [data-testid="stMetricValue"] {
        font-size: 20px;
        color: #1F2A44;
        font-weight: 700;
    }

    .stSelectbox [data-baseweb="select"],
    .stNumberInput [data-baseweb="input"],
    .stTextInput [data-baseweb="input"] {
        border-radius: 8px;
        border: 1px solid #DBEAFE;
        background: #F9FAFB;
        padding: 10px;
    }

    .factor-item {
        padding: 10px 0;
        border-bottom: 1px solid #E5E7EB;
        font-size: 16px;
    }

    .gh-footer {
        text-align: center;
        margin-top: 40px;
        padding: 20px;
        background: #FFFFFF;
        border-radius: 12px;
        color: #4B5563;
        font-size: 16px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    @keyframes slideIn {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }

    @keyframes popIn {
        from { transform: scale(0.9); opacity: 0; }
        to { transform: scale(1); opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def create_model():
    np.random.seed(42)
    n_samples = 5000
    data = {
        'Gender': np.random.choice(['Male', 'Female'], size=n_samples),
        'Married': np.random.choice(['Yes', 'No'], size=n_samples),
        'Dependents': np.random.choice(['0', '1', '2', '3+'], size=n_samples),
        'Education': np.random.choice(['Graduate', 'Not Graduate'], size=n_samples),
        'Self_Employed': np.random.choice(['Yes', 'No'], size=n_samples),
        'ApplicantIncome': np.random.randint(100000, 1000000, size=n_samples),
        'CoapplicantIncome': np.random.randint(0, 500000, size=n_samples),
        'LoanAmount': np.random.randint(5000, 60000, size=n_samples),
        'Loan_Amount_Term': np.random.choice([360, 180, 120, 300, 240, 60, 480], size=n_samples),
        'Credit_History': np.random.choice([0, 1], size=n_samples, p=[0.2, 0.8]),
        'Property_Area': np.random.choice(['Urban', 'Semiurban', 'Rural'], size=n_samples)
    }
    df = pd.DataFrame(data)
    conditions = [
        (df['Credit_History'] == 1, 0.8),
        (df['Credit_History'] == 0, 0.3),
        (df['ApplicantIncome'] > 500000, 0.2),
        (df['ApplicantIncome'] < 250000, -0.15),
        (df['LoanAmount'] > 30000, -0.1),
        (df['Education'] == 'Graduate', 0.1),
        (df['Property_Area'] == 'Urban', 0.05),
        (df['Property_Area'] == 'Rural', -0.05)
    ]
    approval_probability = 0.5
    for condition, weight in conditions:
        approval_probability += np.where(condition, weight, 0)
    approval_probability = np.clip(approval_probability, 0.1, 0.9)
    df['Loan_Status'] = np.random.binomial(1, approval_probability)
    df['Loan_Status'] = df['Loan_Status'].map({1: 'Y', 0: 'N'})
    X = df.drop('Loan_Status', axis=1)
    y = (df['Loan_Status'] == 'Y').astype(int)
    categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    numerical_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    numerical_transformer = StandardScaler()
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    return model, df

model, df = create_model()

st.markdown("""
<div style="display: flex; align-items: center; margin-bottom: 8px;">
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="36" height="36" style="margin-right: 12px;">
        <path fill="#3B82F6" d="M12.75 2.75a.75.75 0 0 0-1.5 0V4.5H9.276a1.75 1.75 0 0 0-.985.303L6.596 5.957A.25.25 0 0 1 6.455 6H2.353a.75.75 0 1 0 0 1.5H3.93L.563 15.18a.762.762 0 0 0 .21.88c.08.064.161.125.309.221.186.121.452.278.792.433.68.311 1.662.62 2.876.62a6.919 6.919 0 0 0 2.876-.62c.34-.155.606-.312.792-.433.15-.097.23-.158.31-.223a.748.748 0 0 0 .209-.878L5.569 7.5h.886c.351 0 .694-.106.984-.303l1.696-1.154A.25.25 0 0 1 9.275 6h1.975v14.5H6.763a.75.75 0 0 0 0 1.5h10.474a.75.75 0 0 0 0-1.5H12.75V6h1.974c.05 0 .1.015.14.043l1.697 1.154c.29.197.633.303.984.303h.886l-3.368 7.68a.75.75 0 0 0 .23.896c.012.009 0 0 .002 0a3.154 3.154 0 0 0 .31.206c.185.112.45.256.79.4a7.343 7.343 0 0 0 2.855.568 7.343 7.343 0 0 0 2.856-.569c.338-.143.604-.287.79-.399a3.5 3.5 0 0 0 .31-.206.75.75 0 0 0 .23-.896L20.07 7.5h1.578a.75.75 0 0 0 0-1.5h-4.102a.25.25 0 0 1-.14-.043l-1.697-1.154a1.75 1.75 0 0 0-.984-.303H12.75V2.75zM2.193 15.198a5.418 5.418 0 0 0 2.557.635 5.418 5.418 0 0 0 2.557-.635L4.75 9.368Zm14.51-.024c.082.04.174.083.275.126.53.223 1.305.45 2.272.45a5.846 5.846 0 0 0 2.547-.576L19.25 9.367Z"/>
    </svg>
    <h1 class='title'>Loan Approval Prediction</h1>
</div>
""", unsafe_allow_html=True)

st.markdown("<p class='subtitle'>Discover your loan eligibility with our smart prediction system</p>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📋 Application", "📊 Insights", "ℹ️ About"])

with tab1:
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 16px;">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="28" height="28" style="margin-right: 10px;">
            <path fill="#3B82F6" d="M12 1c6.075 0 11 4.925 11 11s-4.925 11-11 11S1 18.075 1 12 5.925 1 12 1Zm0 1.75a9.25 9.25 0 1 0 0 18.5 9.25 9.25 0 0 0 0-18.5Zm.5 6.75h3.5a.75.75 0 0 1 0 1.5h-3.5V14h3.5a.75.75 0 0 1 0 1.5h-3.5v2.25a.75.75 0 0 1-1.5 0v-2.25h-3.5a.75.75 0 0 1 0-1.5h3.5v-3h-3.5a.75.75 0 0 1 0-1.5h3.5V6.75a.75.75 0 0 1 1.5 0v2.25Z"/>
        </svg>
        <h2 style="font-size: 24px; font-weight: 700; color: #1F2A44; margin: 0;">Enter Your Details</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='gh-card'>", unsafe_allow_html=True)
        st.markdown("<div class='gh-card-header'>Personal Information</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 12px;">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="20" height="20" style="margin-right: 10px;">
                <path fill="#3B82F6" d="M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0ZM5.614 13.744c-.914-.081-1.856-.43-2.575-1.008a4.16 4.16 0 0 1-1.29-1.968c1.412.364 2.85.566 4.251.566 1.402 0 2.84-.202 4.252-.566a4.16 4.16 0 0 1-1.29 1.968c-.72.578-1.662.927-2.576 1.008a.751.751 0 0 1-.766-.744.75.75 0 0 1 .744-.766Zm2.386-4.147a2.502 2.502 0 1 0 0-5.004 2.502 2.502 0 0 0 0 5.004Z"/>
            </svg>
            <span style="font-weight: 600; color: #1F2A44;">Your Name</span>
        </div>
        """, unsafe_allow_html=True)
        applicant_name = st.text_input("", placeholder="Enter your full name", label_visibility="collapsed")
        
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 12px; margin-top: 16px;">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="20" height="20" style="margin-right: 10px;">
                <path fill="#3B82F6" d="M8 16A8 8 0 1 1 8 0a8 8 0 0 1 0 16Zm.847-8.145a2.502 2.502 0 1 0-1.694 0C5.471 8.261 4 9.775 4 11.5c0 .517.196 1 .75 1h6.5c.554 0 .75-.483.75-1 0-1.725-1.47-3.239-3.153-3.645Z"/>
            </svg>
            <span style="font-weight: 600; color: #1F2A44;">Gender</span>
        </div>
        """, unsafe_allow_html=True)
        gender = st.selectbox("", options=["Male", "Female"], label_visibility="collapsed")
        
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 12px; margin-top: 16px;">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="20" height="20" style="margin-right: 10px;">
                <path fill="#3B82F6" d="M4.25 2A2.25 2.25 0 0 0 2 4.25v7.5A2.25 2.25 0 0 0 4.25 14h7.5A2.25 2.25 0 0 0 14 11.75v-7.5A2.25 2.25 0 0 0 11.75 2h-7.5Zm0 1.5h7.5a.75.75 0 0 1 .75.75v7.5a.75.75 0 0 1-.75.75h-7.5a.75.75 0 0 1-.75-.75v-7.5a.75.75 0 0 1 .75-.75Zm-1.5-.25a.75.75 0 0 1 .75-.75h5.5a.75.75 0 0 1 0 1.5h-5.5a.75.75 0 0 1-.75-.75ZM5.75 2a.75.75 0 0 1 .75.75v1.5a.75.75 0 0 1-1.5 0v-1.5A.75.75 0 0 1 5.75 2Zm4 0a.75.75 0 0 1 .75.75v1.5a.75.75 0 0 1-1.5 0v-1.5A.75.75 0 0 1 9.75 2ZM8 10a2 2 0 1 1-3.999.001A2 2 0 0 1 8 10Z"/>
            </svg>
            <span style="font-weight: 600; color: #1F2A44;">Marital Status</span>
        </div>
        """, unsafe_allow_html=True)
        married = st.selectbox("", options=["Yes", "No"], label_visibility="collapsed")
        
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 12px; margin-top: 16px;">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="20" height="20" style="margin-right: 10px;">
                <path fill="#3B82F6" d="M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0ZM1.5 8a6.5 6.5 0 1 0 13 0 6.5 6.5 0 0 0-13 0Zm4.879-2.773 4.264 2.559a.25.25 0 0 1 0 .428l-4.264 2.559A.25.25 0 0 1 6 10.559V5.442a.25.25 0 0 1 .379-.215Z"/>
            </svg>
            <span style="font-weight: 600; color: #1F2A44;">Number of Dependents</span>
        </div>
        """, unsafe_allow_html=True)
        dependents = st.selectbox("", options=["0", "1", "2", "3+"], label_visibility="collapsed")
        
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 12px; margin-top: 16px;">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="20" height="20" style="margin-right: 10px;">
                <path fill="#3B82F6" d="M7.693 1.066a.75.75 0 0 1 .614 0l7.25 3.25a.75.75 0 0 1 0 1.368L13 6.831v2.794c0 1.024-.816 1.749-1.54 2.169-.7.405-1.538.7-2.317.919a13.354 13.354 0 0 1-1.376.272.75.75 0 0 1-.756-.44.753.753 0 0 1-.175-.525c0-.715-.176-1.595-1.7-2.857-.309-.261-.326-.695-.083-.979.255-.3.73-.324 1.012-.062.524.479.97.894 1.316 1.243.66.467.594-.003.618-.212.024-.213.248-1.329 1.128-2.127l-.29-2.245-.088-.575a.75.75 0 0 1 .537-.828l.883-.192-5.77-2.584a.75.75 0 0 1 .6-1.375ZM9 4.75a.75.75 0 1 0-1.5 0v.261a7.22 7.22 0 0 0-1.35.526.75.75 0 0 0 .7 1.326 5.88 5.88 0 0 1 1.668-.628A.75.75 0 0 0 9 5.522V4.75ZM11.55 3.19a.75.75 0 0 0-.968.432l-.644 1.61a.75.75 0 1 0 1.4.558l.643-1.61a.75.75 0 0 0-.43-.99Z"/>
            </svg>
            <span style="font-weight: 600; color: #1F2A44;">Education</span>
        </div>
        """, unsafe_allow_html=True)
        education = st.selectbox("", options=["Graduate", "Not Graduate"], label_visibility="collapsed")
        
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 12px; margin-top: 16px;">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="20" height="20" style="margin-right: 10px;">
                <path fill="#3B82F6" d="M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0ZM1.5 8a6.5 6.5 0 1 0 13 0 6.5 6.5 0 0 0-13 0Zm9.28-1.72-4.5 4.5a.75.75 0 0 1-1.06 0l-2-2a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018l1.47 1.47 3.97-3.97a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042Z"/>
            </svg>
            <span style="font-weight: 600; color: #1F2A44;">Self Employed</span>
        </div>
        """, unsafe_allow_html=True)
        self_employed = st.selectbox("", options=["No", "Yes"], label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='gh-card'>", unsafe_allow_html=True)
        st.markdown("<div class='gh-card-header'>Financial Information</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 12px;">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="20" height="20" style="margin-right: 10px;">
                <path fill="#3B82F6" d="M7.75 0a.75.75 0 0 1 .75.75V2h1.25a.75.75 0 0 1 0 1.5H8.5v2.75a.75.75 0 0 1-1.5 0V3.5H4.75a.75.75 0 0 1 0-1.5H7V.75A.75.75 0 0 1 7.75 0ZM8 13.22A5.751 5.751 0 0 1 2.25 7.5a.75.75 0 0 1 1.5 0 4.25 4.25 0 0 0 8.5 0 .75.75 0 0 1 1.5 0A5.751 5.751 0 0 1 8 13.22Z"/>
            </svg>
            <span style="font-weight: 600; color: #1F2A44;">Monthly Income (INR)</span>
        </div>
        """, unsafe_allow_html=True)
        applicant_income = st.number_input("", min_value=0, value=30000, label_visibility="collapsed")
        
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 12px; margin-top: 16px;">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="20" height="20" style="margin-right: 10px;">
                <path fill="#3B82F6" d="M5.5 3.5a2 2 0 1 0 0 4 2 2 0 0 0 0-4ZM2 5.5a3.5 3.5 0 1 1 5.898 2.549 5.508 5.508 0 0 1 3.034 4.084.75.75 0 1 1-1.482.235 4 4 0 0 0-7.9 0 .75.75 0 0 1-1.482-.236A5.507 5.507 0 0 1 3.102 8.05 3.493 3.493 0 0 1 2 5.5ZM11 4a.75.75 0 1 1 0 1.5 1.5 1.5 0 0 0 0 3 .75.75 0 1 1 0 1.5 3 3 0 1 1 0-6Z"/>
            </svg>
            <span style="font-weight: 600; color: #1F2A44;">Co-applicant Income (INR, if any)</span>
        </div>
        """, unsafe_allow_html=True)
        coapplicant_income = st.number_input("", min_value=0, value=0, label_visibility="collapsed")
        
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 12px; margin-top: 16px;">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="20" height="20" style="margin-right: 10px;">
                <path fill="#3B82F6" d="M10.75 9a.75.75 0 0 1 .75.75v1.5a.75.75 0 0 1-1.5 0v-1.5a.75.75 0 0 1 .75-.75Zm-5.5 0a.75.75 0 0 1 .75.75v1.5a.75.75 0 0 1-1.5 0v-1.5A.75.75 0 0 1 5.25 9ZM12 1a3 3 0 0 1 3 3v.25h.75a.75.75 0 0 1 0 1.5H15v2.75h.75a.75.75 0 0 1 0 1.5H15v2.75h.75a.75.75 0 0 1 0 1.5H15V15a1 1 0 0 1-1 1H2a1 1 0 0 1-1-1v-.75h-.75a.75.75 0 0 1 0-1.5H1v-2.75h-.75a.75.75 0 0 1 0-1.5H1V5.5h-.75a.75.75 0 0 1 0-1.5H1V3a3 3 0 0 1 3-3h8Zm1.5 6.5v-1h-9v1h9ZM3 4v-.5A1.5 1.5 0 0 1 4.5 2h7A1.5 1.5 0 0 1 13 3.5V4H3Zm1.5 5.25a.75.75 0 1 0-1.5 0v1.5a.75.75 0 0 0 1.5 0v-1.5ZM13 14V13H3v1h10Z"/>
            </svg>
            <span style="font-weight: 600; color: #1F2A44;">Loan Amount (in thousands INR)</span>
        </div>
        """, unsafe_allow_html=True)
        loan_amount = st.number_input("", min_value=1000, value=10000, label_visibility="collapsed")
        
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 12px; margin-top: 16px;">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="20" height="20" style="margin-right: 10px;">
                <path fill="#3B82F6" d="M4.75 0a.75.75 0 0 1 .75.75V3h5V.75a.75.75 0 0 1 1.5 0V3h1.25c.966 0 1.75.784 1.75 1.75V12.5A3.5 3.5 0 0 1 11.5 16h-7A3.5 3.5 0 0 1 1 12.5V4.75C1 3.784 1.784 3 2.75 3H4V.75A.75.75 0 0 1 4.75 0ZM2.5 7.5v5c0 1.1.9 2 2 2h7c1.1 0 2-.9 2-2v-5H2.5ZM8.75 9.75a.75.75 0 0 0 0 1.5h2.5a.75.75 0 0 0 0-1.5h-2.5Z"/>
            </svg>
            <span style="font-weight: 600; color: #1F2A44;">Loan Term (months)</span>
        </div>
        """, unsafe_allow_html=True)
        loan_term = st.selectbox("", options=[360, 180, 120, 300, 240, 60, 480], label_visibility="collapsed")
        
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 12px; margin-top: 16px;">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="20" height="20" style="margin-right: 10px;">
                <path fill="#3B82F6" d="M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0ZM1.5 8a6.5 6.5 0 1 0 13 0 6.5 6.5 0 0 0-13 0Zm9.28-1.72-4.5 4.5a.75.75 0 0 1-1.06 0l-2-2a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018l1.47 1.47 3.97-3.97a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042Z"/>
            </svg>
            <span style="font-weight: 600; color: #1F2A44;">Credit History</span>
        </div>
        """, unsafe_allow_html=True)
        credit_history = st.selectbox("", options=[1, 0], format_func=lambda x: "Good" if x == 1 else "Bad", label_visibility="collapsed")
        
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 12px; margin-top: 16px;">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="20" height="20" style="margin-right: 10px;">
                <path fill="#3B82F6" d="M1.5 2A1.5 1.5 0 0 0 0 3.5v9A1.5 1.5 0 0 0 1.5 14h13a1.5 1.5 0 0 0 1.5-1.5v-9A1.5 1.5 0 0 0 14.5 2h-13Zm0 1h13a.5.5 0 0 1 .5.5v1.793L8.354 8.854a.5.5 0 0 1-.708 0L1 5.293V3.5a.5.5 0 0 1 .5-.5Zm0 3.236l6.146 3.561a1.5 1.5 0 0 0 1.708 0L15 6.236v6.264a.5.5 0 0 1-.5.5h-13a.5.5 0 0 1-.5-.5V6.236Z"/>
            </svg>
            <span style="font-weight: 600; color: #1F2A44;">Property Area</span>
        </div>
        """, unsafe_allow_html=True)
        property_area = st.selectbox("", options=["Urban", "Semiurban", "Rural"], label_visibility="collapsed")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    annual_income = applicant_income * 12
    total_annual_income = annual_income + (coapplicant_income * 12)
    debt_to_income = (loan_amount * 1000) / total_annual_income if total_annual_income > 0 else 0
    
    st.markdown("<h3 style='font-weight: 700; color: #1F2A44;'>Financial Overview</h3>", unsafe_allow_html=True)
    
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.metric("Annual Income", f"₹{annual_income:,.2f}")
    
    with metric_col2:
        st.metric("Total Household Income", f"₹{total_annual_income:,.2f}")
    
    with metric_col3:
        st.metric("Debt-to-Income Ratio", f"{debt_to_income:.2f}")
    
    predict_btn = st.button("Check Loan Eligibility", type="primary")
    
    if predict_btn:
        input_data = pd.DataFrame({
            'Gender': [gender],
            'Married': [married],
            'Dependents': [dependents],
            'Education': [education],
            'Self_Employed': [self_employed],
            'ApplicantIncome': [applicant_income * 12],
            'CoapplicantIncome': [coapplicant_income * 12],
            'LoanAmount': [loan_amount],
            'Loan_Amount_Term': [loan_term],
            'Credit_History': [credit_history],
            'Property_Area': [property_area]
        })
        
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0][1]
        
        st.markdown("<h3 style='font-weight: 700; color: #1F2A44;'>Loan Eligibility Result</h3>", unsafe_allow_html=True)
        
        if prediction == 1:
            st.markdown(f"""
            <div class='success'>
                <h3>Congratulations, {applicant_name or 'Applicant'}!</h3>
                <p>Your loan is likely to be approved with a probability of {prediction_proba:.2%}.</p>
                <p>You're on track for loan approval! Next steps include document verification and final processing.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='danger'>
                <h3>Sorry, {applicant_name or 'Applicant'}!</h3>
                <p>Your loan application may face challenges with a probability of {prediction_proba:.2%}.</p>
                <p>Consider improving your credit score, reducing the loan amount, or adding a co-applicant with a strong financial profile.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.subheader("Key Factors Influencing the Decision")
        
        factors = []
        if credit_history == 0:
            factors.append("❌ Poor credit history significantly reduces approval chances")
        else:
            factors.append("✅ Good credit history improves approval chances")
            
        if applicant_income * 12 < 300000:
            factors.append("❌ Annual income below ₹3,00,000 may be insufficient")
        elif applicant_income * 12 > 600000:
            factors.append("✅ Strong annual income improves approval chances")
            
        if debt_to_income > 0.4:
            factors.append("❌ High debt-to-income ratio (>40%)")
        else:
            factors.append("✅ Healthy debt-to-income ratio")
            
        if loan_amount > 30000:
            factors.append("❌ Large loan amount relative to income")
        
        if education == "Graduate":
            factors.append("✅ Higher education level is favorable")
            
        for factor in factors:
            st.markdown(f"<div class='factor-item'>- {factor}</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<h2 style='font-weight: 700; color: #1F2A44;'>Model Insights & Data Analysis</h2>", unsafe_allow_html=True)
    
    st.write("Explore key insights from our loan approval model based on historical data:")
    
    st.subheader("Factors Affecting Loan Approval")
    
    categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    numerical_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    
    feature_importance = {
        'Credit_History': 0.28,
        'ApplicantIncome': 0.18,
        'LoanAmount': 0.15,
        'CoapplicantIncome': 0.12,
        'Property_Area': 0.08,
        'Education': 0.07,
        'Married': 0.05,
        'Dependents': 0.03,
        'Loan_Amount_Term': 0.02,
        'Gender': 0.01,
        'Self_Employed': 0.01
    }
    
    feature_importance = {k: v for k, v in sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(list(feature_importance.keys()), list(feature_importance.values()), color='#3B82F6')
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance in Loan Approval', fontsize=16, fontweight='bold')
    ax.set_facecolor('#F9FAFB')
    fig.patch.set_facecolor('#F9FAFB')
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.0%}', 
                va='center', fontsize=10, fontweight='bold')
    
    st.pyplot(fig)
    
    st.subheader("Approval Rate Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        credit_approval = pd.crosstab(df['Credit_History'], df['Loan_Status'])
        credit_approval['Approval_Rate'] = credit_approval['Y'] / (credit_approval['Y'] + credit_approval['N'])
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(['Bad Credit (0)', 'Good Credit (1)'], credit_approval['Approval_Rate'], color=['#EF4444', '#10B981'])
        ax.set_ylabel('Approval Rate')
        ax.set_title('Loan Approval Rate by Credit History', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_facecolor('#F9FAFB')
        fig.patch.set_facecolor('#F9FAFB')
        
        for i, v in enumerate(credit_approval['Approval_Rate']):
            ax.text(i, v + 0.05, f'{v:.0%}', ha='center', fontweight='bold')
            
        st.pyplot(fig)
    
    with col2:
        education_approval = pd.crosstab(df['Education'], df['Loan_Status'])
        education_approval['Approval_Rate'] = education_approval['Y'] / (education_approval['Y'] + education_approval['N'])
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(education_approval.index, education_approval['Approval_Rate'], color='#F59E0B')
        ax.set_ylabel('Approval Rate')
        ax.set_title('Loan Approval Rate by Education', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_facecolor('#F9FAFB')
        fig.patch.set_facecolor('#F9FAFB')
        
        for i, v in enumerate(education_approval['Approval_Rate']):
            ax.text(i, v + 0.05, f'{v:.0%}', ha='center', fontweight='bold')
            
        st.pyplot(fig)
    
    st.subheader("Income Distribution by Loan Status")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(data=df, x='ApplicantIncome', hue='Loan_Status', ax=ax, palette=['#EF4444', '#10B981'])
    ax.set_xlabel('Annual Income (INR)')
    ax.set_title('Income Distribution by Loan Status', fontsize=16, fontweight='bold')
    ax.legend(title='Loan Status', labels=['Rejected', 'Approved'])
    ax.set_facecolor('#F9FAFB')
    fig.patch.set_facecolor('#F9FAFB')
    
    st.pyplot(fig)
    
    st.subheader("Tips to Improve Loan Approval Chances")
    
    tips = [
        "Maintain a good credit score and credit history",
        "Keep your debt-to-income ratio below 40%",
        "Apply for a loan amount that's reasonable given your income",
        "If possible, include a co-applicant with strong income",
        "Stable employment history improves your chances",
        "Having collateral or assets can strengthen your application"
    ]
    
    for tip in tips:
        st.markdown(f"<div class='factor-item'>- {tip}</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("<h2 style='font-weight: 700; color: #1F2A44;'>About This Loan Prediction System</h2>", unsafe_allow_html=True)
    
    st.write("""
    This loan approval prediction system uses machine learning to estimate the likelihood of your loan application being approved. The system is built using a Random Forest classifier trained on historical loan application data.
    
    ### How It Works
    
    The system considers various factors when predicting loan approval:
    
    1. **Credit History**: This is the most important factor - a good credit history significantly increases approval chances
    2. **Income**: Higher income improves your chances of loan approval
    3. **Loan Amount**: The amount requested relative to your income affects approval
    4. **Co-applicant Income**: Additional household income strengthens the application
    5. **Other Factors**: Education level, property area, marital status, and dependents
    
    ### Disclaimer
    
    This is a demonstration project and should not be used for actual loan decisions. The model is trained on synthetic data and is meant for educational purposes only. In real-world scenarios, financial institutions use more comprehensive data and sophisticated models for loan approval decisions.
    
    ### Model Information
    
    - **Algorithm**: Random Forest Classifier
    - **Features Used**: 11 features including personal information and financial details
    - **Data**: Synthetic data mimicking real-world loan approval patterns
    """)

    st.info("Note: All input data is processed locally and is not stored or transmitted elsewhere.", icon="ℹ️")

st.markdown("""
<div class='gh-footer'>
    <p>Developed with ❤️ By Varshitha...</p>
</div>
""", unsafe_allow_html=True)