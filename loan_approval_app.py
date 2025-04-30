import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="💰",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .title {
        font-size: 42px !important;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 20px;
    }
    .subtitle {
        font-size: 24px;
        color: #4B5563;
        margin-bottom: 30px;
    }
    .success {
        background-color: #D1FAE5;
        padding: 20px;
        border-radius: 10px;
        color: #065F46;
    }
    .danger {
        background-color: #FEE2E2;
        padding: 20px;
        border-radius: 10px;
        color: #991B1B;
    }
    .info-card {
        background-color: #F3F4F6;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Function to create and train model
@st.cache_resource
def create_model():
    # Create synthetic dataset for loan approval
    np.random.seed(42)
    n_samples = 5000
    
    # Generate synthetic data
    data = {
        'Gender': np.random.choice(['Male', 'Female'], size=n_samples),
        'Married': np.random.choice(['Yes', 'No'], size=n_samples),
        'Dependents': np.random.choice(['0', '1', '2', '3+'], size=n_samples),
        'Education': np.random.choice(['Graduate', 'Not Graduate'], size=n_samples),
        'Self_Employed': np.random.choice(['Yes', 'No'], size=n_samples),
        'ApplicantIncome': np.random.randint(1000, 10000, size=n_samples),
        'CoapplicantIncome': np.random.randint(0, 5000, size=n_samples),
        'LoanAmount': np.random.randint(50, 600, size=n_samples),
        'Loan_Amount_Term': np.random.choice([360, 180, 120, 300, 240, 60, 480], size=n_samples),
        'Credit_History': np.random.choice([0, 1], size=n_samples, p=[0.2, 0.8]),  # 0=bad, 1=good
        'Property_Area': np.random.choice(['Urban', 'Semiurban', 'Rural'], size=n_samples)
    }
    
    # Create decision rules for loan approval (simplified model)
    df = pd.DataFrame(data)
    
    # Rule-based loan approval with some randomness
    conditions = [
        # Good credit history increases approval chance
        (df['Credit_History'] == 1, 0.8),
        # Bad credit history reduces approval chance
        (df['Credit_History'] == 0, 0.3),
        # High income increases approval chance
        (df['ApplicantIncome'] > 5000, 0.2),
        # Low income decreases approval chance
        (df['ApplicantIncome'] < 2500, -0.15),
        # Higher loan amount decreases approval chance
        (df['LoanAmount'] > 300, -0.1),
        # Graduate education increases approval chance
        (df['Education'] == 'Graduate', 0.1),
        # Urban property increases approval chance
        (df['Property_Area'] == 'Urban', 0.05),
        # Rural property slightly decreases approval chance
        (df['Property_Area'] == 'Rural', -0.05)
    ]
    
    # Start with baseline probability
    approval_probability = 0.5
    
    # Apply all rules
    for condition, weight in conditions:
        approval_probability += np.where(condition, weight, 0)
    
    # Ensure probabilities are between 0 and 1
    approval_probability = np.clip(approval_probability, 0.1, 0.9)
    
    # Generate final approval status with some randomness
    df['Loan_Status'] = np.random.binomial(1, approval_probability)
    df['Loan_Status'] = df['Loan_Status'].map({1: 'Y', 0: 'N'})
    
    # Split data
    X = df.drop('Loan_Status', axis=1)
    y = (df['Loan_Status'] == 'Y').astype(int)
    
    # Define categorical and numerical features
    categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    numerical_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    
    # Create preprocessing pipeline
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    numerical_transformer = StandardScaler()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Create and train model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    return model, df

# Create and load the model
model, df = create_model()

# App header
st.markdown("<h1 class='title'>Loan Approval Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Enter your details to check loan eligibility</p>", unsafe_allow_html=True)

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["Loan Application", "Model Insights", "About"])

with tab1:
    st.markdown("<h2>Personal & Loan Information</h2>", unsafe_allow_html=True)
    
    # Create two columns for the form
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='info-card'>", unsafe_allow_html=True)
        st.subheader("Personal Details")
        gender = st.selectbox("Gender", options=["Male", "Female"])
        married = st.selectbox("Marital Status", options=["Yes", "No"])
        dependents = st.selectbox("Number of Dependents", options=["0", "1", "2", "3+"])
        education = st.selectbox("Education", options=["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", options=["No", "Yes"])
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='info-card'>", unsafe_allow_html=True)
        st.subheader("Financial Details")
        applicant_income = st.number_input("Monthly Income (USD)", min_value=0, value=3000)
        coapplicant_income = st.number_input("Co-applicant Income (USD, if any)", min_value=0, value=0)
        loan_amount = st.number_input("Loan Amount (in thousands USD)", min_value=10, value=100)
        loan_term = st.selectbox("Loan Term (in months)", options=[60, 120, 180, 240, 300, 360, 480])
        credit_history = st.selectbox("Credit History (1 = Good, 0 = Bad)", options=[1, 0])
        property_area = st.selectbox("Property Area", options=["Urban", "Semiurban", "Rural"])
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Annual income calculation
    annual_income = applicant_income * 12
    total_annual_income = annual_income + (coapplicant_income * 12)
    
    # Calculate some financial metrics
    debt_to_income = (loan_amount * 1000) / total_annual_income if total_annual_income > 0 else 0
    
    # Display financial summary
    st.markdown("<h3>Financial Summary</h3>", unsafe_allow_html=True)
    
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.metric("Annual Income", f"${annual_income:,.2f}")
    
    with metric_col2:
        st.metric("Total Household Income", f"${total_annual_income:,.2f}")
    
    with metric_col3:
        st.metric("Debt-to-Income Ratio", f"{debt_to_income:.2f}")
    
    # Button to predict
    predict_btn = st.button("Check Loan Eligibility", type="primary")
    
    if predict_btn:
        # Prepare input data
        input_data = pd.DataFrame({
            'Gender': [gender],
            'Married': [married],
            'Dependents': [dependents],
            'Education': [education],
            'Self_Employed': [self_employed],
            'ApplicantIncome': [applicant_income * 12],  # Convert to annual income
            'CoapplicantIncome': [coapplicant_income * 12],  # Convert to annual income
            'LoanAmount': [loan_amount],
            'Loan_Amount_Term': [loan_term],
            'Credit_History': [credit_history],
            'Property_Area': [property_area]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0][1]
        
        # Display result
        st.markdown("<h3>Loan Eligibility Result</h3>", unsafe_allow_html=True)
        
        if prediction == 1:
            st.markdown(f"""
            <div class='success'>
                <h3>Congratulations! Your loan is likely to be approved.</h3>
                <p>Approval Probability: {prediction_proba:.2%}</p>
                <p>Based on your profile, you have a good chance of loan approval. The next steps would be document verification and final processing.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='danger'>
                <h3>Your loan application may face challenges.</h3>
                <p>Approval Probability: {prediction_proba:.2%}</p>
                <p>Based on your current profile, loan approval might be difficult. Consider improving your credit score, reducing the loan amount, or adding a co-applicant with a strong financial profile.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Factors that influenced the decision
        st.subheader("Key Factors Influencing the Decision")
        
        factors = []
        if credit_history == 0:
            factors.append("❌ Poor credit history significantly reduces approval chances")
        else:
            factors.append("✅ Good credit history improves approval chances")
            
        if applicant_income * 12 < 30000:
            factors.append("❌ Annual income below $30,000 may be insufficient")
        elif applicant_income * 12 > 60000:
            factors.append("✅ Strong annual income improves approval chances")
            
        if debt_to_income > 0.4:
            factors.append("❌ High debt-to-income ratio (>40%)")
        else:
            factors.append("✅ Healthy debt-to-income ratio")
            
        if loan_amount > 300:
            factors.append("❌ Large loan amount relative to income")
        
        if education == "Graduate":
            factors.append("✅ Higher education level is favorable")
            
        for factor in factors:
            st.markdown(f"- {factor}")

with tab2:
    st.markdown("<h2>Model Insights & Data Analysis</h2>", unsafe_allow_html=True)
    
    st.write("Below are insights from our loan approval model based on historical data:")
    
    # Feature importance chart
    st.subheader("Factors Affecting Loan Approval")
    
    # Get feature names
    categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    numerical_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    
    # Simplified feature importance (for demonstration)
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
    
    # Sort importance
    feature_importance = {k: v for k, v in sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)}
    
    # Plot feature importance
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(list(feature_importance.keys()), list(feature_importance.values()), color='cornflowerblue')
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance in Loan Approval')
    
    # Add percentage annotations
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.0%}', 
                va='center', fontsize=10)
    
    st.pyplot(fig)
    
    # Show approval rates by different factors
    st.subheader("Approval Rate Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Credit History vs Approval
        credit_approval = pd.crosstab(df['Credit_History'], df['Loan_Status'])
        credit_approval['Approval_Rate'] = credit_approval['Y'] / (credit_approval['Y'] + credit_approval['N'])
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(['Bad Credit (0)', 'Good Credit (1)'], credit_approval['Approval_Rate'], color=['firebrick', 'forestgreen'])
        ax.set_ylabel('Approval Rate')
        ax.set_title('Loan Approval Rate by Credit History')
        ax.set_ylim(0, 1)
        
        # Add percentage annotations
        for i, v in enumerate(credit_approval['Approval_Rate']):
            ax.text(i, v + 0.05, f'{v:.0%}', ha='center')
            
        st.pyplot(fig)
    
    with col2:
        # Education vs Approval
        education_approval = pd.crosstab(df['Education'], df['Loan_Status'])
        education_approval['Approval_Rate'] = education_approval['Y'] / (education_approval['Y'] + education_approval['N'])
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(education_approval.index, education_approval['Approval_Rate'], color='darkorange')
        ax.set_ylabel('Approval Rate')
        ax.set_title('Loan Approval Rate by Education')
        ax.set_ylim(0, 1)
        
        # Add percentage annotations
        for i, v in enumerate(education_approval['Approval_Rate']):
            ax.text(i, v + 0.05, f'{v:.0%}', ha='center')
            
        st.pyplot(fig)
    
    # Income distribution
    st.subheader("Income Distribution by Loan Status")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(data=df, x='ApplicantIncome', hue='Loan_Status', ax=ax, palette=['firebrick', 'forestgreen'])
    ax.set_xlabel('Annual Income')
    ax.set_title('Income Distribution by Loan Status')
    ax.legend(title='Loan Status', labels=['Rejected', 'Approved'])
    
    st.pyplot(fig)
    
    # Tips for approval
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
        st.markdown(f"- {tip}")

with tab3:
    st.markdown("<h2>About This Loan Prediction System</h2>", unsafe_allow_html=True)
    
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

# Add footer
st.markdown("""
<div style='text-align: center; margin-top: 40px; padding: 20px; background-color: #f8f9fa; border-radius: 5px;'>
    <p>Developed with ❤️ using Streamlit and Machine Learning</p>
</div>
""", unsafe_allow_html=True)
