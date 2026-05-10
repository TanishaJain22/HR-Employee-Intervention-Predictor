import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Set Streamlit Page config for consistent rich UI
st.set_page_config(
    page_title="HR Flight Risk Intelligence",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Aesthetic Theming injections
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    div[data-testid="stMetricValue"] { font-size: 28px; color: #1F4E79; }
    .stButton button { background-color: #1F4E79; color: white; border-radius: 8px; font-weight: 600; }
    .stButton button:hover { background-color: #2E6F9A; color: white; }
    .header-style { font-size: 42px; font-weight: 800; color: #2C3E50; margin-bottom: 0.5rem; }
    .sub-header-style { font-size: 18px; color: #7F8C8D; margin-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# CORE LOGIC - CACHED TO PREVENT RETRAIN LAGGING
# ------------------------------------------------------------------------------

def engineer_features(df):
    df = df.copy()
    # Fix static column values
    if 'Attrition' in df.columns and df['Attrition'].dtype == object:
        df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

    df.drop(columns=[c for c in ['EmployeeCount', 'Over18', 'StandardHours'] if c in df.columns], inplace=True, errors='ignore')

    # Ratio and Flag Interactions derived from HR intelligence
    df['IncomePerYear']         = df['MonthlyIncome'] / (df['YearsAtCompany'] + 1)
    df['SatisfactionScore']     = (df['JobSatisfaction'] + df['EnvironmentSatisfaction'] + df['RelationshipSatisfaction']) / 3
    df['WorkLifeStress']        = df['WorkLifeBalance'] * df['JobInvolvement']
    df['CareerGrowthRate']      = df['JobLevel'] / (df['YearsAtCompany'] + 1)
    df['PromotionLag']          = df['YearsAtCompany'] - df['YearsSinceLastPromotion']
    df['TenureRatio']           = df['YearsWithCurrManager'] / (df['YearsAtCompany'] + 1)
    df['OvertimeFlag']          = (df.get('OverTime') == 'Yes').astype(int) if 'OverTime' in df.columns else 0
    df['LowSatisfactionFlag']   = ((df['JobSatisfaction'] <= 2) | (df['EnvironmentSatisfaction'] <= 2)).astype(int)
    df['YoungAndFarFlag']       = ((df['Age'] < 30) & (df['DistanceFromHome'] > 10)).astype(int)
    df['StockRiskFlag']         = (df['StockOptionLevel'] == 0).astype(int)
    df['TotalExperience']       = df['TotalWorkingYears'] - df['YearsAtCompany']
    df['IncomeToLevelRatio']    = df['MonthlyIncome'] / (df['JobLevel'] + 1)
    
    return df

@st.cache_resource
def train_predictive_engine():
    # 1. Load Local Environment Datasets
    train_df = pd.read_csv('train_hr_data.csv')
    
    # Build the statistical baseline row for input fills
    baseline_row = train_df.copy()
    # Map modes for strings, medians for numeric
    for col in baseline_row.columns:
        if baseline_row[col].dtype in ['object', 'string']:
            baseline_row[col] = baseline_row[col].mode()[0]
        else:
            baseline_row[col] = baseline_row[col].median()
    baseline_row = baseline_row.head(1).copy()
    
    # Capture dropdown category scopes
    categorical_scopes = {
        'Department': sorted(train_df['Department'].dropna().unique().tolist()),
        'JobRole': sorted(train_df['JobRole'].dropna().unique().tolist())
    }
    
    # Engineer features immediately
    eng_train = engineer_features(train_df)
    
    # Setup feature matrix
    X = eng_train.drop(columns=['Attrition', 'EmployeeNumber'], errors='ignore')
    y = eng_train['Attrition']
    
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # 2. Preprocessing Pipeline
    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), num_cols),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols)
    ])
    
    X_p = preprocessor.fit_transform(X)
    
    # 3. Counter Class Imbalance via SMOTE
    sm = SMOTE(random_state=42)
    X_balanced, y_balanced = sm.fit_resample(X_p, y)
    
    # 4. Dynamic Soft-Voting Ensemble Initialization
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1)
    lr = LogisticRegression(max_iter=3000, C=0.5, class_weight='balanced', random_state=42, solver='saga')
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, eval_metric='logloss', random_state=42, scale_pos_weight=1)
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42)
    
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('lr', lr), ('xgb', xgb), ('gb', gb)],
        voting='soft', 
        weights=[3, 1, 3, 2]
    )
    
    # Fit entire ensemble once
    ensemble.fit(X_balanced, y_balanced)
    
    return ensemble, preprocessor, num_cols, cat_cols, baseline_row, categorical_scopes


# ------------------------------------------------------------------------------
# MAIN INTERFACE BUILDER
# ------------------------------------------------------------------------------

st.markdown('<div class="header-style">HR Employee Flight Risk Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header-style">AI-powered system translating demographic signals into actionable retention intervention targets.</div>', unsafe_allow_html=True)

# Initialize model and fetch parameters via cache
with st.spinner("Initializing Synthetic AI Engine (Training model in-memory)..."):
    model, processor, num_cols, cat_cols, baseline, categories = train_predictive_engine()

# Load Validation Dataset for scoring
validate_raw = pd.read_csv('validate_hr_data.csv')
validate_eng = engineer_features(validate_raw)

# Scoring Run
X_val = validate_eng.drop(columns=['Attrition', 'EmployeeNumber'], errors='ignore')
X_val_p = processor.transform(X_val)

probs = model.predict_proba(X_val_p)[:, 1]
threshold = 0.35
preds = (probs >= threshold).astype(int)

# Top metrics dashboard row
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Validation Population", f"{len(validate_raw)} Employees")
with col2:
    total_at_risk = sum(preds)
    st.metric("Identified At-Risk", f"{total_at_risk}", delta=f"{(total_at_risk/len(validate_raw)*100):.1f}% of Total", delta_color="inverse")
with col3:
    avg_prob = probs.mean() * 100
    st.metric("Aggregate Churn Probability", f"{avg_prob:.1f}%")

st.write("---")

# View routing setup
view_tab1, view_tab2, view_tab3 = st.tabs(["📊 Risk Diagnostics", "📋 Actionable Target Roster", "🔍 Individual Predictor"])

with view_tab1:
    st.subheader("Structural Drivers of Talent Leakage")
    
    # Feature importance calculation using RF base estimator
    cat_feat = list(processor.named_transformers_['cat'].named_steps['oh'].get_feature_names_out(cat_cols))
    all_feat = num_cols + cat_feat
    
    fi = pd.DataFrame({
        'Predictor Vector': all_feat,
        'Importance Weight': model.named_estimators_['rf'].feature_importances_
    }).sort_values('Importance Weight', ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=fi, x='Importance Weight', y='Predictor Vector', palette='viridis', ax=ax)
    ax.set_title("Top 10 Structural Determinants for Attrition Prediction", fontsize=12, fontweight='bold')
    ax.set_xlabel("Predictive Gini Importance")
    ax.set_ylabel("")
    sns.despine()
    st.pyplot(fig)

with view_tab2:
    st.subheader("Validated Intervention Candidates")
    
    # ---------------- FILTERS ROW ---------------- #
    f_col1, f_col2 = st.columns(2)
    all_depts = sorted(validate_raw['Department'].unique().tolist())
    risk_categories = ['High Risk', 'Medium Risk', 'Low Risk']
    
    with f_col1:
        filter_dept = st.multiselect("Filter by Department", all_depts, default=all_depts)
    with f_col2:
        filter_risk = st.multiselect("Filter by Risk Level", risk_categories, default=risk_categories)
    
    # Build output frame
    roster = pd.DataFrame({
        'EmployeeNumber': validate_raw['EmployeeNumber'],
        'Risk_Probability': (probs * 100).round(1),
        'Risk_Level': pd.cut(probs, bins=[0, 0.35, 0.6, 1.0], labels=['Low Risk', 'Medium Risk', 'High Risk'], include_lowest=True),
        'Department': validate_raw.get('Department', 'N/A'),
        'JobRole': validate_raw.get('JobRole', 'N/A'),
        'YearsAtCompany': validate_raw.get('YearsAtCompany', 0)
    })
    
    # Apply filters
    filtered_roster = roster[
        (roster['Department'].isin(filter_dept)) & 
        (roster['Risk_Level'].isin(filter_risk))
    ].copy()
    
    # Sort so Highest Risk is visible first
    filtered_roster = filtered_roster.sort_values('Risk_Probability', ascending=False)
    
    # Styling logic
    def color_risk(val):
        if val == 'High Risk':
            return 'background-color: #FF4B4B; color: white; font-weight: bold'
        elif val == 'Medium Risk':
            return 'background-color: #FFA500; color: white; font-weight: bold'
        return 'background-color: #28A745; color: white;'

    styled_roster = filtered_roster.style.map(color_risk, subset=['Risk_Level'])\
                    .format({"Risk_Probability": "{:.1f}%"})
    
    st.dataframe(styled_roster, use_container_width=True, height=400)
    
    # Excel Export Workflow
    def generate_excel(df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Retention Analysis')
        return output.getvalue()
        
    processed_data = generate_excel(filtered_roster)
    st.download_button(
        label="📥 Download Filtered Retention Report (Excel)",
        data=processed_data,
        file_name='hr_retention_report.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )

with view_tab3:
    st.subheader("Ad-Hoc Prediction Interface")
    st.markdown("Specify operational and static signals to generate instantaneous intervention indicators.")
    
    with st.form("single_input_form", border=True):
        inp_c1, inp_c2 = st.columns(2)
        
        with inp_c1:
            in_age = st.number_input("Age", min_value=18, max_value=70, value=35)
            in_dept = st.selectbox("Department", categories['Department'])
            in_role = st.selectbox("Job Role", categories['JobRole'])
            in_income = st.number_input("Monthly Income ($)", min_value=500, max_value=50000, value=5000)
        
        with inp_c2:
            in_years = st.number_input("Years At Company", min_value=0, max_value=40, value=4)
            in_ot = st.selectbox("OverTime Status", ["No", "Yes"])
            in_sat = st.slider("Job Satisfaction (1=Low, 4=High)", 1, 4, 3)
            in_wlb = st.slider("Work-Life Balance (1=Low, 4=High)", 1, 4, 3)
            
        submit_btn = st.form_submit_button("🚀 Predict Flight Risk")

    if submit_btn:
        # 1. Derive candidate matrix from baseline row
        candidate = baseline.copy()
        
        # 2. Inject User Overrides
        candidate['Age'] = in_age
        candidate['Department'] = in_dept
        candidate['JobRole'] = in_role
        candidate['MonthlyIncome'] = in_income
        candidate['YearsAtCompany'] = in_years
        candidate['OverTime'] = in_ot
        candidate['JobSatisfaction'] = in_sat
        candidate['WorkLifeBalance'] = in_wlb
        
        # 3. Engineer Features and Preprocess
        candidate_eng = engineer_features(candidate)
        candidate_clean = candidate_eng.drop(columns=['Attrition', 'EmployeeNumber'], errors='ignore')
        candidate_p = processor.transform(candidate_clean)
        
        # 4. Predict
        p_prob = model.predict_proba(candidate_p)[0, 1]
        p_pct = p_prob * 100
        
        st.write("---")
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            if p_pct > 60:
                st.error("🚨 ALERT: HIGH RISK")
            elif p_pct >= 35:
                st.warning("⚠️ WARNING: MEDIUM RISK")
            else:
                st.success("✅ SECURE: LOW RISK")
                
        with res_col2:
            st.metric("Computed Probability Score", f"{p_pct:.1f}%", 
                      help="Aggressive threshold yields recommendations at >35%.")
            st.progress(int(p_pct))


st.sidebar.image("https://img.icons8.com/external-flat-gradient-icons-maxicons/512/external-hr-human-resources-flat-gradient-flat-gradient-icons-maxicons.png", width=100)
st.sidebar.title("System Controller")
st.sidebar.markdown("---")
st.sidebar.markdown("**Model Configuration**")
st.sidebar.text("Architecture: Soft-Voting Ensemble")
st.sidebar.text("Threshold Calibration: 0.35")
st.sidebar.markdown("---")
st.sidebar.markdown("**Balanced Precision**")
st.sidebar.info("Minority dataset imbalance resolved via SMOTE synthesizing real-time before scoring pass.")
