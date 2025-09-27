import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ----------------------------
# Load and preprocess data
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("loan.csv")
    df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
    df['LoanAmount_log'] = np.log(df['LoanAmount'].replace(0, 1))
    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['TotalIncome_log'] = np.log(df['TotalIncome'].replace(0, 1))

    for c in ['Gender', 'Married', 'Self_Employed', 'Dependents', 'Loan_Amount_Term', 'Credit_History']:
        df[c].fillna(df[c].mode()[0], inplace=True)

    return df

df = load_data()

# ----------------------------
# Sidebar navigation
# ----------------------------
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to:", ["📊 Data Analysis", "🤖 Prediction"])

# ----------------------------
# ----------------------------
# 1. Data Analysis Page
# ----------------------------
if menu == "📊 Data Analysis":
    st.title("Loan Data Analysis")

    st.subheader("Dataset Preview")
    st.write(df.head())

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    # Loan Amount Histogram
    st.subheader("Loan Amount Distribution")
    fig, ax = plt.subplots()
    df['LoanAmount'].hist(bins=20, ax=ax, color='skyblue', edgecolor='black')
    ax.set_xlabel("Loan Amount")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # Total Income Histogram
    st.subheader("Total Income (log) Distribution")
    fig, ax = plt.subplots()
    df['TotalIncome_log'].hist(bins=20, ax=ax, color='green', edgecolor='black')
    ax.set_xlabel("Log Total Income")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # Countplots for categorical columns
    st.subheader("Categorical Distributions")
    cat_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']
    for col in cat_cols:
        st.write(f"Distribution of {col}")
        fig, ax = plt.subplots()
        sns.countplot(x=col, data=df, palette='Set2', ax=ax)
        st.pyplot(fig)

    # 🔥 Correlation Heatmap
    st.subheader("Correlation Heatmap (Numerical Features)")
    num_cols = ['ApplicantIncome','CoapplicantIncome','LoanAmount','LoanAmount_log','TotalIncome','TotalIncome_log']
    fig, ax = plt.subplots(figsize=(8,5))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # 🔥 Boxplots for Outlier Detection
    st.subheader("Boxplots (Outlier Detection)")
    for col in ['ApplicantIncome','LoanAmount','TotalIncome']:
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], ax=ax, color="orange")
        st.write(f"Boxplot of {col}")
        st.pyplot(fig)

    # 🔥 Pairplot (small sample for performance)
    st.subheader("Pairplot (Sample of Data)")
    sample_df = df[['ApplicantIncome','CoapplicantIncome','LoanAmount','TotalIncome_log','Loan_Status']].dropna().sample(100, random_state=1)
    fig = sns.pairplot(sample_df, hue="Loan_Status", diag_kind="kde", palette="husl")
    st.pyplot(fig)
# ----------------------------
# 2. Prediction Page
# ----------------------------
if menu == "🤖 Prediction":
    st.title("Loan Approval Prediction")

    # Features & target
    features = ['Gender','Married','Dependents','Education','Self_Employed',
                'LoanAmount','LoanAmount_log','Loan_Amount_Term','Credit_History','TotalIncome_log']
    TARGET = 'Loan_Status'

    X = df[features]
    y = df[TARGET].map({'Y':1,'N':0})

    X = pd.get_dummies(X, drop_first=True)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Train
    rf = RandomForestClassifier(n_estimators=100, random_state=0)
    rf.fit(X_train_s, y_train)

    st.write("Model Test Accuracy:", f"{rf.score(X_test_s, y_test):.2f}")

    # Sidebar Inputs
    st.sidebar.subheader("Enter Applicant Details")

    Gender = st.sidebar.selectbox("Gender", ("Male","Female"))
    Married = st.sidebar.selectbox("Married", ("Yes","No"))
    Dependents = st.sidebar.selectbox("Dependents", ("0","1","2","3"))
    Education = st.sidebar.selectbox("Education", ("Graduate","Not Graduate"))
    Self_Employed = st.sidebar.selectbox("Self Employed", ("Yes","No"))
    ApplicantIncome = st.sidebar.number_input("Applicant Income", min_value=0, value=2500)
    CoapplicantIncome = st.sidebar.number_input("Coapplicant Income", min_value=0, value=0)
    LoanAmount = st.sidebar.number_input("Loan Amount", min_value=1, value=100)
    Loan_Amount_Term = st.sidebar.selectbox("Loan Term (days)", (360, 180, 240, 120, 60))
    Credit_History = st.sidebar.selectbox("Credit History", (1.0, 0.0))

    input_dict = {
      "Gender": Gender, "Married": Married, "Dependents": Dependents,
      "Education": Education, "Self_Employed": Self_Employed,
      "LoanAmount": LoanAmount, "LoanAmount_log": np.log(max(LoanAmount,1)),
      "Loan_Amount_Term": Loan_Amount_Term, "Credit_History": Credit_History,
      "TotalIncome_log": np.log(max(ApplicantIncome+CoapplicantIncome,1))
    }

    input_df = pd.DataFrame(input_dict, index=[0])
    combined = pd.concat([input_df, df[features]], axis=0)
    combined = combined.replace('3+', '3')
    combined = pd.get_dummies(combined, drop_first=True)
    input_proc = combined.iloc[0:1, :].reindex(columns=X.columns, fill_value=0)

    input_scaled = scaler.transform(input_proc)
    pred = rf.predict(input_scaled)[0]
    pred_proba = rf.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction")
    st.write("✅ Loan Approved" if pred == 1 else "❌ Loan Rejected")
    st.write("Approval Probability:", f"{pred_proba*100:.2f}%")
