import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np

### load Saved model
model=joblib.load("fraud_model.pkl")
scaler=joblib.load("scaler.pkl")
feature_columns=joblib.load("feature_columns.pkl")

#### page seperation
page = st.sidebar.radio(
    "Navigation",
    ["🤖 Model Details","🔮 Fraud Prediction","📊 Insights"]  
)

###page3
if page == "🔮 Fraud Prediction":


### Titles

    st.title("🛡️ Fraud Score Risk Prediction ")



## 

    st.subheader("💳 Transaction Details")

    col1,col2=st.columns(2)
    with col1:
        amount = st.number_input("Transaction Amount", min_value=0.0)
    
    transaction_hour = st.slider("Transaction Hour (0-23)", 0, 23, 12)

    with col2:
        is_new_device = st.selectbox("Is New Device?", ["Yes","No"])
        new_map={
            "Yes" : 1,
            "No" : 0 }

        
    
    col3,col4=st.columns(2)

    with col3:
        is_foreign_transaction = st.selectbox("Is Foreign Transaction?", 
                                              ["Yes","No"])
        foreign_map={
            "Yes" : 1,
            "No" : 0 }
        day_of_week = st.selectbox(
        "Day of Week",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        )
        day_mapping = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6
         }

    with col4:
        distance_from_home_km=st.number_input("Distance From Home" ,min_value=0.0)
        time_since_last_txn_hrs=st.number_input("Time Since last Txn (hrs)",min_value=0)

    login_attempts_today=st.number_input("Number of login Attempt Today",min_value=0)
    
    st.subheader("👤Customer profile")

    col5,col6=st.columns(2)

    with col5:
        gender = st.selectbox("Gender", ["Male", "Female"])
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900,step=1)
        account_tenure_days=st.number_input("Account Tenure Days",min_value=0,step=1)
        daily_transaction_count=st.number_input("Daily Txn Count",min_value=0,step=1)
    
    with col6:
        customer_age = st.number_input("Customer Age", min_value=18, max_value=100,step=1)
        avg_monthly_balance = st.number_input("Average Monthly Balance", min_value=0,step=1)
        num_previous_disputes=st.number_input("Number of previous disputes",min_value=0,step=1)
        failed_txn_last_7days=st.number_input("Failed txns in Last 7 days",min_value=0,step=1)

    num_cards_linked = st.number_input("Number of cards Linked",min_value=0,step=1)

    st.subheader("🌍 Category Info")

    col7,col8=st.columns(2)

    with col7:
        transaction_type = st.selectbox(
        "Transaction Type",
        ["Card", "IMPS", "NEFT", "UPI", "ATM"]
         )
        city = st.selectbox(
        "City",
        ["Bangalore", "Chennai", "Delhi", "Hyderabad",
         "Kolkata", "Mumbai", "Pune", "Ahmedabad"]
    )

    with col8:
        merchant_category = st.selectbox(
        "Merchant Category",
        ["electronics", "entertainment", "grocery", "healthcare", "travel", "unknown", "dining"]
         )
        is_high_risk_merchant=st.selectbox("Is High Risk Merchant",["Yes","No"])
        is_high={
            "Yes" : 1,
            "No" : 0
        }
    
    



### getting automated
    is_odd_hour = 1 if transaction_hour < 6 or transaction_hour > 22 else 0
    day_num = day_mapping[day_of_week]

    is_weekend = 1 if day_num in [5, 6] else 0
    balance_to_amount_ratio =(avg_monthly_balance / amount if amount > 0 else 0)

### predict button

    if st.button("Predict Fraud Risk"):

        input_dict = {
            "amount": amount,
            "transaction_hour": transaction_hour,
            "day_of_week": day_mapping[day_of_week],
            "customer_age": customer_age,
            "account_tenure_days": account_tenure_days,
            "avg_monthly_balance": avg_monthly_balance,
            "credit_score": credit_score,
            "num_previous_disputes": num_previous_disputes,
            "is_new_device": new_map[is_new_device],
            "is_foreign_transaction": foreign_map[is_foreign_transaction],
            "distance_from_home_km": distance_from_home_km,
            "time_since_last_txn_hrs": time_since_last_txn_hrs,
            "daily_transaction_count": daily_transaction_count,
            "failed_txn_last_7days": failed_txn_last_7days,
            "is_high_risk_merchant": is_high[is_high_risk_merchant],
            "num_cards_linked": num_cards_linked,
            "login_attempts_today": login_attempts_today,
            "is_weekend": is_weekend,
            "is_odd_hour": is_odd_hour,
            "balance_to_amount_ratio": balance_to_amount_ratio,
            "transaction_type": transaction_type,
            "merchant_category": merchant_category,
            "gender": gender,
            "city": city
        }

        input_df = pd.DataFrame([input_dict])

    # Encode
        input_encoded = pd.get_dummies(input_df)

    # Align columns
        input_encoded = input_encoded.reindex(
            columns=feature_columns,
            fill_value=0
        )

    # Scale
        input_scaled = scaler.transform(input_encoded)

    # Predict
        prediction = model.predict(input_scaled)[0]
        st.metric("Fraud Risk Score", f"{prediction*100:.2f}%")
        if prediction >= 0.50:
            st.error("⚠  HIGH RISK ")
            st.write("Confidence         : Strong suspicious behaviour detected")
            st.write("Recommended Action : Block transaction or require OTP / manual review")
            st.write("Possible Reasons   : New device, foreign location, high transaction amount")
        
        elif prediction >= 0.30:
            st.warning("⚡ MEDIUM RISK")
            st.write("Risk Level           : MEDIUM")
            st.write("Confidence           : Some unusual patterns detected")
            st.write("Recommended Action   : Monitor transaction or request additional verification")
    
        else:
            st.success("✅ LOW RISK")
            st.write("Confidence           : Transaction behaviour appears normal")
            st.write("Recommended Action   : Approve transaction")
            
### -------- end of page 1 ------ 

if page == "🤖 Model Details":
    
    st.title("🤖 Fraud Score Prediction System")
    st.markdown("---")
    st.header("Model Architecture & Technical Details")
    st.markdown("---")
    st.markdown("""
    ### 🧠 Model Type
    - **Algorithm:** Random Forest Regressor
    - **Learning Approach:** Supervised Machine Learning
    - **Objective:** Predict continuous fraud risk score (0–1)

    Random Forest was selected because:
    - Handles nonlinear relationships
    - Robust to outliers
    - Works well with mixed feature types
    - Provides feature importance for interpretability
    """)

    
    
    
    st.markdown("---")
    st.subheader("Machine Learning–Driven Transaction Risk Assessment using Random Forest")
    st.write("This dashboard presents an end-to-end Fraud Risk Score Prediction System developed using machine learning techniques. The objective of the model      is to evaluate financial transaction patterns and generate a quantitative fraud risk score based on behavioral and transactional features.")
    st.write("The system was built using a Random Forest Regressor after performing comprehensive data cleaning, preprocessing, feature scaling, exploratory        data analysis (EDA), and hyperparameter tuning. Feature importance analysis was conducted to understand the key drivers influencing fraud risk.")
    st.write("The model processes transaction-related inputs and produces a risk score that can assist financial institutions and businesses in identifying         potentially suspicious activities and supporting informed decision-making.")

    df=pd.read_csv("C://Users//Muhammed Sinan m//Documents//cs//python//Machine Learning//Supervised ml//fraud_detection_dataset.csv")
    st.markdown("---")
    st.subheader("Dataset Information")

    if st.checkbox('Show DataFrame'):
     chart_data= df
     chart_data

    df.info()
    
#Create a summary Dataframe
    if st.checkbox("Show DataFrame Information"):
        info_data = {
        "Columns" : df.columns ,
        "Non-Null Counts" : df.notnull().sum().values,
        "Dtype" : df.dtypes.values
        }
        info_df = pd.DataFrame(info_data)

        st.dataframe(info_df)

# Add memory usage
    st.write(f"**Memory usage:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

    ###-------

    st.markdown("---")
    st.subheader("📊 Training Strategy")

    st.markdown("""
    - Train/Test Split: 80% / 20%
    - Feature Scaling: StandardScaler
    - Categorical Encoding: One-Hot Encoding
    - Data Leakage Prevention:
      - Scaler fitted only on training data
      - Test data transformed using fitted scaler
     """)


    st.markdown("---")
    st.subheader("⚙ Hyperparameter Optimization")

    st.markdown("""
    Hyperparameter tuning was performed using **GridSearchCV**.

    Optimized parameters included:
    - n_estimators
    - max_depth
    - min_samples_leaf
    

    This improved generalization performance and reduced overfitting.
    """)



    
    st.markdown("---")
    st.subheader("📈 Model Performance")

    st.markdown(f"""
    - **Train R² Score:** 0.9040  
    - **Test R² Score:** 0.8733  
    - **Cross-Validation R²:** 0.8779  
    - **RMSE:** 0.0377  
    - **MAE:** 0.0293  

    ### Interpretation

    - The model explains approximately **87% of fraud risk variance**, indicating strong predictive capability.
    - The small gap between train and test R² suggests minimal overfitting.
    - RMSE and MAE values are very low (≈3–4%), meaning predictions are highly accurate within the 0–1 risk scale.
    - Cross-validation confirms stable performance across different data splits.
    """)


    st.markdown("---")
    st.subheader("🚀 Deployment Architecture")

    st.markdown("""
    - Model serialized using joblib
    - Scaler and feature columns saved separately
    - Streamlit used for real-time inference
    - Input features aligned using reindex()
    - Production-safe pipeline prevents shape mismatch

    This ensures consistent model behavior between training and deployment.
    """)



####  --------------         End of page 2 --------------------------

if page == "📊 Insights":
    
    st.header("Data Analysis Dashboard")
 ##Plot1
   
    df=pd.read_csv("C://Users//Muhammed Sinan m//Documents//cs//python//Machine Learning//Supervised ml//fraud_detection_dataset.csv")
    
## plot 2 
    st.markdown("---")
    st.subheader("⏰ Fraud Risk by Transaction Hour")

    hour_avg = df.groupby(
        ["transaction_hour", "is_foreign_transaction"]
    )["fraud_risk_score"].mean().reset_index()

    fig = px.line(
        hour_avg,
        x="transaction_hour",
        y="fraud_risk_score",
        color="is_foreign_transaction",
        markers=True,
        title="Fraud Risk Trend by Hour (Foreign vs Local)"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Key Insights:**
    - Fraud risk significantly increases during late-night hours (12 AM – 4 AM).
    - Foreign transactions consistently show higher fraud risk across all time periods.
    - The combination of unusual transaction time and foreign origin is a strong fraud indicator.
    - Fraud activity declines during normal business hours.
    """)

### plot 3
    st.markdown("---")
    st.subheader("🌍 Average Fraud Risk by Distance Range")

    df["distance_bin"] = pd.cut(
        df["distance_from_home_km"],
        bins=[0, 10, 30, 60, 100, 200],
        labels=["0-10km", "10-30km", "30-60km", "60-100km", "100km+"]
   )

    distance_avg = df.groupby("distance_bin")["fraud_risk_score"].mean().reset_index()

    fig = px.bar(
        distance_avg,
        x="distance_bin",
        y="fraud_risk_score",
        color="distance_bin",
        title="Average Fraud Risk by Distance Range"
   )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Insight:**
    - Fraud risk increases as geographic distance increases.
    - Transactions occurring beyond 60km show elevated anomaly patterns.
    - Geographic displacement is a strong fraud indicator.
    """)
    #### plot4
    st.markdown("---")
    st.subheader("🔐 Average Fraud Risk by Failed Attempts")

    failed_avg = df.groupby("failed_txn_last_7days")["fraud_risk_score"].mean().reset_index()

    fig = px.bar(
        failed_avg,
        x="failed_txn_last_7days",
        y="fraud_risk_score",
        color="failed_txn_last_7days",
        title="Average Fraud Risk vs Failed Attempts"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Insight:**
    - Fraud risk increases with repeated failed transaction attempts.
    - Accounts with multiple failed attempts show elevated anomaly signals.
    - Failed attempts can indicate credential testing or account takeover activity.
    """)


    #### plot 5
    st.markdown("---")
    st.subheader("💳 Fraud Risk by Transaction Type")

    fig = px.box(
    df,
    x="transaction_type",
    y="fraud_risk_score",
    color="is_high_risk_merchant",
    title="Fraud Risk Distribution by Transaction Type"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Key Insights:**
    - Card and UPI transactions show higher fraud risk variability.
    - High-risk merchants significantly increase fraud scores across all types.
    - Fraudsters prefer faster digital payment channels.
    """)

    #### plot 6
    st.markdown("---")
    st.subheader("🤖 Top Features Influencing Fraud Prediction")
    

    importance = model.feature_importances_
    feat_imp = pd.Series(importance, index=feature_columns)

    feat_imp = feat_imp.sort_values(ascending=False)
    top10=feat_imp.head(10)
    
    fig = px.bar(
        x=top10.values,
        y=top10.index,
        orientation="h",
        title="Top 10 Features Driving Fraud Risk"
    )

    fig.update_layout(yaxis={'categoryorder':'total ascending'})

    st.plotly_chart(fig)

    

    st.markdown("""
    **Key Insights:**
    - Device change, foreign transactions, and distance from home are strong fraud indicators.
    - Behavioral features contribute more to fraud detection than static demographic features.
    - The model effectively captures anomaly-based fraud patterns.
    """)


    #### Final

    st.markdown("---")
    st.subheader("📌 Final Analysis & Business Interpretation")

    st.markdown("""
    ### 🔍 Key Behavioral Fraud Indicators

    - Fraud risk increases significantly during late-night hours (12 AM – 4 AM).
    - Foreign transactions consistently show elevated fraud risk.
    - Large geographic distance from home is a strong anomaly signal.
    - Multiple failed transaction attempts indicate potential credential testing.
    - High-risk merchants amplify fraud probability across payment types.

    ### 🤖 Model Intelligence

    - Behavioral features (device change, location shift, transaction timing) contribute more than static demographic attributes.
    - The Random Forest model effectively captures nonlinear fraud patterns.
    - Feature importance analysis confirms anomaly-based detection approach.

    ### 📊 Risk Segmentation

    - Low Risk: Normal transaction behavior.
    - Medium Risk: Minor anomalies detected.
    - High Risk: Strong multi-feature anomaly signals.

    ### 💼 Business Impact

    - Enables early detection of suspicious activity.
    - Reduces financial losses from fraudulent transactions.
    - Supports real-time risk scoring in fintech applications.
    - Can be integrated into payment gateways or banking systems.

    ---

    **Conclusion:**  
    This system demonstrates a behavior-driven fraud detection framework using machine learning, capable of identifying high-risk transactions through pattern-     based anomaly detection.
    """)