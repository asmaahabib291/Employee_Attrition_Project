import streamlit as st
import pandas as pd
import joblib

model = joblib.load("attrition_pipeline.pkl")

st.title("🔍 Employee Attrition Prediction")

st.markdown("Enter the employee details below.")




age = st.slider("Age", 18, 60, 30)

distance = st.slider("Distance From Home (km)", 1, 30, 5)

monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=50000, value=5000)

gender = st.selectbox("Gender", ["Male", "Female"])

overtime = st.selectbox("OverTime", ["Yes", "No"])

business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])

department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])

education_field = st.selectbox("Education Field", ["Life Sciences", "Other", "Medical", "Marketing", "Technical Degree", "Human Resources"])

job_role = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"])

marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

stock_option = st.selectbox("Stock Option Level", [0, 1, 2, 3])

relationship_satisfaction = st.slider("Relationship Satisfaction", 1, 4, 2)

years_at_company = st.slider("Years At Company", 0, 40, 3)

monthly_rate = st.number_input("Monthly Rate", min_value=1000, max_value=30000, value=10000)

years_since_promotion = st.slider("Years Since Last Promotion", 0, 15, 1)

training_times = st.slider("Training Times Last Year", 0, 10, 3)

job_satisfaction = st.slider("Job Satisfaction", 1, 4, 2)

job_involvement = st.slider("Job Involvement", 1, 4, 2)

performance_rating = st.slider("Performance Rating", 1, 4, 3)



input_data = pd.DataFrame({
    "Age": [age],
    "DistanceFromHome": [distance],
    "MonthlyIncome": [monthly_income],
    "Gender": [1 if gender == "Male" else 0],
    "OverTime": [1 if overtime == "Yes" else 0],
    "BusinessTravel": [business_travel],
    "Department": [department],
    "EducationField": [education_field],
    "JobRole": [job_role],
    "MaritalStatus": [marital_status],
    "StockOptionLevel": [stock_option],
    "RelationshipSatisfaction": [relationship_satisfaction],
    "YearsAtCompany": [years_at_company],
    "MonthlyRate": [monthly_rate],
    "YearsSinceLastPromotion": [years_since_promotion],
    "TrainingTimesLastYear": [training_times],
    "JobSatisfaction": [job_satisfaction],
    "JobInvolvement": [job_involvement],
    "PerformanceRating": [performance_rating]
})


if st.button("🔮 Predict Attrition"):
    prediction = model.predict(input_data)
    result = "Employee is more likely to leave the company❌" if prediction[0] == 1 else "Employee is more likely to stay in the company✅"
    st.subheader(result)
