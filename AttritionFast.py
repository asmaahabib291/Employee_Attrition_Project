from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# تحميل الموديل
model = joblib.load("attrition_pipeline.pkl")

# تعريف FastAPI app
app = FastAPI(title="Attrition Prediction API")

# تعريف بيانات الإدخال
class EmployeeData(BaseModel):
    Age: int
    DistanceFromHome: float
    MonthlyIncome: float
    Gender: int
    OverTime: int
    BusinessTravel: str
    Department: str
    EducationField: str
    JobRole: str
    MaritalStatus: str
    StockOptionLevel: int
    RelationshipSatisfaction: int
    YearsAtCompany: int
    MonthlyRate: float
    YearsSinceLastPromotion: int
    TrainingTimesLastYear: int
    JobSatisfaction: int
    JobInvolvement: int
    PerformanceRating: int

@app.post("/predict")
def predict_attrition(data: EmployeeData):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }
