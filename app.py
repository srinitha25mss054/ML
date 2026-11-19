from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

class StudentData(BaseModel):
    study_hours: float
    attendance: float
    previous_marks: float

app = FastAPI()
model = joblib.load("models/student_models.pkl")

@app.get("/")
def home():
    return {"message": "Student Performance prediction api running"}

@app.post("/predict")
def predict(student: StudentData):
    try:
        input_data = pd.DataFrame([{
            "study_hours": student.study_hours, 
            "attendance": student.attendance, 
            "previous_marks": student.previous_marks
        }])
        prediction = model.predict(input_data)
        return {"predicted_result": str(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))