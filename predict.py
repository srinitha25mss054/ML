import joblib
import pandas as pd
model=joblib.load("Models/student_models.pkl")
new_student=pd.DataFrame({"study_hours":[7],"attendance":[85],"previous_marks":[70]})
prediction=model.predict(new_student)
if prediction[0]==1:
    print("student result:pass")
else:
    print("student result:fail")