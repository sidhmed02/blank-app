import streamlit as st
import joblib
import pandas as pd

# تحميل النموذج والسكالر
model = joblib.load("student_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Student Performance")

st.title("🎓 التنبؤ بالأداء الدراسي")
st.write("أدخل معلومات الطالب للحصول على النتيجة المتوقعة")

# ====== إدخال البيانات ======
age = st.number_input("العمر", 5, 25, 16)
study_hours = st.number_input("ساعات الدراسة يومياً", 0.0, 10.0, 2.0)
attendance = st.number_input("نسبة الحضور (%)", 0.0, 100.0, 85.0)

math = st.number_input("علامة الرياضيات", 0.0, 100.0, 70.0)
science = st.number_input("علامة العلوم", 0.0, 100.0, 70.0)
english = st.number_input("علامة الإنجليزية", 0.0, 100.0, 70.0)

gender = st.selectbox("الجنس", ["Male", "Female"])
school = st.selectbox("نوع المدرسة", ["Public", "Private"])
parent = st.selectbox("تعليم الوالدين", ["High School", "College", "University"])
internet = st.selectbox("إنترنت", ["Yes", "No"])
travel = st.selectbox("وقت التنقل", ["Short", "Long"])
activities = st.selectbox("نشاطات إضافية", ["Yes", "No"])
study_method = st.selectbox("طريقة الدراسة", ["Group", "Solo"])

# ====== تحويل البيانات ======
data = {
    "age": age,
    "study_hours": study_hours,
    "attendance_percentage": attendance,
    "math_score": math,
    "science_score": science,
    "english_score": english,
    "gender_Male": 1 if gender == "Male" else 0,
    "school_type_Public": 1 if school == "Public" else 0,
    "parent_education_College": 1 if parent == "College" else 0,
    "parent_education_High School": 1 if parent == "High School" else 0,
    "internet_access_Yes": 1 if internet == "Yes" else 0,
    "travel_time_Short": 1 if travel == "Short" else 0,
    "extra_activities_Yes": 1 if activities == "Yes" else 0,
    "study_method_Group": 1 if study_method == "Group" else 0
}

df = pd.DataFrame([data])
df_scaled = scaler.transform(df)

# ====== التنبؤ ======
if st.button("🔮 توقع النتيجة"):
    result = model.predict(df_scaled)[0]
    st.success(f"🎯 النتيجة المتوقعة: {result:.2f}")
