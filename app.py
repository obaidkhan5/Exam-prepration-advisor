import streamlit as st
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    layout="wide"
)

# Load Model
model = joblib.load("student_performance_predictor.pkl")

# Header
st.title("Student Performance Predictor")

st.write(
    "This application predicts a student's expected examination score using a Machine Learning model trained on academic performance data."
)

st.divider()

# Sidebar
st.sidebar.header("Student Information")

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

hours_studied = st.sidebar.slider(
    "Hours Studied Per Week",
    0,
    40,
    15
)

attendance = st.sidebar.slider(
    "Attendance Percentage",
    0,
    100,
    80
)

tutoring = st.sidebar.selectbox(
    "Tutoring Support",
    ["Yes", "No"]
)

region = st.sidebar.selectbox(
    "Region",
    ["Urban", "Suburban", "Rural"]
)

parent_education = st.sidebar.selectbox(
    "Parent Education",
    ["High School", "Bachelor", "Master", "PhD"]
)

# Encoding

gender_encoded = 1 if gender == "Male" else 0
tutoring_encoded = 1 if tutoring == "Yes" else 0

region_map = {
    "Rural": 0,
    "Suburban": 1,
    "Urban": 2
}

parent_map = {
    "High School": 0,
    "Bachelor": 1,
    "Master": 2,
    "PhD": 3
}

region_encoded = region_map[region]
parent_encoded = parent_map[parent_education]

# Dashboard Summary

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Attendance", f"{attendance}%")

with col2:
    st.metric("Study Hours", f"{hours_studied} Hours")

with col3:
    st.metric("Tutoring", tutoring)

st.divider()

# Prediction Button

if st.button("Predict Performance", use_container_width=True):

    input_df = pd.DataFrame(
        [[
            gender_encoded,
            hours_studied,
            tutoring_encoded,
            region_encoded,
            attendance,
            parent_encoded
        ]],
        columns=[
            "Gender",
            "HoursStudied/Week",
            "Tutoring",
            "Region",
            "Attendance(%)",
            "Parent Education"
        ]
    )

    prediction = model.predict(input_df)

    score = prediction[0]

    # Restrict score between 0 and 100
    score = max(0, min(score, 100))
    score = round(score, 2)

    # Performance Category

    if score >= 85:
        category = "Excellent"
        risk = "Low"

        recommendation = """
Maintain current study habits.
Continue regular revision.
Practice advanced questions.
"""

    elif score >= 70:
        category = "Good"
        risk = "Moderate"

        recommendation = """
Continue consistent preparation.
Focus on weaker topics.
Practice previous papers.
"""

    elif score >= 50:
        category = "Average"
        risk = "High"

        recommendation = """
Increase study hours.
Improve attendance.
Seek additional academic support.
"""

    else:
        category = "At Risk"
        risk = "Critical"

        recommendation = """
Immediate academic intervention is recommended.
Consult an academic advisor.
Follow a structured study plan.
"""

    st.subheader("Prediction Results")

    r1, r2, r3 = st.columns(3)

    with r1:
        st.metric(
            "Predicted Score",
            f"{score}/100"
        )

    with r2:
        st.metric(
            "Performance Category",
            category
        )

    with r3:
        st.metric(
            "Risk Level",
            risk
        )

    st.progress(score / 100)

    st.info(
        f"The model estimates that the student may achieve approximately {score}% in the examination."
    )

    st.subheader("Student Profile Summary")

    summary = pd.DataFrame({
        "Attribute": [
            "Gender",
            "Study Hours",
            "Attendance",
            "Tutoring",
            "Region",
            "Parent Education"
        ],
        "Value": [
            gender,
            hours_studied,
            f"{attendance}%",
            tutoring,
            region,
            parent_education
        ]
    })

    st.dataframe(
        summary,
        use_container_width=True
    )

    st.subheader("Academic Recommendations")

    st.write(recommendation)

st.divider()

st.caption(
    "Student Performance Predictor | Machine Learning Project | Linear Regression"
)