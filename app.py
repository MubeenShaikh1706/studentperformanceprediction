

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Find project files
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "student-mat.csv"
MODEL_PATH = BASE_DIR / "model" / "model.pkl"

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Performance Prediction System")
st.write(
    "Predict an estimated final grade using study time, "
    "absences and previous academic grades."
)

# Check that the model has been trained
if not MODEL_PATH.exists():
    st.error("Model not found. Please run: python train_model.py")
    st.stop()

model = joblib.load(MODEL_PATH)

st.subheader("Enter Student Details")

with st.form("prediction_form"):
    studytime = st.selectbox(
        "Weekly study-time category",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "1 - Less than 2 hours",
            2: "2 - 2 to 5 hours",
            3: "3 - 5 to 10 hours",
            4: "4 - More than 10 hours"
        }[x]
    )

    absences = st.number_input(
        "Number of absences",
        min_value=0,
        max_value=100,
        value=2,
        step=1
    )

    g1 = st.slider(
        "First-period grade (G1)",
        min_value=0,
        max_value=20,
        value=12
    )

    g2 = st.slider(
        "Second-period grade (G2)",
        min_value=0,
        max_value=20,
        value=13
    )

    submitted = st.form_submit_button("Predict Final Grade")

if submitted:
    # Input order must match the model's training features
    input_data = pd.DataFrame(
        [[studytime, absences, g1, g2]],
        columns=["studytime", "absences", "G1", "G2"]
    )

    prediction = float(model.predict(input_data)[0])

    # Keep the displayed estimate within the dataset's grade scale
    prediction = max(0.0, min(20.0, prediction))

    st.subheader("Prediction Result")
    st.metric("Estimated Final Grade", f"{prediction:.1f} / 20")

    if prediction < 10:
        st.warning(
            "Estimated category: Needs support. "
            "Consider reviewing topics and seeking academic help."
        )
    elif prediction < 15:
        st.info(
            "Estimated category: Satisfactory. "
            "Consistent revision may help improve performance."
        )
    else:
        st.success(
            "Estimated category: Strong performance. "
            "Keep practising and revising."
        )

    st.caption(
        "This is a model-based estimate, not a guaranteed result "
        "or an official academic assessment."
    )

# Show a simple dataset dashboard
st.divider()
st.subheader("Dataset Overview")

if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH, sep=";")

    col1, col2 = st.columns(2)
    col1.metric("Student Records", len(df))
    col2.metric("Average Final Grade", f"{df['G3'].mean():.1f} / 20")

    st.write("Final grade distribution")
    st.bar_chart(df["G3"].value_counts().sort_index())

    with st.expander("View sample student records"):
        st.dataframe(df.head(10), use_container_width=True)
else:
    st.info("Place student-mat.csv in the data folder to view the dashboard.")
