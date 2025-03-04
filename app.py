import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load trained AI model
model = pickle.load(open("ai_student_model.pkl", "rb"))

# Define label encoders for categorical fields
categorical_columns = ["Enrollment_Type", "Program_Type", "Career_Interest"]
label_encoders = {}

# Placeholder encoders (to be replaced with trained ones)
for col in categorical_columns:
    label_encoders[col] = LabelEncoder()

st.title("AI-Powered Student Success Predictor")

st.write("Upload student data to predict dropout risk and receive personalized interventions.")

# Upload file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    # Read uploaded file
    df_input = pd.read_csv(uploaded_file)

    # Encode categorical values
    for col in categorical_columns:
        if col in df_input.columns:
            df_input[col] = label_encoders[col].fit_transform(df_input[col])

    # Predict dropout risk
    predictions = model.predict(df_input)

    # Show results
    df_input["Predicted_Dropout_Risk"] = predictions
    st.write("Prediction Results:")
    st.dataframe(df_input)

    # Provide personalized interventions
    for index, row in df_input.iterrows():
        st.subheader(f"Student {index + 1}")
        if row["Predicted_Dropout_Risk"] == 1:
            st.warning("⚠️ High Dropout Risk! Suggested Intervention: Personalized mentoring, financial aid assistance.")
        else:
            st.success("✅ Low Dropout Risk! Keep up the good engagement.")

st.write("Developed as an AI prototype for student retention insights.")
