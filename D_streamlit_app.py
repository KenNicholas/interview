import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Model Inference Portal", layout="wide")
st.title(" Model Prediction Dashboard")

@st.cache_resource
def load_model():
    return joblib.load("best_pipeline.joblib")

try:
    model = load_model()
    st.success("Model loaded successfully!")
    
    uploaded_file = st.file_uploader("Upload CSV Data for Inference", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview Input Data", df.head())
        
        if st.button("Run Predictions"):
            preds = model.predict(df)
            df['Prediction'] = preds
            st.write("### Prediction Results", df.head())
            
            # Download button
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions CSV", data=csv_data, file_name="predictions.csv", mime="text/csv")
            
except Exception as e:
    st.error(f"Failed to load model or process data: {e}")