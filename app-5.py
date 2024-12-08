import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import setup as clf_setup, compare_models as clf_compare, pull as clf_pull, save_model as clf_save, load_model as clf_load
from pycaret.regression import setup as reg_setup, compare_models as reg_compare, pull as reg_pull, save_model as reg_save, load_model as reg_load
import os

# Initialize session state for dataset
if 'df' not in st.session_state:
    st.session_state['df'] = None

# Sidebar Navigation
with st.sidebar:
    st.image(
        'https://qtxasset.com/cdn-cgi/image/w=850,h=478,f=auto,fit=crop,g=0.5x0.5/https://qtxasset.com/quartz/qcloud4/media/image/fiercevideo/1554925532/googlecloud.jpg?VersionId=hJC0G.4VGlXbcc35EzyI9RhCJI.mslxN'
    )
    st.title("PLN AutoML")
    choice = st.radio("Navigation", ["Upload Dataset", "Profiling", "Modelling", "Download Model", "Predict"])
    st.info("This application helps you build and explore Machine Learning models.")

# Upload Section for Dataset
if choice == "Upload Dataset":
    st.title("Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload Your Dataset (CSV format only)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state['df'] = df  # Save dataset to session state
        st.success("Dataset uploaded successfully!")
        st.dataframe(st.session_state['df'])
    else:
        st.warning("Please upload a CSV file.")

# Profiling Section
if choice == "Profiling":
    st.title("Exploratory Data Analysis")
    if st.session_state['df'] is not None:
        df = st.session_state['df']
        profile = ProfileReport(df)
        st_profile_report(profile)
    else:
        st.error("Please upload a dataset first.")

# Modelling Section
if choice == "Modelling":
    if st.session_state['df'] is not None:
        df = st.session_state['df']
        st.title("Build Your Model")
        mdl_type = st.selectbox("Choose Modelling Type:", ["Classification", "Regression"])
        target_col = st.selectbox("Choose Target Column:", df.columns)
        if st.button("Run Modelling"):
            try:
                if mdl_type == "Classification":
                    clf_setup(df, target=target_col, session_id=123)
                    st.subheader("Setup Data Summary")
                    st.dataframe(clf_pull())

                    best_model = clf_compare()
                    st.subheader("Model Comparison")
                    st.dataframe(clf_pull())

                    st.success("Best classification model trained!")

                    # Save the best model without duplicating the file extension
                    clf_save(best_model, "best_classification_model.pkl")
                    st.success("Best classification model saved!")
                    st.write("Model saved at:", os.path.abspath("best_classification_model.pkl"))

                elif mdl_type == "Regression":
                    reg_setup(df, target=target_col, session_id=123)
                    st.subheader("Setup Data Summary")
                    st.dataframe(reg_pull())

                    best_model = reg_compare()
                    st.subheader("Model Comparison")
                    st.dataframe(reg_pull())

                    st.success("Best regression model trained!")
                    # Save the best model without duplicating the file extension
                    reg_save(best_model, "best_regression_model.pkl")
                    st.success("Best regression model saved!")
                    st.write("Model saved at:", os.path.abspath("best_regression_model.pkl"))

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.error("Please upload a dataset first.")

# Download Section for Model
if choice == "Download Model":
    if os.path.exists("best_classification_model.pkl") or os.path.exists("best_regression_model.pkl"):
        model_file = "best_classification_model.pkl" if os.path.exists("best_classification_model.pkl") else "best_regression_model.pkl"
        with open(model_file, "rb") as f:
            st.download_button("Download Model", f, file_name=model_file)
    else:
        st.warning("No model available. Please build and save a model first.")

# Predict Section
if choice == "Predict":
    st.title("Predict with Your Model")

    # Memuat model yang sesuai tanpa menambahkan ekstensi dua kali
    model_file = None
    if os.path.exists("best_classification_model.pkl"):
        model_file = "best_classification_model.pkl"
    elif os.path.exists("best_regression_model.pkl"):
        model_file = "best_regression_model.pkl"

    if model_file is None:
        st.warning("Please train and save a model first.")
    else:
        st.write("Model loaded:", model_file)

        # Load the model correctly
        if "classification" in model_file:
            model = clf_load(model_file)
        else:
            model = reg_load(model_file)

        test_file = st.file_uploader("Upload Your Test Data (CSV format only)", type=["csv"])
        if test_file:
            test_df = pd.read_csv(test_file)
            st.dataframe(test_df)

            # Menggabungkan test_df dengan predictions_df
            predictions = model.predict(test_df)

            predictions_df = pd.DataFrame(predictions, columns=['Predictions'])

            # Menggabungkan test_df dengan predictions_df
            result_df = pd.concat([test_df, predictions_df], axis=1)

            # Menampilkan tabel dengan hasil prediksi di Streamlit
            st.subheader("Test Data with Predictions")
            st.dataframe(result_df)

            #Opsional: Menyimpan hasil prediksi dalam file CSV
            predictions_csv = result_df.to_csv(index=False)
            st.download_button("Download Predictions", predictions_csv, file_name="predictions.csv", mime="text/csv")