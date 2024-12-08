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
    st.image('https://qtxasset.com/cdn-cgi/image/w=850,h=478,f=auto,fit=crop,g=0.5x0.5/https://qtxasset.com/quartz/qcloud4/media/image/fiercevideo/1554925532/googlecloud.jpg?VersionId=hJC0G.4VGlXbcc35EzyI9RhCJI.mslxN')
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
                # Define filenames for the models
                model_filename_classification = "best_classification_model.pkl"
                model_filename_regression = "best_regression_model.pkl"

                if mdl_type == "Classification":
                    clf_setup(df, target=target_col, session_id=123)
                    st.subheader("Setup Data Summary")
                    st.dataframe(clf_pull())

                    best_model = clf_compare()
                    st.subheader("Model Comparison")
                    st.dataframe(clf_pull())

                    st.success("Best classification model trained!")

                    # Save the best model
                    clf_save(best_model, model_filename_classification)
                    st.success(f"Best classification model saved as {model_filename_classification}!")

                elif mdl_type == "Regression":
                    reg_setup(df, target=target_col, session_id=123)
                    st.subheader("Setup Data Summary")
                    st.dataframe(reg_pull())

                    best_model = reg_compare()
                    st.subheader("Model Comparison")
                    st.dataframe(reg_pull())

                    st.success("Best regression model trained!")

                    # Save the best model
                    reg_save(best_model, model_filename_regression)
                    st.success(f"Best regression model saved as {model_filename_regression}!")

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.error("Please upload a dataset first.")

# Download Section for Model
if choice == "Download Model":
    model_filename_classification = "best_classification_model.pkl"
    model_filename_regression = "best_regression_model.pkl"

    if os.path.exists(model_filename_classification) or os.path.exists(model_filename_regression):
        model_file = model_filename_classification if os.path.exists(model_filename_classification) else model_filename_regression
        with open(model_file, "rb") as f:
            st.download_button("Download Model", f, file_name=model_file)
    else:
        st.warning("No model available. Please build and save a model first.")

# Predict Section
if choice == "Predict":
    st.title("Predict with Your Model")
    model_filename_classification = "best_classification_model.pkl"
    model_filename_regression = "best_regression_model.pkl"

    model_file = model_filename_classification if os.path.exists(model_filename_classification) else model_filename_regression

    if os.path.exists(model_file):
        st.write(f"Model file found: {model_file}")

        # Load the appropriate model based on type
        if model_file == model_filename_classification:
            model = clf_load(model_file)  # Load classification model
        elif model_file == model_filename_regression:
            model = reg_load(model_file)  # Load regression model

        test_file = st.file_uploader("Upload Your Test Data (CSV format only)", type=["csv"])
        if test_file:
            test_df = pd.read_csv(test_file)
            st.dataframe(test_df)

            # Perform prediction
            predictions = model.predict(test_df)

            # Combine test data with predictions in a single table
            test_df['Predictions'] = predictions
            st.subheader("Predictions with Test Data")
            st.dataframe(test_df)

            predictions_csv = test_df.to_csv(index=False)
            st.download_button("Download Predictions", predictions_csv, file_name="predictions.csv", mime="text/csv")
    else:
        st.warning("Please train and save a model first.")