

api_url = "http://localhost:8502"
import streamlit.components.v1 as components
import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import plotly.express as px
from io import StringIO
import pandas as pd

st.title("Fraud Detection App")

# Display site header
#image = Image.open("../images/dsif header.jpeg")

image_path = "../images/dsif header 2.jpeg"
try:
    # Open and display the image
    img = Image.open(image_path)
    st.image(img, use_column_width=True)  # Caption and resizing options
except FileNotFoundError:
    st.error(f"Image not found at {image_path}. Please check the file path.")

transaction_amount = st.number_input("Transaction Amount")
customer_age = st.number_input("Customer Age")
customer_balance = st.number_input("Customer Balance")

data = {
        "transaction_amount": transaction_amount,
        "customer_age": customer_age,
        "customer_balance": customer_balance
    }

if st.button("Show Feature Importance"):
    import matplotlib.pyplot as plt
    response = requests.get(f"{api_url}/feature-importance")
    feature_importance = response.json().get('feature_importance', {})

    features = list(feature_importance.keys())
    importance = list(feature_importance.values())

    fig, ax = plt.subplots()
    ax.barh(features, importance)
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance')
    st.pyplot(fig)

if st.button("Predict and show prediction confidence"):
    # Make the API call

    response = requests.post(f"{api_url}/predict/",
                            json=data)

    result = response.json()
    confidence = result['confidence']

    if result['fraud_prediction'] == 0:
        st.write("Prediction: Not fraudulent")
    else:
        st.write("Prediction: Fraudulent")

    # Confidence Interval Visualization
    labels = ['Not Fraudulent', 'Fraudulent']
    fig, ax = plt.subplots()
    ax.bar(labels, confidence, color=['green', 'red'])
    ax.set_ylabel('Confidence')
    ax.set_title('Prediction Confidence')
    st.pyplot(fig)

if st.button("Predict and show SHAP values"):
    response = requests.post(f"{api_url}/predict/",
                             json=data)
    st.write(response)
    result = response.json()

    if result['fraud_prediction'] == 0:
        st.write("Prediction: Not fraudulent")
    else:
        st.write("Prediction: Fraudulent")

    ######### SHAP #########
    # Extract SHAP values and feature names
    shap_values = np.array(result['shap_values'])
    features = result['features']

    # Display SHAP values
    st.subheader("SHAP Values Explanation")

    # Bar plot for SHAP values
    fig, ax = plt.subplots()
    ax.barh(features, shap_values[0])
    ax.set_xlabel('SHAP Value (Impact on Model Output)')
    st.pyplot(fig)

# File uploader for the CSV
csv_transactions = st.file_uploader("Choose a CSV file", type=["csv"])

# Define the function to send the CSV to the FastAPI endpoint and get predictions
def get_predictions(file):
    try:
        file.seek(0)
        files = {"csv_transactions": file}
        response = requests.post(f"{api_url}/batch_scoring", files=files)

        if response.status_code == 200:
            predictions_list = response.json()  # Use response.json() to parse JSON
            predictions_df = pd.DataFrame(predictions_list)
            return predictions_df
        else:
            st.error(f"Error from API: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"An error occurred in get_predictions: {e}")
        return None

# Function to convert DataFrame to CSV format for download
@st.cache_data
def convert_df(df):
    try:
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.write("CSV data generated successfully.")
        return csv_data
    except Exception as e:
        st.error(f"Error converting DataFrame to CSV: {e}")
        return None

# If a file is uploaded, send it to FastAPI and display predictions
if csv_transactions is not None:
    st.success("File uploaded successfully, making predictions...")

    try:
        # Get predictions from FastAPI
        predictions_df = get_predictions(csv_transactions)

        if predictions_df is not None and not predictions_df.empty:
            st.write("Predictions DataFrame:")
            st.write(predictions_df)

            # Create a new feature: Transaction Amount to Balance Ratio
            if 'transaction_amount' in predictions_df.columns and 'customer_balance' in predictions_df.columns:
                predictions_df['amount_to_balance_ratio'] = (
                    predictions_df['transaction_amount'] / predictions_df['customer_balance']
                )
                st.write("Transaction Amount to Balance Ratio feature added.")
            else:
                st.warning("Required columns for 'amount_to_balance_ratio' not found.")

            # Display the predictions DataFrame in the app
            st.dataframe(predictions_df)

            # Convert the DataFrame to CSV for download
            csv = convert_df(predictions_df)

            if csv is not None:
                # Code to give option to choose where to save downloaded CSV
                js = (
                        """
                    <button type="button" id="picker">Download predictions as CSV</button>
    
                    <script>
    
                    async function run() {
                        console.log("Running")
                      const handle = await showSaveFilePicker({
                          suggestedName: 'data.csv',
                          types: [{
                              description: 'CSV Data',
                              accept: {'text/plain': ['.csv']},
                          }],
                      });
                    """
                        + f"const blob = new Blob([`{csv}`]);"
                        + """

                  const writableStream = await handle.createWritable();
                  await writableStream.write(blob);
                  await writableStream.close();
                }

                document.getElementById("picker").onclick = run
                console.log("Done")
                </script>

                """
                )

                components.html(js, height=30)

                # Code for normal Streamlit download button, which I think looks nicer

                # Create a download button for the user to download the predictions as CSV
                # st.download_button(
                #     label="Download predictions as CSV",
                #     data=csv,
                #     file_name="predictions.csv",
                #     mime="text/csv",
                # )
            else:
                st.error("Failed to convert predictions to CSV for download.")

            # Interactive Scatter Plot
            st.subheader("Interactive Scatter Plot")

            # Select only numerical columns for the scatter plot
            numerical_columns = predictions_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            st.write("Numerical Columns Available for Plotting:")
            st.dataframe(numerical_columns)

            if len(numerical_columns) > 1:
                # Let the user select the X-axis and Y-axis
                x_axis = st.selectbox("Select the X-axis", numerical_columns)
                y_axis = st.selectbox("Select the Y-axis", numerical_columns)

                # Create the scatter plot using Plotly
                scatter_fig = px.scatter(
                    predictions_df, x=x_axis, y=y_axis,
                    title=f"Scatter plot of {x_axis} vs {y_axis}"
                )
                st.plotly_chart(scatter_fig)
            else:
                st.warning("Not enough numerical columns to create a scatter plot.")
        else:
            st.error("No predictions were returned from the API.")
    except Exception as e:
        st.error(f"An exception occurred: {e}")
        st.stop()  # Stop execution if an exception occurs
else:
    st.write("No file uploaded yet.")