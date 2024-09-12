import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from fpdf import FPDF
from datetime import datetime

# Page config
st.set_page_config(page_title="Laptop Price Predictor 🧑‍💻", layout="centered")

# Custom CSS for background color and button styles
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
    }
    .stButton > button {
        color: white;
        background-color: #ff4b4b; 
        border-radius: 10px;
        padding: 10px;
        border: none;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #e03e3e;
    }
    .green-box {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
        font-size: 20px;
        margin-top: 5px;
        margin-bottom: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the trained model pipeline and dataset
with open('pipe_yarbi_tkhdem.pkl', 'rb') as file:
    rf = pickle.load(file)

data = pd.read_csv("traineddata.csv")

# Initialize LabelEncoders
label_encoders = {
    'Company': LabelEncoder(),
    'TypeName': LabelEncoder(),
    'CPU_name': LabelEncoder(),
    'Gpu brand': LabelEncoder(),
    'OpSys': LabelEncoder()
}

# Fit the label encoders
for column, encoder in label_encoders.items():
    encoder.fit(data[column])

# Streamlit interface
st.title("💻 Laptop Price Predictor")
st.write("Use this tool to predict the price of a laptop based on various features.")

# Multi-step form using session state
if 'step' not in st.session_state:
    st.session_state.step = 1

# Step 1 - General Information
if st.session_state.step == 1:
    st.header("Step 1: General Information")

    col1, col2 = st.columns(2)
    with col1:
        company = st.selectbox('Brand', data['Company'].unique())
    with col2:
        type = st.selectbox('Type', data['TypeName'].unique())

    col3, col4 = st.columns(2)
    with col3:
        ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
    with col4:
        os = st.selectbox('OS', data['OpSys'].unique())

    col5, col6 = st.columns(2)
    with col5:
        weight = st.number_input('Weight (kg)', min_value=0.0, step=0.1)

    _, next_button_col = st.columns([4, 1])
    with next_button_col:
        if st.button("Next ➡️"):
            st.session_state.company = company
            st.session_state.type = type
            st.session_state.ram = ram
            st.session_state.os = os
            st.session_state.weight = weight
            st.session_state.step = 2

# Step 2 - Performance & Screen Details
elif st.session_state.step == 2:
    st.header("Step 2: Screen & Performance Details")

    col1, col2 = st.columns(2)
    with col1:
        touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
    with col2:
        ips = st.selectbox('IPS', ['No', 'Yes'])

    col3, col4 = st.columns(2)
    with col3:
        screen_size = st.number_input('Screen Size (in inches)', min_value=0.1, step=0.1)
    with col4:
        resolution = st.selectbox('Resolution', [
            '1920x1080', '1366x768', '1600x900', '3840x2160',
            '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'
        ])

    col5, col6 = st.columns(2)
    with col5:
        cpu = st.selectbox('CPU', data['CPU_name'].unique())
    with col6:
        gpu = st.selectbox('GPU', data['Gpu brand'].unique())

    col7, col8 = st.columns(2)
    with col7:
        hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
    with col8:
        ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])

    back_button_col, next_button_col = st.columns([1, 1])
    with back_button_col:
        if st.button("⬅️ Back"):
            st.session_state.step = 1

    with next_button_col:
        if st.button('💡 Predict Price'):
            try:
                touchscreen = 1 if touchscreen == 'Yes' else 0
                ips = 1 if ips == 'Yes' else 0

                try:
                    X_resolution, Y_resolution = map(int, resolution.split('x'))
                    ppi = ((X_resolution ** 2) + (Y_resolution ** 2)) ** 0.5 / screen_size
                except ZeroDivisionError:
                    st.error("Screen size must be greater than 0")
                    ppi = 0

                query_df = pd.DataFrame({
                    'Company': [st.session_state.company],
                    'TypeName': [st.session_state.type],
                    'Ram': [int(st.session_state.ram)],
                    'Weight': [float(st.session_state.weight)],
                    'TouchScreen': [touchscreen],
                    'IPS': [ips],
                    'PPI': [float(ppi)],
                    'CPU_name': [cpu],
                    'HDD': [int(hdd)],
                    'SSD': [int(ssd)],
                    'Gpu brand': [gpu],
                    'OpSys': [st.session_state.os]
                })

                if query_df.isnull().values.any():
                    st.error("Input contains NaN values. Please ensure all inputs are filled in correctly.")
                else:
                    try:
                        prediction = int(np.exp(rf.predict(query_df)[0]) / 10)

                        st.session_state.user_input_summary = f"""
                        **Brand**: {st.session_state.company}\n
                        **Type**: {st.session_state.type}\n
                        **RAM**: {st.session_state.ram} GB\n
                        **Operating System**: {st.session_state.os}\n
                        **Weight**: {st.session_state.weight} kg\n
                        **Touchscreen**: {'Yes' if touchscreen else 'No'}\n
                        **IPS**: {'Yes' if ips else 'No'}\n
                        **Screen Size**: {screen_size} inches\n
                        **Resolution**: {resolution}\n
                        **PPI**: {ppi:.2f}\n
                        **CPU**: {cpu}\n
                        **GPU**: {gpu}\n
                        **HDD**: {hdd} GB\n
                        **SSD**: {ssd} GB\n
                        """
                        st.session_state.prediction = prediction
                        st.session_state.step = 3

                    except Exception as pred_error:
                        st.error(f"Error during prediction: {pred_error}")

            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

# Step 3 - Results Page with PDF Download
elif st.session_state.step == 3:
    st.header("📋 Summary of Your Input & Prediction")

    st.markdown(st.session_state.user_input_summary)

    st.markdown(f"""
    <div class="green-box">
        💻 Predicted Price: {st.session_state.prediction - 100} MAD to {st.session_state.prediction + 100} MAD 💵
    </div>
    """, unsafe_allow_html=True)

    # Function to create and download PDF using default fonts
    def create_pdf(user_input_summary, prediction):
        pdf = FPDF()
        pdf.add_page()

        # Use Arial font (default Windows font)
        pdf.set_font("Arial", 'B', 16)
        
        # Title
        pdf.set_text_color(0, 102, 204)  # Blue color for the title
        pdf.cell(200, 10, txt="Laptop Price Prediction", ln=True, align='C')

        # User Input Summary in standard text
        pdf.set_font("Arial", '', 12)
        pdf.set_text_color(0, 0, 0)  # Black color for the text
        pdf.ln(10)
        pdf.multi_cell(200, 10, txt=user_input_summary)
        
        pdf.ln(10)

        # Predicted Price in bold green text, centered
        pdf.set_font("Arial", 'B', 14)
        pdf.set_text_color(34, 139, 34)  # Green color for the predicted price
        pdf.cell(200, 10, txt=f"Predicted Price: {prediction - 100} MAD to {prediction + 100} MAD", ln=True, align='C')

        pdf.ln(10)
        
        # Footer with gray color and italic text
        pdf.set_font("Arial", 'I', 10)
        pdf.set_text_color(128, 128, 128)  # Gray color for the footer
        pdf.cell(200, 10, txt=f"Prediction Date: {datetime.now().strftime('%Y-%m-%d')}", ln=True, align='C')

        return pdf

    # Streamlit button to download the PDF
    if st.button("Download PDF"):
        pdf = create_pdf(st.session_state.user_input_summary, st.session_state.prediction)
        pdf_output = pdf.output(dest='S').encode('latin1')  # This will work fine now with default fonts and no emojis
        st.download_button(label="Download PDF", data=pdf_output, file_name="laptop_price_prediction.pdf", mime="application/pdf")