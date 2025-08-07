import os
import warnings
import io
import json
import datetime

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2

# Suppress TensorFlow and protobuf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')
warnings.filterwarnings('ignore', message='.*tf.reset_default_graph.*')

import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, LSTM
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MinMaxScaler

from ultralytics import YOLO
import random

# --- Constants and disease classes ---
YOLO_MODEL_PATH = "yolov8n-seg.pt"  # Replace with your fine-tuned onion disease model weights
DISEASE_CLASSES = ['Healthy', 'Anthracnose', 'Purple Blotch', 'Stemphylium Blight', 'Twister']

# --- Model loading utilities ---
@st.cache_resource(show_spinner=False)
def load_segmentation_model(model_path=YOLO_MODEL_PATH):
    if not os.path.exists(model_path):
        st.warning(f"Model '{model_path}' not found locally. Downloading weights...")
    try:
        model = YOLO(model_path)
        st.success(f"YOLOv8 segmentation model '{model_path}' loaded! 🎉")
        return model
    except Exception as e:
        st.error(f"Failed to load YOLOv8 model: {e}")
        st.stop()
        return None

def preprocess_image(image: Image.Image, target_size=(224,224)):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    arr = img_to_array(image) / 255.0
    return np.expand_dims(arr, axis=0)

def segment_image(image: Image.Image, model: YOLO, confidence_threshold: float):
    img_cv = np.array(image.convert('RGB'))
    img_cv = img_cv[:, :, ::-1].copy()  # RGB to BGR
    results = model(img_cv, conf=confidence_threshold, verbose=False)
    img_annotated = img_cv.copy()
    overlay = img_annotated.copy()
    alpha = 0.4
    detected_objects_info = []
    polygons = []
    num_objects = 0
    if results and results[0].masks is not None:
        for r in results:
            if r.masks is None:
                continue
            for mask, box in zip(r.masks.xy, r.boxes):
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                object_color = [random.randint(0,255) for _ in range(3)]
                mask_polygon = np.int32([mask])
                polygons.append(mask_polygon.tolist())
                cv2.fillPoly(overlay, [mask_polygon], object_color)
                cv2.polylines(img_annotated, [mask_polygon], isClosed=True, color=object_color, thickness=2)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{class_name} {confidence:.2f}"
                text_color = (255,255,255) if sum(object_color) < 300 else (0,0,0)
                text_x = x1
                text_y = y1 - 10 if y1 > 20 else y1 + 20
                (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img_annotated, (text_x, text_y - label_h - baseline),
                              (text_x + label_w, text_y + baseline), object_color, cv2.FILLED)
                cv2.putText(img_annotated, label, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)
                detected_objects_info.append(f"- **{class_name}** (Confidence: {confidence:.2f})")
                num_objects += 1
        cv2.addWeighted(overlay, alpha, img_annotated, 1 - alpha, 0, img_annotated)
    img_annotated_pil = Image.fromarray(img_annotated[:, :, ::-1])
    return img_annotated_pil, num_objects, detected_objects_info, polygons

# --- Price Prediction Models ---
@st.cache_data(show_spinner=False)
def generate_synthetic_price_data(days=365):
    np.random.seed(42)
    dates = pd.date_range(end=datetime.datetime.now(), periods=days)
    trend = np.linspace(1000, 1500, days)
    seasonal = 200 * np.sin(2 * np.pi * np.arange(days) / 365.25)
    noise = np.random.normal(0, 50, days)
    prices = trend + seasonal + noise
    prices = np.maximum(prices, 500)
    return pd.DataFrame({'date': dates, 'price': prices})

@st.cache_data(show_spinner=False)
def prepare_lstm_data(df, time_steps=30):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['price'].values.reshape(-1, 1))
    X, y = [], []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i-time_steps:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y, scaler

@st.cache_resource(show_spinner=False)
def get_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# --- Disease CNN classifier ---
@st.cache_resource(show_spinner=False)
def get_cnn_model(num_classes=5, input_shape=(224,224,3)):
    base = EfficientNetB3(weights=None, include_top=False, input_shape=input_shape)
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- Modules with descriptive expanders and improved layout ---

def price_prediction_module():
    st.header("1. 🧅 Onion Price Prediction - LSTM Time Series Regression")
    with st.expander("About this module"):
        st.write("""
        This module uses an LSTM recurrent neural network to predict onion prices based on historical data.
        It captures trends and seasonal patterns with sequential data encoding.
        Outputs estimated price for the next day to assist market planning.
        """)
    df = generate_synthetic_price_data()
    time_steps = 30
    X, y, scaler = prepare_lstm_data(df, time_steps=time_steps)
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    model = get_lstm_model(X_train.shape[1:])
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Train Model (Approx 15-20 seconds)"):
            with st.spinner("Training LSTM model on synthetic onion price data..."):
                model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
            st.success("Model trained!")
    with col2:
        st.subheader("Recent Onion Price Trend (Last 60 Days)")
        st.line_chart(df.set_index('date').price.tail(60))
    with st.expander(f"Enter last {time_steps} days onion prices for prediction"):
        user_input = []
        last_prices = df['price'].values[-time_steps:]
        for i in range(time_steps):
            val = st.number_input(f"Price Day -{time_steps - i} (₹/quintal)",
                                  value=float(last_prices[i]), step=1.0, format="%.2f", key=f'price_{i}')
            user_input.append(val)
        if st.button("Predict Next Day Price"):
            if len(user_input) == time_steps:
                arr = np.array(user_input).reshape(-1,1)
                scaled_input = scaler.transform(arr)
                X_input = scaled_input.reshape((1, time_steps, 1))
                scaled_pred = model.predict(X_input)
                pred_price = scaler.inverse_transform(scaled_pred)[0][0]
                st.success(f"Predicted Onion Price for Next Day: ₹{pred_price:.2f} per quintal")
            else:
                st.error(f"Please provide exactly {time_steps} prices.")

def quality_grading_module():
    st.header("2. 🧅 Onion Quality Grading - Image Segmentation and Polygon Visualization")
    with st.expander("Module Description"):
        st.write("""
        This module applies YOLOv8 instance segmentation to detect and segment individual onion bulbs.
        It allows automated grading by extracting bulb regions with polygon masks.
        Users can adjust detection confidence and download annotations.
        """)
    uploaded_img = st.file_uploader("Upload Onion Image for Quality Grading", type=['jpg','jpeg','png'], key="quality_upload")
    if uploaded_img:
        image = Image.open(uploaded_img).convert('RGB')
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
        confidence_threshold = st.slider("Confidence Threshold for Polygon Segmentation", 0.0, 1.0, 0.3, 0.05, key="quality_confidence")
        yolo_model = load_segmentation_model()
        if yolo_model:
            annotated_image, num_objects, detected_objects_info, polygons = segment_image(image, yolo_model, confidence_threshold)
            if annotated_image:
                st.subheader("Annotated Image with Polygons")
                st.image(annotated_image, use_container_width=True)
                if num_objects > 0:
                    st.success(f"Detected {num_objects} object(s).")
                    with st.expander("View Detected Objects Details"):
                        for i, info in enumerate(detected_objects_info):
                            st.markdown(f"{i+1}. {info}")
                    buffer = io.BytesIO()
                    annotated_image.save(buffer, format="PNG")
                    buffer.seek(0)
                    st.download_button("Download Annotated Image (PNG)", buffer, "annotated_onion_quality.png", "image/png")
                    polygons_json = json.dumps({"polygons": polygons})
                    st.download_button("Download Polygon Coordinates (JSON)", polygons_json, "quality_polygons.json", "application/json")
                else:
                    st.warning("No objects detected. Try lowering confidence threshold.")
            else:
                st.error("Failed to create annotated image.")
        else:
            st.error("YOLOv8 segmentation model could not be loaded.")

def disease_detection_module():
    st.header("3. 🧅 Onion Disease Detection - Segmentation and CNN Classification")
    with st.expander("Module Description"):
        st.write("""
        Detects multiple onion leaf diseases by combining YOLOv8 segmentation masks with EfficientNetB3-based CNN classification.
        Diseases classified include Healthy, Anthracnose, Purple Blotch, Stemphylium Blight, and Twister.
        Supports confidence scoring and polygon-based disease localization.
        """)
    uploaded_img = st.file_uploader("Upload Onion Leaf Image for Disease Detection", type=['jpg','jpeg','png'], key='disease_upload')
    if uploaded_img:
        image = Image.open(uploaded_img).convert('RGB')
        st.subheader("Uploaded Leaf Image")
        st.image(image, use_container_width=True)
        confidence_threshold = st.slider("Confidence Threshold for Polygon Segmentation", 0.0, 1.0, 0.3, 0.05, key="disease_confidence")
        yolo_model = load_segmentation_model()
        if yolo_model is None:
            st.error("Failed to load YOLOv8 model for segmentation.")
            return
        annotated_image, num_objects, detected_objects_info, polygons = segment_image(image, yolo_model, confidence_threshold)
        if annotated_image:
            st.subheader("Annotated Image with Polygons")
            st.image(annotated_image, use_container_width=True)
            if num_objects > 0:
                st.success(f"Detected {num_objects} object(s) with polygon segmentation.")
                with st.expander("View Detected Objects Details"):
                    for i, info in enumerate(detected_objects_info):
                        st.markdown(f"{i+1}. {info}")
                buffer = io.BytesIO()
                annotated_image.save(buffer, format="PNG")
                buffer.seek(0)
                st.download_button("Download Annotated Image (PNG)", buffer, "annotated_onion_disease.png", "image/png")
                polygons_json = json.dumps({"polygons": polygons})
                st.download_button("Download Polygon Coordinates (JSON)", polygons_json, "disease_polygons.json", "application/json")
            else:
                st.warning("No objects detected. Try lowering confidence threshold.")
        else:
            st.error("Failed to create annotated image.")
        model = get_cnn_model(num_classes=5)
        if st.button("Detect Disease"):
            with st.spinner("Detecting disease..."):
                img_tensor = preprocess_image(image)
                preds = model.predict(img_tensor)[0]
                confidences = {d: float(round(preds[i],4)) for i,d in enumerate(DISEASE_CLASSES)}
                pred_disease = DISEASE_CLASSES[np.argmax(preds)]
                st.success(f"Detected Disease Status: **{pred_disease}**")
                st.json(confidences)

def yield_forecasting_module():
    st.header("4. 🧅 Onion Yield Forecasting - Regression")
    with st.expander("Module Description"):
        st.write("""
        Predicts expected onion yield based on environmental factors such as rainfall, temperature,
        soil nitrogen, and irrigation count using a simple regression formula.
        """)
    with st.expander("Enter Agronomic and Environmental Features"):
        rainfall = st.number_input("Cumulative Rainfall Last Month (mm)", min_value=0.0, max_value=500.0, value=100.0, step=1.0)
        avg_temp = st.number_input("Average Temperature (°C)", min_value=0.0, max_value=50.0, value=27.0, step=0.1)
        soil_nitrogen = st.number_input("Soil Nitrogen Level (%)", min_value=0.0, max_value=10.0, value=0.3, step=0.01)
        irrigation_count = st.number_input("Irrigation Count Last Month", min_value=0, max_value=30, value=5, step=1)
        if st.button("Forecast Yield"):
            with st.spinner("Forecasting yield..."):
                predicted_yield = (rainfall * 0.018) + (avg_temp * 1.4) + (soil_nitrogen * 110) + (irrigation_count * 4)
                st.success(f"Predicted Yield: **{predicted_yield:.2f}** tons/hectare")

def storage_optimization_module():
    st.header("5. 🧅 Storage Optimization - Reinforcement Learning Stub")
    with st.expander("Module Description"):
        st.write("""
        Estimates recommended storage temperature and humidity to minimize spoilage risk.
        Currently uses heuristic thresholds and plans to extend to a reinforcement learning model.
        """)
    temp = st.slider("Storage Temperature (°C)", 0, 25, 10)
    humidity = st.slider("Humidity Level (%)", 40, 90, 65)
    optimal_temp = max(5, min(15, 25 - temp))
    optimal_humidity = max(50, min(80, 70 - (humidity - 65) * 0.5))
    risk_level = "Low"
    if abs(temp - optimal_temp) > 5 or abs(humidity - optimal_humidity) > 10:
        risk_level = "Moderate"
    elif abs(temp - optimal_temp) > 10 or abs(humidity - optimal_humidity) > 20:
        risk_level = "High"
    if st.button("Optimize Storage Conditions"):
        st.success("Recommended Storage Parameters:")
        st.write(f"- Temperature: {optimal_temp:.1f} °C")
        st.write(f"- Humidity: {optimal_humidity:.1f} %")
        st.info(f"Estimated Spoilage Risk: {risk_level}")
        st.caption("Integration of a trained RL model is planned for production.")

def seed_production_optimization_module():
    st.header("6. 🧅 Seed Production Optimization")
    with st.expander("Module Description"):
        st.write("""
        AI-driven seed quality assessment from uploaded seed or umbel images using CNN classifiers.
        Includes umbel counting and disease detection to optimize seed production.
        """)
    uploaded_files = st.file_uploader("Upload Seed or Umbel Images (multiple)", type=['jpg','jpeg','png'], accept_multiple_files=True)
    if uploaded_files:
        model = get_cnn_model(num_classes=5)  # Use disease detection CNN model
        disease_classes = ['Healthy', 'Anthracnose', 'Purple Blotch', 'Stemphylium Blight', 'Twister']
        umbel_counts = []
        disease_results = []
        for f in uploaded_files:
            image = Image.open(f).convert('RGB')
            st.image(image, caption=f"Image: {f.name}")
            img_tensor = preprocess_image(image)
            preds = model.predict(img_tensor)[0]
            pred_disease = disease_classes[np.argmax(preds)]
            confidence = preds[np.argmax(preds)]
            disease_results.append((pred_disease, confidence))
            
            # Simple umbel counting heuristic: count contours in grayscale image
            gray = np.array(image.convert('L'))
            _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            umbel_count = len(contours)
            umbel_counts.append(umbel_count)
            
            st.write(f"Disease Prediction: {pred_disease} (Confidence: {confidence:.2f})")
            st.write(f"Estimated Umbel Count: {umbel_count}")
        
        # Summary
        avg_umbel_count = int(np.mean(umbel_counts)) if umbel_counts else 0
        st.info(f"Average Estimated Umbel Count: {avg_umbel_count}")
        
        disease_summary = {}
        for d, conf in disease_results:
            disease_summary[d] = disease_summary.get(d, 0) + 1
        st.info("Disease Detection Summary:")
        for d, count in disease_summary.items():
            st.write(f"- {d}: {count} images")
    else:
        st.info("Upload seed or umbel images for quality assessment.")

def hydroponic_growth_monitoring_module():
    st.header("7. 🧅 Hydroponic Growth Monitoring")
    with st.expander("Module Description"):
        st.write("""
        Monitors hydroponic growth by visualizing uploaded sensor data including pH, EC, temperature, and humidity.
        Future plans include ML-based growth stage prediction and nutrient optimization.
        """)
    uploaded_file = st.file_uploader("Upload Sensor Data CSV", type=['csv'])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        
        # Let user select columns for visualization
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            st.subheader("Select parameters to visualize:")
            selected_cols = st.multiselect("Choose columns", numeric_cols, default=numeric_cols[:3])
            if selected_cols:
                st.line_chart(df[selected_cols])
            else:
                st.info("Please select at least one numeric column to visualize.")
        else:
            st.warning("No numeric columns found in the uploaded file.")
            
        # Simple growth health status prediction based on average pH and EC
        avg_pH = df['pH'].mean() if 'pH' in df.columns else None
        avg_EC = df['EC'].mean() if 'EC' in df.columns else None
        
        health_status = "Unknown"
        if avg_pH is not None and avg_EC is not None:
            if 5.5 <= avg_pH <= 6.5 and 1.0 <= avg_EC <= 2.5:
                health_status = "Optimal Growth Conditions"
            elif avg_pH < 5.5 or avg_EC < 1.0:
                health_status = "Nutrient Deficiency Risk"
            elif avg_pH > 6.5 or avg_EC > 2.5:
                health_status = "Nutrient Toxicity Risk"
            else:
                health_status = "Suboptimal Conditions"
            st.success(f"Growth Health Status Prediction: {health_status}")
        else:
            st.info("pH and EC columns not found for growth health prediction.")
    else:
        st.info("Upload hydroponic sensor data CSV for visualization.")

def weed_detection_control_module():
    st.header("8. 🧅 Weed Detection & Control")
    with st.expander("Module Description"):
        st.write("""
        Allows upload of UAV or field images to detect weeds using object detection models.
        Planned integration of YOLOv8 fine-tuned for weed-crop distinction.
        """)
    uploaded_img = st.file_uploader("Upload UAV/Field Image", type=['jpg','jpeg','png'])
    if uploaded_img:
        image = Image.open(uploaded_img).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.info("Weed detection model integration is under development.")
    else:
        st.info("Upload field or UAV images to detect weeds.")

def disease_outbreak_forecasting_module():
    st.header("9. 🧅 Disease Outbreak Forecasting")
    with st.expander("Module Description"):
        st.write("""
        Predicts onion disease outbreak risk based on environmental inputs like temperature, humidity,
        rainfall, and soil moisture. Uses heuristic risk scoring with plans for ML integration.
        """)
    temperature = st.number_input("Average Temperature (°C)", value=27.0)
    humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=65)
    rainfall = st.number_input("Rainfall Last Week (mm)", value=10.0)
    soil_moisture = st.number_input("Soil Moisture (%)", min_value=0, max_value=100, value=50)
    if st.button("Forecast Disease Outbreak Risk"):
        risk_score = 0.3 * humidity + 0.4 * rainfall + 0.2 * soil_moisture - 0.1 * temperature
        risk_level = "Low"
        if risk_score > 70:
            risk_level = "High"
        elif risk_score > 40:
            risk_level = "Moderate"
        st.success(f"Disease Outbreak Risk Level: {risk_level}")

def harvest_scheduling_module():
    st.header("10. 🧅 Harvest Scheduling")
    with st.expander("Module Description"):
        st.write("""
        Provides harvest timing recommendations based on crop maturity, labor availability,
        and current market prices to maximize profitability and logistical efficiency.
        """)
    days_to_maturity = st.number_input("Days until crop maturity", min_value=0, max_value=120, value=45)
    labor_availability = st.number_input("Available laborers", min_value=0, max_value=100, value=10)
    market_price = st.number_input("Current Market Price (₹/quintal)", min_value=0.0, value=1000.0)
    if st.button("Get Harvest Schedule Recommendation"):
        if days_to_maturity < 7:
            suggestion = "Harvest immediately to maximize freshness."
        elif market_price > 1200:
            suggestion = "Consider early harvest to sell at high price."
        else:
            suggestion = "Optimal harvest time in a few days; monitor crop health."
        st.info(suggestion)

def storage_spoilage_risk_prediction_module():
    st.header("11. 🧅 Storage Spoilage Risk Prediction")
    with st.expander("Module Description"):
        st.write("""
        Estimates the risk of spoilage for stored onions based on temperature, humidity, and storage duration.
        Aims to integrate reinforcement learning models for adaptive environment management in future.
        """)
    temp = st.slider("Storage Temperature (°C)", 0, 30, 15, key="storage_temp")
    humidity = st.slider("Storage Humidity (%)", 20, 90, 60, key="storage_humidity")
    duration = st.number_input("Storage Duration (days)", min_value=1, max_value=365, value=30, key="storage_duration")
    
    # Calculate risk dynamically on input change
    risk = 0.05 * duration + 0.1 * (humidity - 50) + 0.15 * max(0, temp - 10)
    risk_level = "Low"
    if risk > 15:
        risk_level = "High"
    elif risk > 7:
        risk_level = "Moderate"
    
    st.success(f"Estimated Spoilage Risk: {risk_level} (Risk Score: {risk:.2f})")

# --- Main App ---

st.set_page_config(page_title="Onion AI/ML Suite", page_icon="🧅", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better styling and attractiveness
st.markdown(
    """
    <style>
    /* General page styling */
    .stApp {
        background-color: #f0f8ff;
        color: #2f4f4f;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        transition: background-color 0.5s ease;
    }

    /* Header styling */
    .css-1v3fvcr h1 {
        color: #228b22;
        font-weight: 800;
        font-size: 3rem;
        text-align: center;
        margin-bottom: 1.5rem;
        text-shadow: 1px 1px 2px #a9a9a9;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #d0e8ff;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(34, 139, 34, 0.2);
    }

    /* Button styling */
    button[kind="primary"] {
        background-color: #228b22 !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 0.7rem 1.5rem !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 8px rgba(34, 139, 34, 0.3);
        transition: background-color 0.3s ease, box-shadow 0.3s ease;
    }
    button[kind="primary"]:hover {
        background-color: #32cd32 !important;
        box-shadow: 0 6px 12px rgba(50, 205, 50, 0.5);
    }

    /* Sidebar selectbox */
    .css-1v3fvcr select {
        border-radius: 12px;
        border: 2px solid #228b22;
        padding: 0.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        color: #2f4f4f;
        background-color: #f0fff0;
        transition: border-color 0.3s ease;
    }
    .css-1v3fvcr select:hover {
        border-color: #32cd32;
    }

    /* Card styling for modules */
    .stExpander {
        background-color: #ffffff;
        border-radius: 15px;
        box-shadow: 0 6px 15px rgba(34, 139, 34, 0.15);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: box-shadow 0.3s ease;
    }
    .stExpander:hover {
        box-shadow: 0 8px 20px rgba(50, 205, 50, 0.3);
    }

    /* Metrics styling */
    .stMetric {
        background-color: #d0e8ff;
        border-radius: 15px;
        padding: 1.2rem;
        margin: 0.7rem 0;
        font-weight: 800;
        color: #228b22;
        box-shadow: 0 4px 10px rgba(34, 139, 34, 0.25);
        transition: box-shadow 0.3s ease;
    }
    .stMetric:hover {
        box-shadow: 0 6px 15px rgba(50, 205, 50, 0.4);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🧅 Comprehensive Onion AI/ML System")

modules = [
    "Price Prediction",
    "Quality Grading",
    "Disease Detection",
    "Yield Forecasting",
    "Storage Optimization",
    "Seed Production Optimization",
    "Hydroponic Growth Monitoring",
    "Weed Detection & Control",
    "Disease Outbreak Forecasting",
    "Harvest Scheduling",
    "Storage Spoilage Risk Prediction"
]

choice = st.sidebar.selectbox("Select Module", modules)

# Dispatch to corresponding module function
if choice == "Price Prediction":
    price_prediction_module()
elif choice == "Quality Grading":
    quality_grading_module()
elif choice == "Disease Detection":
    disease_detection_module()
elif choice == "Yield Forecasting":
    yield_forecasting_module()
elif choice == "Storage Optimization":
    storage_optimization_module()
elif choice == "Seed Production Optimization":
    seed_production_optimization_module()
elif choice == "Hydroponic Growth Monitoring":
    hydroponic_growth_monitoring_module()
elif choice == "Weed Detection & Control":
    weed_detection_control_module()
elif choice == "Disease Outbreak Forecasting":
    disease_outbreak_forecasting_module()
elif choice == "Harvest Scheduling":
    harvest_scheduling_module()
elif choice == "Storage Spoilage Risk Prediction":
    storage_spoilage_risk_prediction_module()

with st.sidebar:
    st.title("🧅 About Onion AI/ML Suite")
    st.markdown("---")
    st.markdown("""
This application provides an advanced AI/ML system for managing onion farming, covering:

- **Price Prediction:** Market price forecasting using LSTM.
- **Quality Grading:** Onion bulb segmentation via YOLOv8 instance segmentation.
- **Disease Detection:** Multi-class onion leaf disease detection combining YOLOv8 and CNN.
- **Yield Forecasting:** Regression using key environmental and agronomic variables.
- **Storage Optimization:** Heuristic-based storage parameter recommendations (RL planned).
- **Seed Production Optimization:** Seed quality classification and umbel monitoring (future).
- **Hydroponic Growth Monitoring:** Sensor data visualization and growth prediction (future).
- **Weed Detection & Control:** UAV imagery analysis for weed management (future).
- **Disease Outbreak Forecasting:** Environmental risk prediction for disease outbreaks (future).
- **Harvest Scheduling:** Optimized harvest timing based on maturity, labor, and market prices (future).
- **Storage Spoilage Risk Prediction:** Spoilage risk estimation with plans for RL integration (future).

Developed by Janhavi ❤️ using Python, TensorFlow, YOLOv8, and Streamlit.
""")
    # Uncomment if you add an image to your app folder or host it
    # st.image("onion_disease_diagram.jpg", caption="Onion Disease Overview")
    st.markdown("---")
    st.caption("© 2025 Onion AI/ML Team")

st.markdown("---")
st.caption(f"App version: 1.1.0 | Last updated: {datetime.datetime.now().strftime('%B %d, %Y')}")
