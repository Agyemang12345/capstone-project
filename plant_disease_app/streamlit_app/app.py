"""
Plant Disease Detection Streamlit App
A web interface for detecting plant diseases from leaf images using deep learning.
"""

import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os
import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.predict import PlantDiseasePredictior, get_disease_info, load_predictor_with_config


# Page configuration
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .stApp {
        background-color: #f5f5f5;
    }
    .header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: white;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .disease-name {
        font-size: 28px;
        font-weight: bold;
        color: #333;
        margin: 0.5rem 0;
    }
    .confidence-score {
        font-size: 20px;
        color: #667eea;
        font-weight: bold;
    }
    .info-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #f0f4ff;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .solution-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        font-size: 14px;
        border-top: 1px solid #ddd;
        margin-top: 3rem;
    }
    .metric-card {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: white;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .top-predictions {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
    }
    .prediction-badge {
        padding: 1rem;
        border-radius: 8px;
        background-color: #f5f5f5;
        text-align: center;
        flex: 1;
        min-width: 150px;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """
    Load the trained model (cached for performance).
    """
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    
    # Check if model exists
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.h5')]
    
    if not model_files:
        return None, None
    
    # Use the latest model or allow user selection
    model_path = os.path.join(model_dir, sorted(model_files)[-1])
    
    try:
        predictor = PlantDiseasePredictior(model_path)
        
        # Try to load class names from results
        results_files = [f for f in os.listdir(model_dir) if f.endswith('_results.json')]
        if results_files:
            results_path = os.path.join(model_dir, results_files[-1])
            with open(results_path, 'r') as f:
                results = json.load(f)
                predictor.class_names = results.get('class_names', [])
                predictor.num_classes = len(predictor.class_names)
        
        return predictor, model_path
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None


def main():
    """
    Main Streamlit application.
    """
    
    # Header
    st.markdown("""
        <div class="header">
        <h1>🌿 Plant Disease Detection System</h1>
        <p>AI-Powered Leaf Disease Classifier</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Instructions")
        st.write("""
        1. **Upload an Image**: Choose a leaf image (JPG, PNG, etc.)
        2. **View Results**: Get instant disease prediction
        3. **Review Advice**: See disease information and solutions
        4. **Save Results**: Download prediction results if needed
        """)
        
        st.divider()
        
        st.header("ℹ️ About")
        st.write("""
        This application uses deep learning to classify plant diseases
        from leaf images. It can detect various plant diseases with
        high accuracy.
        
        **Model Architecture**: Transfer Learning with Pre-trained Networks
        **Input Size**: 224×224 pixels
        **Output**: Disease classification with confidence score
        """)
        
        st.divider()
        
        # Model information
        predictor, model_path = load_model()
        
        if predictor and predictor.class_names:
            st.subheader("🤖 Model Information")
            st.write(f"**Classes Detected**: {len(predictor.class_names)}")
            with st.expander("Disease Classes"):
                for i, cls in enumerate(predictor.class_names, 1):
                    st.write(f"{i}. {cls}")
    
    # Load model
    predictor, model_path = load_model()
    
    if predictor is None:
        st.error("""
        ⚠️ No trained model found!
        
        Please train a model first using:
        ```
        python src/train.py --data_dir path/to/data --model MobileNetV2
        ```
        
        The model should be saved in the `models/` directory.
        """)
        return
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a leaf image",
            type=['jpg', 'jpeg', 'png', 'bmp', 'gif'],
            help="Upload a clear image of a plant leaf"
        )
    
    with col2:
        st.subheader("📊 Prediction Results")
        results_placeholder = st.empty()
    
    # Process uploaded image
    if uploaded_file is not None:
        # Save temporary image
        temp_path = "temp_image.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display uploaded image
        st.divider()
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Input Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
        
        with col2:
            st.subheader("Prediction Results")
            
            try:
                # Make prediction
                with st.spinner("🔄 Analyzing image..."):
                    result = predictor.predict_single(temp_path)
                
                # Extract results
                disease = result['predicted_disease']
                confidence = result['confidence']
                top_3 = result['top_3_predictions']
                
                # Display main prediction
                st.markdown(f"""
                    <div class="result-box">
                    <div class="disease-name">{disease}</div>
                    <div class="confidence-score">Confidence: {confidence:.2%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display confidence bar
                st.progress(confidence)
                
                # Display top 3 predictions
                st.subheader("Top Predictions")
                cols = st.columns(3)
                for i, pred in enumerate(top_3):
                    with cols[i]:
                        st.metric(
                            label=pred['class'],
                            value=f"{pred['confidence']:.2%}"
                        )
            
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                return
        
        # Display disease information
        st.divider()
        st.subheader("📖 Disease Information & Recommendations")
        
        # Get disease info
        disease_info = get_disease_info(disease)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Description")
            st.markdown(f"""
                <div class="info-box">
                {disease_info.get('description', 'Information not available')}
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("### Symptoms")
            symptoms = disease_info.get('symptoms', 'Not available')
            st.markdown(f"""
                <div class="info-box">
                {symptoms}
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Recommended Solutions")
            solutions = disease_info.get('solutions', [])
            
            solution_html = '<div class="solution-box">'
            for solution in solutions:
                solution_html += f'<li style="margin: 0.5rem 0;">{solution}</li>'
            solution_html += '</div>'
            
            st.markdown(solution_html, unsafe_allow_html=True)
        
        # Additional actions
        st.divider()
        st.subheader("💾 Export Results")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("📥 Download Prediction Results"):
                # Prepare results for download
                results_text = f"""
Plant Disease Detection Results
================================

Image File: {uploaded_file.name}
Predicted Disease: {disease}
Confidence Score: {confidence:.2%}

Top 3 Predictions:
"""
                for pred in top_3:
                    results_text += f"\n- {pred['class']}: {pred['confidence']:.2%}"
                
                results_text += f"\n\nDisease Information:\n"
                results_text += f"Description: {disease_info.get('description', 'N/A')}\n"
                results_text += f"Symptoms: {disease_info.get('symptoms', 'N/A')}\n"
                results_text += f"\nRecommended Solutions:\n"
                for sol in disease_info.get('solutions', []):
                    results_text += f"- {sol}\n"
                
                st.download_button(
                    label="Download as TXT",
                    data=results_text,
                    file_name=f"disease_prediction_{uploaded_file.name.split('.')[0]}.txt",
                    mime="text/plain"
                )
        
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    else:
        st.info("""
        👆 Upload a plant leaf image to detect diseases.
        
        **Supported Formats**: JPG, PNG, BMP, GIF
        **Recommended**: Clear images with good lighting and focus
        """)
    
    # Footer
    st.markdown("""
        <div class="footer">
        <p>🤖 AI-Powered Plant Disease Detector</p>
        <p>Built with TensorFlow, Streamlit, and Transfer Learning</p>
        <p>© 2024 Plant Disease Detection System. All rights reserved.</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
