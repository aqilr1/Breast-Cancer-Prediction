import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import sys
import os

# Add the parent directory of src to the system path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import get_clean_data # Import from new utils file

# Get the directory of the current script (app/main.py)
app_script_dir = os.path.dirname(os.path.abspath(__file__))

def add_sidebar():
    # ... (rest of the add_sidebar function remains the same)
    st.sidebar.header("Cell Nuclei Measurements Input")
    st.sidebar.markdown("""
        Adjust the sliders below to explore how different cell measurements
        impact the breast cancer prediction.
    """)

    data = get_clean_data() # This will now use the robust path from src/utils.py

    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    input_dict = {}

    for label,key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label=label,
            min_value=float(data[key].min()),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )
    return input_dict


def get_scaled_values_for_radar(input_dict):
    # ... (rest of the get_scaled_values_for_radar function remains the same)
    data = get_clean_data()

    X = data.drop(['diagnosis'], axis=1)

    scaled_dict = {}
    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        if (max_val - min_val) != 0:
            scaled_value = (value - min_val) / (max_val - min_val)
        else:
            scaled_value = 0.5
        scaled_dict[key] = scaled_value
    return scaled_dict

def get_radar_chart(input_data):
    # ... (rest of the get_radar_chart function remains the same)
    input_data_scaled_for_radar = get_scaled_values_for_radar(input_data)

    categories = ['Radius', 'Texture', 'Perimeter', 'Area',
                  'Smoothness', 'Compactness',
                  'Concavity', 'Concave Points',
                  'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data_scaled_for_radar['radius_mean'], input_data_scaled_for_radar['texture_mean'], input_data_scaled_for_radar['perimeter_mean'],
            input_data_scaled_for_radar['area_mean'], input_data_scaled_for_radar['smoothness_mean'], input_data_scaled_for_radar['compactness_mean'],
            input_data_scaled_for_radar['concavity_mean'], input_data_scaled_for_radar['concave points_mean'], input_data_scaled_for_radar['symmetry_mean'],
            input_data_scaled_for_radar['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value',
        line_color='blue'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data_scaled_for_radar['radius_se'], input_data_scaled_for_radar['texture_se'], input_data_scaled_for_radar['perimeter_se'], input_data_scaled_for_radar['area_se'],
            input_data_scaled_for_radar['smoothness_se'], input_data_scaled_for_radar['compactness_se'], input_data_scaled_for_radar['concavity_se'],
            input_data_scaled_for_radar['concave points_se'], input_data_scaled_for_radar['symmetry_se'], input_data_scaled_for_radar['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error',
        line_color='green'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data_scaled_for_radar['radius_worst'], input_data_scaled_for_radar['texture_worst'], input_data_scaled_for_radar['perimeter_worst'],
            input_data_scaled_for_radar['area_worst'], input_data_scaled_for_radar['smoothness_worst'], input_data_scaled_for_radar['compactness_worst'],
            input_data_scaled_for_radar['concavity_worst'], input_data_scaled_for_radar['concave points_worst'], input_data_scaled_for_radar['symmetry_worst'],
            input_data_scaled_for_radar['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value',
        line_color='red'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title_text="Cell Feature Comparison (Scaled for Visualization)",
        title_font_size=16
    )

    return fig

def add_predictions(input_data):
    """
    Loads the trained model and scaler, makes a prediction,
    and displays the result in the Streamlit app.
    """
    try:
        # Construct absolute paths for model and scaler
        # From app/main.py, go up one level to Project_Root, then into 'model'
        model_path = os.path.join(app_script_dir, '..', 'model', 'model.pkl')
        scaler_path = os.path.join(app_script_dir, '..', 'model', 'scaler.pkl')

        model = pickle.load(open(model_path, "rb"))
        scaler = pickle.load(open(scaler_path, "rb"))
    except FileNotFoundError:
        st.error("Error: Model or scaler files not found. Please ensure 'model.pkl' and 'scaler.pkl' are in the 'model' directory. Rerun `python model/main.py` first.")
        return
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the model or scaler: {e}")
        return

    input_array = np.array(list(input_data.values())).reshape(1,-1)
    input_array_scaled = scaler.transform(input_array)
    prediction = model.predict(input_array_scaled)
    prediction_proba = model.predict_proba(input_array_scaled)[0]

    st.subheader("Breast Mass Prediction")
    st.write("Based on the provided measurements, the cell mass is predicted to be:")

    if prediction[0] == 0:
        st.markdown("<span class='diagnosis benign'>Benign (Non-cancerous)</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='diagnosis malicious'>Malignant (Cancerous)</span>", unsafe_allow_html=True)

    st.write(f"Probability of being Benign: **{prediction_proba[0]*100:.2f}%**")
    st.write(f"Probability of being Malignant: **{prediction_proba[1]*100:.2f}%**")

    st.markdown("""
        <div class='disclaimer'>
            <p><strong>Disclaimer:</strong> This application utilizes a machine learning model for predictive purposes and is intended to assist, not replace, professional medical diagnosis. Always consult with a qualified healthcare provider for accurate diagnosis and treatment plans.</p>
        </div>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title = "Breast Cancer Prognosis Tool",
        page_icon = ":ribbon:",
        layout = 'wide',
        initial_sidebar_state = 'expanded',
    )

    try:
        # Load custom CSS for styling
        css_file_path = os.path.join(app_script_dir, '..', 'assets', 'style.css')
        with open(css_file_path) as f:
            st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Custom CSS file not found. App will use default Streamlit styling.")
    except Exception as e:
        st.error(f"An error occurred while loading custom CSS: {e}")


    input_data = add_sidebar()

    with st.container():
        st.title("Interactive Breast Cancer Prognosis Tool")
        st.markdown("""
            This interactive tool helps in predicting whether a breast mass is benign or malignant
            based on cytological measurements. Utilize the sliders in the sidebar to input various
            features of the cell nuclei from a tissue sample.
            The predictions are generated by a machine learning model trained on a comprehensive dataset.
        """)

        st.subheader("Model Information")
        st.markdown("""
            **Model Used:** Random Forest Classifier (with hyperparameter tuning)<br>
            This model was chosen for its robustness and good performance in classification tasks.
            It was trained on the Wisconsin Breast Cancer Diagnostic dataset.
        """, unsafe_allow_html=True)


    col1, col2 = st.columns([4,1])

    with col1:
        st.subheader("Feature Visualization")
        st.markdown("The radar chart below visualizes the scaled input values, allowing for a quick comparison across different feature categories.")
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart, use_container_width=True)
    with col2:
        add_predictions(input_data)

if __name__=='__main__':
    main()