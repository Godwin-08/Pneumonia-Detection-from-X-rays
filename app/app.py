# =========================================================
# IMPORTATIONS
# =========================================================

import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import plotly.graph_objects as go
from PIL import Image
import os


# =========================================================
# CONFIGURATION PAGE
# =========================================================

st.set_page_config(

    page_title="Pneumonia Detection App",
    page_icon="assets/logo/logo.png",
    layout="centered"
)

# =========================================================
# TITRE APPLICATION
# =========================================================

col1, col2 = st.columns([1, 6])

with col1:
    st.image(
        "assets/logo/logo.png",
        width=80
    )

with col2:
    st.title("Pneumonia Detection App")

# =========================================================
# ABOUT PROJECT
# =========================================================

st.markdown(
    "##### AI application for Pneumonia Detection using Deep Learning."
)

# =========================================================
# CHARGEMENT DES MODELES
# =========================================================

# =========================================================
# CHARGEMENT DES MODELES
# =========================================================

@st.cache_resource
def load_models():

    cnn_path = "models/cnn_new.keras"
    densenet_path = "models/densenet_new.keras"

    cnn = tf.keras.models.load_model(
    cnn_path,
    compile=False,
    safe_mode=False
    )

    densenet = tf.keras.models.load_model(
        densenet_path,
    compile=False,
    safe_mode=False
    )

    return cnn, densenet


with st.spinner("Loading AI models..."):

    cnn_model, densenet_model = load_models()

# =========================================================
# ACCURACY DES MODELES
# =========================================================

cnn_accuracy = 84
densenet_accuracy = 87

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("AI Settings")

# =========================================================
# CHOIX MODELE
# =========================================================

selected_model = st.sidebar.selectbox(

    "Choose AI Model",

    [

        "Custom CNN",

        "DenseNet121"

    ]
)

# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.sidebar.markdown("## Model Performance")

performance_model = st.sidebar.selectbox(

    "Choose Performance Model",

    [

        "Custom CNN",

        "DenseNet121"

    ]
)

# =========================================================
# CNN PERFORMANCE
# =========================================================

if performance_model == "Custom CNN":

    st.sidebar.markdown(
        """
        <div style="
            background:linear-gradient(135deg,#1E3A8A,#2563EB);
            padding:18px;
            border-radius:15px;
            margin-bottom:15px;
            box-shadow:0 4px 12px rgba(0,0,0,0.25);
        ">

        <h3 style="
            color:white;
            text-align:center;
            margin-bottom:15px;
        ">
            Custom CNN
        </h3>

        <p style="color:white;">
             Accuracy : <b>84%</b>
        </p>

        <p style="color:white;">
             F1-Score : <b>0.8738</b>
        </p>

        <p style="color:white;">
             Precision : <b>0.84</b>
        </p>

        <p style="color:white;">
             Recall : <b>0.91</b>
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

# =========================================================
# DENSENET PERFORMANCE
# =========================================================

else:

    st.sidebar.markdown(
        """
        <div style="
            background:linear-gradient(135deg,#065F46,#10B981);
            padding:18px;
            border-radius:15px;
            margin-bottom:15px;
            box-shadow:0 4px 12px rgba(0,0,0,0.25);
        ">

        <h3 style="
            color:white;
            text-align:center;
            margin-bottom:15px;
        ">
            DenseNet121
        </h3>

        <p style="color:white;">
             Accuracy : <b>87%</b>
        </p>

        <p style="color:white;">
             F1-Score : <b>0.8907</b>
        </p>

        <p style="color:white;">
             Precision : <b>0.93</b>
        </p>

        <p style="color:white;">
             Recall : <b>0.86</b>
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )
# =========================================================
# VISUALIZATION OPTIONS
# =========================================================

st.sidebar.markdown(
    "## Visualization Options"
)

show_probability = st.sidebar.checkbox(
    "Probability Chart",
    value=True
)

show_comparison = st.sidebar.checkbox(
    "Model Comparison",
    value=True
)

# =========================================================
# SELECTION MODELE
# =========================================================

if selected_model == "Custom CNN":

    model = cnn_model
    current_accuracy = cnn_accuracy

else:

    model = densenet_model
    current_accuracy = densenet_accuracy

# =========================================================
# MODELE ACTIF
# =========================================================

st.info(
    f"Current Model : {selected_model}"
)

# =========================================================
# UPLOAD IMAGE
# =========================================================

uploaded_file = st.file_uploader(

    "Upload Chest X-Ray",

    type=["jpg", "jpeg", "png"]

)

# =========================================================
# AFFICHAGE IMAGE
# =========================================================

if uploaded_file is not None:

    preview_img = Image.open(uploaded_file)

    st.image(
        preview_img,
        caption="Selected Chest X-Ray",
        width=250
    )

# =========================================================
# BOUTON ANALYSE
# =========================================================

analyze = st.button(
    "Analyze Image"
)

# =========================================================
# ANALYSE
# =========================================================

if analyze:

    if uploaded_file is not None:

        with st.spinner(
            "Analyzing X-Ray..."
        ):

            time.sleep(2)

            # =================================================
            # OUVERTURE IMAGE
            # =================================================

            img = Image.open(
                uploaded_file
            ).convert("RGB")

            img = img.resize((224, 224))

            original_img = np.array(img)

            img_array = np.array(img)

            img_array = img_array / 255.0

            img_array = np.expand_dims(
                img_array,
                axis=0
            )

            # =================================================
            # PREDICTION
            # =================================================

            prediction = model.predict(
                img_array,
                verbose=0
            )

            score = prediction[0][0]

            risk_score = score * 100

            if score > 0.5:

                predicted_class = "PNEUMONIA"

            else:

                predicted_class = "NORMAL"



        # =================================================
        # RESULTATS
        # =================================================

        st.subheader(
            "Prediction Result"
        )

        # =================================================
        # RESULTAT TEXTE
        # =================================================

        if predicted_class == "PNEUMONIA":

            st.error(
                "PNEUMONIA DETECTED"
            )

            st.write(
                f"Confidence : {score*100:.2f}%"
            )

        else:

            st.success(
                "NORMAL"
            )

            st.write(
                f"Confidence : {(1-score)*100:.2f}%"
            )

        st.write(
            f"Model Accuracy : {current_accuracy}%"
        )

        # =================================================
        # COULEUR RISQUE
        # =================================================

        if risk_score < 40:

            gauge_color = "green"

        elif risk_score < 70:

            gauge_color = "orange"

        else:

            gauge_color = "red"

        # =================================================
        # GAUGE
        # =================================================

        fig_plotly = go.Figure(

            go.Indicator(

                mode="gauge+number",

                value=risk_score,

                title={

                    'text': "Pneumonia Risk Level",

                    'font': {

                        'size': 28,

                        'color': "#E5E7EB"
                    }
                },

                number={

                    'suffix': "%",

                    'font': {

                        'size': 45,

                        'color': "#F9FAFB"
                    }
                },

                gauge={

                    'axis': {

                        'range': [0, 100],

                        'tickwidth': 1,

                        'tickcolor': "#9CA3AF"
                    },

                    'bar': {

                        'color': "#FFFFFF",

                        'thickness': 0.1
                    },

                    'bgcolor': "#111827",

                    'borderwidth': 1,

                    'bordercolor': "#374151",

                    'steps': [

                        {
                            'range': [0, 40],

                            'color': "#27D614"
                        },

                        {
                            'range': [40, 70],

                            'color': "#F5550B"
                        },

                        {
                            'range': [70, 100],

                            'color': "#E91515"
                        }
                    ],

                    'threshold': {

                        'line': {

                            'color': "#FFFFFF",

                            'width': 6
                        },

                        'thickness': 0.8,

                        'value': risk_score
                    }
                }
            )
        )

        fig_plotly.update_layout(

            paper_bgcolor="#0E1117",

            font={

                'color': "white",

                'family': "Arial"
            },

            height=420
        )

        st.plotly_chart(

            fig_plotly,

            use_container_width=True
        )
        # =================================================
        # PROBABILITY VALUES
        # =================================================

        normal_prob = (1 - score) * 100
        pneumonia_prob = score * 100

        # =================================================
        # PIE CHART
        # =================================================

        st.subheader("Prediction Distribution")

        fig_pie = go.Figure(

            data=[

                go.Pie(

                    labels=["NORMAL", "PNEUMONIA"],

                    values=[normal_prob, pneumonia_prob],

                    hole=0.5
                )

            ]
        )

        fig_pie.update_layout(

            height=500
        )

        st.plotly_chart(

            fig_pie,

            use_container_width=True
        )

        # =================================================
        # PROBABILITY CHART
        # =================================================

        if show_probability:

            st.subheader("Prediction Probabilities")

            fig_bar = go.Figure(data=[

                go.Bar(

                    x=["NORMAL", "PNEUMONIA"],

                    y=[normal_prob, pneumonia_prob],

                    text=[

                        f"{normal_prob:.1f}%",

                        f"{pneumonia_prob:.1f}%"
                    ],

                    textposition="auto",

                    marker=dict(

                        color=[

                            "#3B82F6",

                            "#8DBFFF"

                        ]
                    )
                )

            ])

            fig_bar.update_layout(

                title="AI Prediction Confidence",

                yaxis_title="Probability (%)",

                xaxis_title="Classes",

                paper_bgcolor="#0E1117",

                plot_bgcolor="#111827",

                font=dict(

                    color="white",

                    size=14
                ),

                height=450
            )

            st.plotly_chart(

                fig_bar,

                use_container_width=True
            )
        # =================================================
        # MODEL COMPARISON
        # =================================================

        if show_comparison:

            cnn_pred = cnn_model.predict(
                img_array,
                verbose=0
            )[0][0] * 100

            densenet_pred = densenet_model.predict(
                img_array,
                verbose=0
            )[0][0] * 100

            fig_compare = go.Figure()

            fig_compare.add_trace(

                go.Bar(

                    x=["Custom CNN"],

                    y=[cnn_pred],

                    name="CNN"
                )
            )

            fig_compare.add_trace(

                go.Bar(

                    x=["DenseNet121"],

                    y=[densenet_pred],

                    name="DenseNet121"
                )
            )

            st.subheader("Model Prediction Comparison")
            
            fig_compare.update_layout(


                yaxis_title="Pneumonia Risk (%)",

                height=500
            )

            st.plotly_chart(

                fig_compare,

                use_container_width=True
            )
            # =================================================
            # RADAR CHART
            # =================================================

            st.subheader("🕸️ AI Metrics Radar")

            fig_radar = go.Figure()

            fig_radar.add_trace(go.Scatterpolar(

                r=[84, 84, 91, 87],

                theta=["Accuracy", "Precision", "Recall", "F1-Score"],

                fill='toself',

                name='Custom CNN'
            ))

            fig_radar.add_trace(go.Scatterpolar(

                r=[87, 93, 86, 89],

                theta=["Accuracy", "Precision", "Recall", "F1-Score"],

                fill='toself',

                name='DenseNet121'
            ))

            fig_radar.update_layout(

                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0,100]
                    )
                ),

                height=500
            )

            st.plotly_chart(
                fig_radar,
                use_container_width=True
            )

            st.markdown(
                """
                <p style="
                    color:red;
                    font-size:14px;
                    text-align:center;
                    margin-top:15px;
                ">
                    This AI tool is for educational purposes and does not replace professional medical diagnosis.
                </p>
                """,
                unsafe_allow_html=True
)

    else:

        st.warning(
            "Please upload an image first."
        )