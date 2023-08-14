import streamlit as st
import numpy as np
import pandas as pd
#from sklearn.linear_model import LogisticRegression
import joblib


# Titre de l'application
st.title("Prédiction de Risque de Décès par COVID-19")

# Description de l'application
st.write("Saisissez les informations du patient pour prédire s'il présente un risque élevé de décès par COVID-19.")

# Saisie des informations du patient
st.sidebar.header("Informations du Patient")
USMER=st.sidebar.slider('USMER', 1, 2)
MEDICAL_UNIT=st.sidebar.slider('MEDICAL_UNIT', 1, 13)
PATIENT_TYPE=st.sidebar.slider('PATIENT_TYPE', 1, 2)
PNEUMONIA=st.sidebar.slider('PNEUMONIA', 1, 2)
AGE = st.sidebar.slider("AGE", 0, 100, 40)
DIABETES=st.sidebar.slider('DIABETES', 1, 2)
HIPERTENSION=st.sidebar.slider('HIPERTENSION', 1, 2)
RENAL_CHRONIC=st.sidebar.slider('RENAL_CHRONIC', 1, 2)
CLASIFFICATION_FINAL=st.sidebar.slider('CLASIFFICATION_FINAL', 1, 7)

# Bouton de prédiction
if st.sidebar.button("Prédire le Risque"):
    # chargement du model 
    model=joblib.load(filename="final_model_covid.joblib")

    # Prédire le risque de décès (2: faible risque, 1: risque élevé)
    patient_data = np.array([[USMER, MEDICAL_UNIT, PATIENT_TYPE, PNEUMONIA, AGE, DIABETES,
       HIPERTENSION, RENAL_CHRONIC, CLASIFFICATION_FINAL]])
    prediction = model.predict(patient_data)

    # Afficher la prédiction
    if prediction[0] == 2:
        st.write("Le patient présente un faible risque de décès par COVID-19.")
    else:
        st.write("Le patient présente un risque élevé de décès par COVID-19.")

