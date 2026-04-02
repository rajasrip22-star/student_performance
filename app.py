import streamlit as st
import pickle

model = pickle.load(open('model.pkl', 'rb'))

st.title("Disease Prediction System")

fever = st.selectbox("Fever", [0,1])
cough = st.selectbox("Cough", [0,1])
headache = st.selectbox("Headache", [0,1])
fatigue = st.selectbox("Fatigue", [0,1])
vomiting = st.selectbox("Vomiting", [0,1])
cold = st.selectbox("Cold", [0,1])

if st.button("Predict"):
    result = model.predict([[fever, cough, headache, fatigue, vomiting, cold]])
    st.success(f"Predicted Disease: {result[0]}")