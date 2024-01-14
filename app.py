import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import torch
import joblib

# load the saved model and tokenizer
model_folder_path = '/home/linux/NLP/model'  
model = BertForSequenceClassification.from_pretrained(model_folder_path)
tokenizer = BertTokenizer.from_pretrained(model_folder_path)

# load label encoder
label_encoder_path ='/home/linux/NLP/model/label_encoder.pkl'  
label_encoder = joblib.load(label_encoder_path)

# function to make predictions
def predict_problem(input_text):
    # tokenize and prepare tensors for the input text
    inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)

    # make the prediction
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()

    # convert prediction to original label
    predicted_label = label_encoder.inverse_transform([prediction])[0]

    return predicted_label

# streamlit app
st.title('Disease Prediction App')

# user input for symptoms
symptoms = st.text_area('Enter symptoms (comma-separated):')

# make predictions when the user clicks the button
if st.button('Predict'):
    if symptoms:
        predicted_problem = predict_problem(symptoms)
        st.success(f'You may suffer from: {predicted_problem}')
        st.success(f'Please Consult to the doctor it is not the 100% accurate.')
    else:
        st.warning('Please enter symptoms.')
