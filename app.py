import streamlit as st
import transformers
from transformers import VitsModel, AutoTokenizer
import torch
import numpy as np
import io
import soundfile as sf

# Load model and tokenizer
model = VitsModel.from_pretrained("facebook/mms-tts-eng")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

def generate_speech(text):
  inputs = tokenizer(text, return_tensors="pt")

  with torch.no_grad():
    output = model(**inputs).waveform  # Corrected typo: waveform (not waveformform)

  # Convert the waveform tensor to a NumPy array
  waveform = output.squeeze().cpu().numpy()

  # Convert the waveform to bytes
  audio_bytes_io = io.BytesIO()
  sf.write(audio_bytes_io, waveform, samplerate=22050, format='WAV')
  audio_bytes_io.seek(0)

  return audio_bytes_io

# Streamlit UI
st.title("Text-to-Speech Converter")
st.write("Developed by Safwan Ahmad Saffi")
st.write("Enter text below and click 'Generate Speech' to convert it to audio.")

# Text input
text_input = st.text_area("Text to convert:", "Some example text in the English language")

if st.button("Generate Speech"):
  if text_input:
    st.write("Generating speech...")
    audio_bytes_io = generate_speech(text_input)

    # Display audio in Streamlit
    st.audio(audio_bytes_io, format="audio/wav")
  else:
    st.write("Please enter some text.")