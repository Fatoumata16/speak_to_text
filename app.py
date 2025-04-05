import streamlit as st
import numpy as np
import wave
import io
import joblib
import tempfile
import pyttsx3
from streamlit_TTS import text_to_audio, auto_play
from gtts.lang import tts_langs

# Initialisation des variables de session
if 'state' not in st.session_state:
    st.session_state['state'] = "idle"         # états : "idle", "done", "transcription"
if 'audio_data' not in st.session_state:
    st.session_state['audio_data'] = None
if 'transcribed_text' not in st.session_state:
    st.session_state['transcribed_text'] = ""

st.title("🎤 Traduction bambara - français 🎤")

def transcribe_audio(audio_buffer):
    """Appelle un modèle de transcription (à remplacer)"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        tmp_wav.write(audio_buffer.read())
        tmp_wav_path = tmp_wav.name
    model = joblib.load("mon_modele_transcription.joblib") 
    transcription = model.transcribe(tmp_wav_path)  # Remplacer par votre modèle réel
    return transcription

# === ÉTAT 1 : Enregistrement via st.audio_input ===
if st.session_state['state'] == "idle":
    st.write("Appuyez  pour enregistrer un message vocal :")
    audio_data = st.audio_input("🎙️ Démarrer l'enregistrement")

    if audio_data is not None:
        st.session_state['audio_data'] = audio_data
        st.session_state['state'] = "done"
        st.rerun()

# === ÉTAT 2 : Prévisualisation de l'enregistrement ===
if st.session_state['state'] == "done" and st.session_state['audio_data']:
    st.audio(st.session_state['audio_data'], format="audio/wav")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Confirmer"):
            st.session_state['state'] = "transcription"
            st.rerun()
    with col2:
        if st.button("↩️ Recommencer"):
            st.session_state['audio_data'] = None
            st.session_state['state'] = "idle"
            st.rerun()

# === ÉTAT 3 : Transcription ===
if st.session_state['state'] == "transcription":
    placeholder = st.empty()
    placeholder.info("Transcription et traduction en cours...")
    # Simulation de transcription (remplacer par l'appel à transcribe_audio)
    #transcription = transcribe_audio(audio_data)
    transcription = "Test de transcription. Un chasseur sachant chasser sans son chien est un bon chasseur. Les chaussettes de l'archiduchesse sont sèches et archi-sèches."
    placeholder.empty()
    
    st.session_state['transcribed_text'] = transcription
    st.success("Transcription et traduction terminées !")
    st.markdown(f"**Texte traduit :**\n\n> {transcription}")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔊 Écouter le texte"):
            langs=tts_langs().keys()
            audio = text_to_audio(transcription, language='fr')
            auto_play(audio)
    with col2:
        if st.button("🔁 Nouvel enregistrement"):
            st.session_state['audio_data'] = None
            st.session_state['transcribed_text'] = ""
            st.session_state['state'] = "idle"
            st.rerun()
