import streamlit as st
import numpy as np
import wave
import io
import joblib
import tempfile
import pyttsx3
from streamlit_TTS import text_to_audio, auto_play
from gtts.lang import tts_langs
from djelia import Djelia

# Importation et initialisation du client DJELIA
client = Djelia(api_key="f56e7e51-8dcb-4af9-bf07-a58ec23bc72c")

# Initialisation des variables de session
if 'state' not in st.session_state:
    st.session_state['state'] = "idle"         # états : "idle", "done", "transcription"
if 'audio_data' not in st.session_state:
    st.session_state['audio_data'] = None
if 'transcribed_text' not in st.session_state:
    st.session_state['transcribed_text'] = ""
if 'translated_text' not in st.session_state:
    st.session_state['translated_text'] = ""

st.title("🎤 Traduction bambara - français 🎤")

def transcribe_audio(audio_buffer):
    """Appelle un modèle de transcription (à remplacer)"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        tmp_wav.write(audio_buffer.read())
        tmp_wav_path = tmp_wav.name
    model = joblib.load("mon_modele_transcription.joblib")
    transcription = model.transcribe(tmp_wav_path)  # Remplacer par votre modèle réel
    return transcription


def translate_text(text):
    """Appelle un modèle de traduction (à remplacer)"""
    # Exemple de modèle de traduction chargé avec joblib
    # model = joblib.load("mon_modele_traduction.joblib")
    # traduction = model.translate(text)
    # return traduction
    
    # Version exemple en attendant le modèle
    return "Exemple de traduction en français."

# === ÉTAT 1 : Enregistrement via st.audio_input ===
if st.session_state['state'] == "idle":
    st.write("Appuyez pour enregistrer un message vocal :")
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

# === ÉTAT 3 : Transcription et traduction ===
if st.session_state['state'] == "transcription":
    placeholder = st.empty()
    placeholder.info("Transcription et traduction en cours...")
    # Simulation de transcription (remplacer par transcribe_audio si votre modèle est prêt)
    #transcription = transcribe_audio(audio_data)
    transcription = "A taara Kucala kunu a nana bi sɔgɔma"
    # Appel à la fonction de traduction
    #translation = translate_text(transcription)
    translation = "Ceci est un exemple de traduction en attendant le modele."
    placeholder.empty()

    # Sauvegarde dans le state
    st.session_state['transcribed_text'] = transcription
    st.session_state['translated_text'] = translation

    st.success("Transcription et traduction terminées !")

    # Affichage côte à côte de la transcription et de la traduction
    text_col, trans_col = st.columns(2)
    with text_col:
        st.markdown("**Texte transcrit :**")
        st.write(transcription)
    with trans_col:
        st.markdown("**Traduction :**")
        st.write(translation)

    # Boutons pour écouter et recommencer
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    with btn_col1:
        if st.button("🔊 Écouter en Français"):
            audio = text_to_audio(translation, language='fr')
            auto_play(audio)
    with btn_col2:
        if st.button("🔊 Écouter en Bambara (DJELIA)"):
            audio_bytes = client.text_to_speech(transcription, speaker=1)
            st.audio(audio_bytes, format="audio/mp3")
    with btn_col3:
        if st.button("🔁 Nouvel enregistrement"):
            st.session_state['audio_data'] = None
            st.session_state['transcribed_text'] = ""
            st.session_state['translated_text'] = ""
            st.session_state['state'] = "idle"
            st.rerun()
