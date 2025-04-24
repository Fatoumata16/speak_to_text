import os
import streamlit as st
import numpy as np
import torch
from gtts import gTTS
from io import BytesIO
import base64
# Workaround: évite l'inspection de torch.classes par Streamlit
torch.classes.__path__ = []
import librosa
import tempfile
from transformers import AutoConfig, AutoModelForSeq2SeqLM, NllbTokenizer

# === Imports supplémentaires (commentés) ===
import wave, io, joblib, pyttsx3, json, ast
from streamlit_TTS import text_to_audio, auto_play
from gtts.lang import tts_langs
from djelia import Djelia, DjeliaError
from maliba_ai import ASR
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
from pydub import AudioSegment

# === Transformers pour Whisper ===
from transformers import WhisperProcessor, WhisperForConditionalGeneration



def generate_base64_audio(text, lang="fr"):
    """
    Génère un audio en base64 avec GTTS (Google Text-to-Speech)
    """
    tts = gTTS(text=text, lang=lang)  # Générer l'audio avec GTTS
    fp = BytesIO()  # Fichier audio en mémoire
    tts.write_to_fp(fp)  # Sauvegarder l'audio dans le fichier mémoire
    fp.seek(0)  # Retourne au début du fichier
    return base64.b64encode(fp.read()).decode()  # Retourne le texte audio encodé en base64s

# === 1. Chargement du modèle Whisper ===
@st.cache_resource
def load_whisper_model(model_path: str):
    processor = WhisperProcessor.from_pretrained(model_path, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
    if hasattr(model.config, 'forced_decoder_ids'):
        model.config.forced_decoder_ids = None
    if hasattr(model, 'generation_config') and hasattr(model.generation_config, 'forced_decoder_ids'):
        model.generation_config.forced_decoder_ids = None
    return processor, model

# processor, model = load_whisper_model("mon_model1_whisper")
whisper_path = os.path.join(os.path.dirname(__file__), "modele_whisper_transcription")
processor, model = load_whisper_model(whisper_path)


# === 2. Fonction de transcription Whisper ===
def transcribe_with_whisper(audio_bytes: bytes) -> (str, str):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        wav_path = tmp.name
    waveform, sr = librosa.load(wav_path, sr=16000)
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(torch.float32)
    gen_kwargs = {"input_features": input_features, "max_new_tokens": 200}
    if hasattr(inputs, 'attention_mask'):
        gen_kwargs['attention_mask'] = inputs.attention_mask
    with torch.inference_mode():
        generated_ids = model.generate(**gen_kwargs)
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription, wav_path

# === 3. Chargement du modèle local de traduction ===
@st.cache_resource
def load_translation_model(model_dir: str):
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    tokenizer_local = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", local_files_only=True)
    model_local = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir,
        config=config,
        local_files_only=True
    )
    return tokenizer_local, model_local

# tokenizer_local, model_local = load_translation_model("premier_modele")
nllb_path = os.path.join(os.path.dirname(__file__), "modele_nllb_traduction")
tokenizer_local, model_local = load_translation_model(nllb_path)

def translate_local(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    inputs = tokenizer_local(text, return_tensors="pt", padding=True)
    forced_bos_token_id = tokenizer_local.lang_code_to_id["fra_Latn"]
    with torch.inference_mode():
        outputs = model_local.generate(
            **inputs,
            max_length=200,
            forced_bos_token_id=forced_bos_token_id
        )
    return tokenizer_local.batch_decode(outputs, skip_special_tokens=True)[0]


# === 4. Initialisation DJELIA ===
client = Djelia(api_key="f56e7e51-8dcb-4af9-bf07-a58ec23bc72c")

# === 5. Fonctions DJELIA ===
def translate_djelia(text: str) -> str:
    try:
        result = client.translate(text=text, source="bam", target="fr")
        return result.get("text", "") if isinstance(result, dict) else str(result)
    except Exception as e:
        st.error(f"Erreur de traduction DJELIA: {e}")
        return ""

def transcribe_djelia_file(wav_path: str, with_timestamps: bool=False, to_french: bool=False) -> str:
    """
    Transcrit un fichier .wav avec l'API DJELIA et retourne toujours un str.
    """
    try:
        # Choix de l'appel selon paramètres
        if to_french:
            result = client.transcribe(wav_path, translate_to_french=True)
        elif with_timestamps:
            result = client.transcribe(wav_path, version=2)
        else:
            result = client.transcribe(wav_path)
        # Extraction du texte
        if isinstance(result, dict):
            return result.get("text", "")
        if isinstance(result, list):
            # Liste de segments
            return "\n".join([seg.get("text", "") for seg in result if isinstance(seg, dict)])
        # Autre format
        return str(result)
    except Exception as e:
        st.error(f"Erreur transcription DJELIA: {e}")
        return ""

# === 6. Application Streamlit ===
st.title("🎤 Transcription et traduction Bambara ↔ Français")
if 'state' not in st.session_state:
    st.session_state.state = 'idle'

# ÉTAT 1 : Enregistrement
if st.session_state.state == 'idle':
    st.write("Cliquez pour enregistrer un message :")
    audio_data = st.audio_input("🎙️ Enregistrer")
    if audio_data is not None:
        st.session_state.audio_data = audio_data
        st.session_state.state = 'preview'
        st.rerun()

# ÉTAT 2 : Prévisualisation
elif st.session_state.state == 'preview':
    st.audio(st.session_state.audio_data)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Valider"):
            st.session_state.state = 'transcribing'
            st.rerun()
    with col2:
        if st.button("↩️ Recommencer"):
            st.session_state.audio_data = None
            st.session_state.state = 'idle'
            st.rerun()

# ÉTAT 3 : Transcription & Comparaison
elif st.session_state.state == 'transcribing':
    with st.spinner("Transcription et comparaison en cours…"):
        audio_bytes = st.session_state.audio_data.getvalue()
        whisper_text, wav_path = transcribe_with_whisper(audio_bytes)
        djelia_text = transcribe_djelia_file(wav_path)
        try: os.remove(wav_path)
        except: pass
        trad_djelia_whisper = translate_djelia(whisper_text)
        trad_local_whisper = translate_local(whisper_text)
        trad_djelia_djelia = translate_djelia(djelia_text)
        trad_local_djelia = translate_local(djelia_text)

    # Affichage en 2 colonnes (transcriptions+traductions)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Whisper (ASR) :")
        st.write(whisper_text)
        
        # Écouter la traduction de Whisper en Bambara avec DJELIA
        if st.button("🔊 Écouter Bambara (Whisper)", key="tts_bambara_whisper"):
            try:
                # Utiliser l'API DJELIA pour générer l'audio à partir du texte
                audio_path = client.text_to_speech(text=whisper_text, speaker=1, output_file="tts_whisper_bambara.mp3")
                st.audio(audio_path)  # Lire l'audio
            except Exception as e:
                st.error(f"Erreur TTS DJELIA (Whisper): {e}")

        st.subheader("Traduction DJELIA :")
        st.write(trad_djelia_whisper)
        if st.button("🔊 Écouter (Djelia Whisper)", key="tts_djelia_w"):
            audio_base64 = generate_base64_audio(trad_djelia_whisper)
            st.audio(f"data:audio/mp3;base64,{audio_base64}", format="audio/mp3")
        st.subheader("Traduction Locale :")
        st.write(trad_local_whisper)
        if st.button("🔊 Écouter (Locale Whisper)", key="tts_local_w"):
            audio_base64 = generate_base64_audio(trad_local_whisper)
            st.audio(f"data:audio/mp3;base64,{audio_base64}", format="audio/mp3")
    with col2:
        st.subheader("DJELIA (ASR) :")
        st.write(djelia_text)
        # Écouter la traduction en Bambara avec DJELIA
        if st.button("🔊 Écouter Bambara (DJELIA)", key="tts_bambara_djelia"):
            try:
                # Utiliser l'API DJELIA pour générer l'audio à partir du texte
                audio_path = client.text_to_speech(text=djelia_text, speaker=1, output_file="tts_djelia_bambara.mp3")
                st.audio(audio_path)  # Lire l'audio
            except Exception as e:
                st.error(f"Erreur TTS DJELIA (Djelia): {e}")
        st.subheader("Traduction DJELIA :")
        st.write(trad_djelia_djelia)
        if st.button("🔊 Écouter (Djelia Djelia)", key="tts_djelia_d"):
            audio_base64 = generate_base64_audio(trad_djelia_djelia)
            st.audio(f"data:audio/mp3;base64,{audio_base64}", format="audio/mp3")
        st.subheader("Traduction Locale :")
        st.write(trad_local_djelia)
        if st.button("🔊 Écouter (Locale Djelia)", key="tts_local_d"):
            audio_base64 = generate_base64_audio(trad_local_djelia)
            st.audio(f"data:audio/mp3;base64,{audio_base64}", format="audio/mp3")

    if st.button("🔁 Nouvel enregistrement"):
        st.session_state.audio_data = None
        st.session_state.state = 'idle'
        st.rerun()

# === Commentaires ===
# Pour désactiver fileWatcher: streamlit run app.py --server.fileWatcherType none