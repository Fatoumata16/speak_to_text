import streamlit as st
import numpy as np
import wave
import io
import joblib
import tempfile
import pyttsx3
from streamlit_TTS import text_to_audio, auto_play
from gtts.lang import tts_langs
from djelia import Djelia, DjeliaError
from maliba_ai import ASR
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
import os
import torch
import whisper
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
from pydub import AudioSegment
import io

# Initialisation du processeur et du modèle Whisper
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.eval()
# Initialisez le processeur et le modèle Whisper
# processor = WhisperProcessor.from_pretrained("openai/whisper-large")
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")
# Vérifiez si le pad_token est défini
if processor.tokenizer.pad_token is None:
    # Définissez le pad_token comme étant égal au eos_token
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
# if processor.tokenizer.pad_token_id is None:
#     processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
#     model.config.pad_token_id = processor.tokenizer.pad_token_id


def transcribe_with_whisper(uploaded_file):
    # Créez un fichier temporaire pour enregistrer l'audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    # Chargez l'audio
    waveform, sample_rate = torchaudio.load(tmp_file_path)

    # Rééchantillonnez à 16 kHz si nécessaire
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
        sample_rate = 16000

    # Préparez l'entrée pour le modèle Whisper
    inputs = processor(waveform.squeeze(), sampling_rate=sample_rate, return_tensors="pt")

    # Effectuez la transcription
    with torch.no_grad():
        generated_ids = model.generate(inputs.input_features)
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return transcription
from pydub import AudioSegment
import io, os, tempfile, torch, torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# … initialisation de processor et model …

# def transcribe_with_whisper(uploaded_file):
#     # 1) Revenir au début du buffer
#     uploaded_file.seek(0)

#     # 2) Lire tous les octets
#     audio_bytes = uploaded_file.read()

#     # 3) Déterminer le format (ex: "audio/webm;codecs=opus")
#     content_type = uploaded_file.type  # ex: "audio/webm;codecs=opus"
#     fmt = content_type.split(';')[0].split('/')[-1]  # -> "webm"

#     # 4) Charger avec pydub en précisant le format
#     audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)

#     # 5) Normaliser à 16 kHz mono PCM si besoin
#     audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)

#     # 6) Exporter en WAV dans un fichier temporaire
#     with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
#         audio.export(tmp_wav.name, format="wav")
#         tmp_path = tmp_wav.name

#     # 7) Charger avec torchaudio (Whisper attend un Tensor)
#     waveform, sr = torchaudio.load(tmp_path)
#     os.remove(tmp_path)

#     # 8) Transcription Whisper
#     inputs = processor(waveform.squeeze(), sampling_rate=sr, return_tensors="pt")
#     with torch.no_grad():
#         ids = model.generate(inputs.input_features)
#         return processor.batch_decode(ids, skip_special_tokens=True)[0]


@st.cache_resource
def load_translation_pipeline(model_dir: str):
    # 1) Config
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    # 2) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    # 3) Modèle
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir,
        config=config,
        local_files_only=True,
        trust_remote_code=True  # si ton modèle a du code perso
    )
    return tokenizer, model

# Chargement du modele de traduction
MODEL_DIR = "./premier_modele"
tokenizer, translator_model = load_translation_pipeline(MODEL_DIR)


# Initialisation du client DJELIA
client = Djelia(api_key="f56e7e51-8dcb-4af9-bf07-a58ec23bc72c")

# Initialisation des variables de session
if 'state' not in st.session_state:
    st.session_state['state'] = "idle"
if 'audio_data' not in st.session_state:
    st.session_state['audio_data'] = None
if 'transcribed_text' not in st.session_state:
    st.session_state['transcribed_text'] = ""
if 'translated_text' not in st.session_state:
    st.session_state['translated_text'] = ""

st.title("🎤 Traduction bambara - français 🎤")

def transcribe_audio(audio_buffer):
    """Appelle un modèle de transcription local (à remplacer)"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        tmp_wav.write(audio_buffer.read())
        tmp_wav_path = tmp_wav.name
     
    transcriber = ASR(model_id = "sudoping01/bambara-asr-v4-ic")
    #model = joblib.load("mon_modele_transcription.joblib")
    #transcription = model.transcribe(tmp_wav_path)
    result = transcriber.transcribe_audio(tmp_wav_path)
    print(result)
    return result

def transcribe_with_djelia(audio_buffer):
    """Transcription via l'API DJELIA avec options et gestion d'erreurs"""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_wav.write(audio_buffer.read())
            tmp_path = tmp_wav.name

        # === OPTIONS POSSIBLES ===
        # 1. Transcription simple
        result = client.transcribe(tmp_path)

        # 2. Transcription avec timestamps
        # result = client.transcribe(tmp_path, version=2)

        # 3. Transcription avec traduction automatique vers le français
        # result = client.transcribe(tmp_path, translate_to_french=True)

        # === Extraction du texte selon le format retourné ===
        if isinstance(result, list):
            transcription = "".join([seg['text'] for seg in result])
        elif isinstance(result, dict) and 'text' in result:
            transcription = result['text']
        else:
            transcription = result

        return transcription

    except DjeliaError as e:
        st.error(f"Erreur avec l'API Djelia : {e}")
        return None

    except Exception as e:
        st.error(f"Erreur inattendue : {e}")
        return None

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

def translate_text(text: str) -> str:
    """
    Traduit du bambara vers le français en utilisant le modèle HF local.
    """
    # 1) Tokenize
    inputs = tokenizer(text, return_tensors="pt")
    # 2) Génération
    outputs = translator_model.generate(**inputs)
    # 3) Décodage
    translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return translated


# === ÉTAT 1 : Enregistrement ===
if st.session_state['state'] == "idle":
    st.write("Appuyez pour enregistrer un message vocal :")
    audio_data = st.audio_input("🎙️ Démarrer l'enregistrement")
    if audio_data is not None:
        st.session_state['audio_data'] = audio_data
        st.session_state['state'] = "done"
        st.rerun()

# === ÉTAT 2 : Prévisualisation ===
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

    #transcription = transcribe_with_djelia(st.session_state['audio_data'])
    #transcription = "A' ni sɔgɔma"
    transcription = transcribe_with_whisper(st.session_state['audio_data'])

    if transcription:
        translation = translate_text(transcription)
        # translation = translate_text(transcription)
        placeholder.empty()

        st.session_state['transcribed_text'] = transcription
        st.session_state['translated_text'] = translation

        st.success("Transcription et traduction terminées !")

        # Affichage côte à côte
        text_col, trans_col = st.columns(2)
        with text_col:
            st.markdown("**Texte transcrit :**")
            st.write(transcription)
        with trans_col:
            st.markdown("**Traduction :**")
            st.write(translation)

        # Boutons
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        with btn_col1:
            if st.button("🔊 Écouter en Français"):
                try:
                    audio = text_to_audio(translation, language='fr')
                    auto_play(audio)
                except Exception:
                    # fallback local
                    engine = pyttsx3.init()
                    tmpfile = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                    engine.save_to_file(translation, tmpfile.name)
                    engine.runAndWait()
                    st.audio(tmpfile.name, format="audio/mp3")

        with btn_col2:
            if st.button("🔊 Écouter en Bambara (DJELIA)"):
                # audio_bytes = client.text_to_speech(transcription, speaker=2)
                # st.audio(audio_bytes, format="audio/mp3")
                audio = text_to_audio(translation, language='fr')
                auto_play(audio)
        with btn_col3:
            if st.button("🔁 Nouvel enregistrement"):
                st.session_state['audio_data'] = None
                st.session_state['transcribed_text'] = ""
                st.session_state['translated_text'] = ""
                st.session_state['state'] = "idle"
                st.rerun()
    else:
        placeholder.empty()
        st.warning("⚠️ La transcription a échoué. Veuillez réessayer.")
        if st.button("🔁 Recommencer"):
            st.session_state['audio_data'] = None
            st.session_state['state'] = "idle"
            st.rerun()
