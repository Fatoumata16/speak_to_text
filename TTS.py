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
import librosa
import json

import ast
from typing import Any

import streamlit as st
from transformers import WhisperProcessor, WhisperForConditionalGeneration

@st.cache_resource
def load_whisper_model(model_path):
    processor = WhisperProcessor.from_pretrained(model_path, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
    return processor, model

processor, model = load_whisper_model("mon_model1_whisper")

def generate_base64_audio(text, lang):
    from gtts import gTTS
    from io import BytesIO
    import base64
    
    tts = gTTS(text=text, lang=lang)
    fp = BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return base64.b64encode(fp.read()).decode()

def extract_text_from_transcription(raw: str) -> str:
    """
    Extrait le champ 'text' du premier segment d'une chaîne de transcription
    au format Python list-of-dicts.

    :param raw: Chaîne brute, ex. "[{'text': 'Ne sɔgɔma.', 'start': 0.0, 'end': 1.53}]"
    :return: Contenu textuel ou chaîne vide si échec.
    """
    try:
        # 1) Convertit la chaîne en liste Python
        segments = ast.literal_eval(raw)  # sûr pour Python literals :contentReference[oaicite:4]{index=4}
        # 2) Vérifie le format et extrait
        if isinstance(segments, list) and segments:
            first = segments[0]
            # Renvoie la clé 'text' si présente
            return first.get("text", "")
    except (ValueError, SyntaxError):
        # En cas de contenu malformé, on retourne vide
        return ""
    # Sinon
    return ""

# # Avant de charger le modèle
# torch.set_num_threads(1)
# torch._C._jit_set_profiling_executor(False)
# torch._C._jit_set_profiling_mode(False)

# Initialisation du processeur et du modèle Whisper
# processor = WhisperProcessor.from_pretrained("openai/whisper-small")
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
# processor = WhisperProcessor.from_pretrained("/Users/moctar/Desktop/Projet_Master_2/speak_to_text_bambara/mon_model1_whisper")
# model = WhisperForConditionalGeneration.from_pretrained("/Users/moctar/Desktop/Projet_Master_2/speak_to_text_bambara/mon_model1_whisper")
# model.eval()
# from transformers import PreTrainedTokenizer

# tokenizer: PreTrainedTokenizer = processor.tokenizer

# if processor.tokenizer.pad_token is None:
#     # Définissez le pad_token comme étant égal au eos_token
#     processor.tokenizer.pad_token = processor.tokenizer.eos_token
# if processor.tokenizer.pad_token_id is None:
#     processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
#     model.config.pad_token_id = processor.tokenizer.pad_token_id
# Spécifiez que vous utilisez des safetensors
# Avant le chargement du modèle
# torch.set_num_threads(4)  # Utilisez 4 coeurs CPU
# model._slow_forward = model.forward
# model.forward = lambda input_features: model._slow_forward(input_features.to(torch.float32))


# processor = WhisperProcessor.from_pretrained(
#     "/Users/moctar/Desktop/Projet_Master_2/speak_to_text_bambara/mon_model1_whisper",
#     tokenizer_file="mon_model1_whisper/tokenizer.json"  # Force le chargement du bon tokenizer
# )

# model = WhisperForConditionalGeneration.from_pretrained(
#     "/Users/moctar/Desktop/Projet_Master_2/speak_to_text_bambara/mon_model1_whisper",
#     use_safetensors=True,
#     torch_dtype=torch.float32,  # Forçage en float32
#     low_cpu_mem_usage=True
# ).to("cpu")

# def transcribe_with_whisper(uploaded_file):
#     # Créez un fichier temporaire pour enregistrer l'audio
#     with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
#         tmp_file.write(uploaded_file.read())
#         tmp_file_path = tmp_file.name

#     # Chargez l'audio
#     waveform, sample_rate = torchaudio.load(tmp_file_path)

#     # Rééchantillonnez à 16 kHz si nécessaire
#     if sample_rate != 16000:
#         resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
#         waveform = resampler(waveform)
#         sample_rate = 16000

#     # Préparez l'entrée pour le modèle Whisper
#     inputs = processor(waveform.squeeze(), sampling_rate=sample_rate, return_tensors="pt")

#     # Effectuez la transcription
#     with torch.no_grad():
#         generated_ids = model.generate(inputs.input_features)
#         transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

#     return transcription
from pydub import AudioSegment
import io, os, tempfile, torch, torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration


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

def transcribe_with_whisper(audio_bytes):
    # Conversion en tensor
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(audio_bytes)
        audio, sr = librosa.load(tmp.name, sr=16000)
    
    # Conversion explicite en float32
    inputs = processor(
        audio.astype(np.float32),  # Conversion cruciale
        sampling_rate=16000,
        return_tensors="pt"
    )
    
    # Conversion des features en float32
    input_features = inputs.input_features.to(torch.float32)
    
    with torch.inference_mode():
        outputs = model.generate(
            input_features,  # Utiliser la version convertie
            forced_decoder_ids=processor.get_decoder_prompt_ids(language="french", task="transcribe"),
            max_new_tokens=200
        )
    
    return processor.batch_decode(outputs, skip_special_tokens=True)[0]@st.cache_resource
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

def extract_transcription_text(result: list) -> str:
    """
    Extrait uniquement le texte depuis la liste de segments retournée par DJELIA (version 2).

    :param result: Liste de segments [{'text': ..., 'start': ..., 'end': ...}]
    :return: Texte combiné.
    """
    if isinstance(result, list) and len(result) > 0 and "text" in result[0]:
        return result[0]["text"]
    return ""

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

# def transcribe_audio(audio_buffer):
#     """Appelle un modèle de transcription local (à remplacer)"""
#     with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
#         tmp_wav.write(audio_buffer.read())
#         tmp_wav_path = tmp_wav.name
     
#     transcriber = ASR(model_id = "sudoping01/bambara-asr-v4-ic")
#     #model = joblib.load("mon_modele_transcription.joblib")
#     #transcription = model.transcribe(tmp_wav_path)
#     result = transcriber.transcribe_audio(tmp_wav_path)
#     print(result)
#     return result

def transcribe_djelia_file(wav_path: str, with_timestamps: bool = False, to_french: bool = False) -> str:
    """
    Transcrit un fichier .wav en texte avec l'API DJELIA.

    :param wav_path: Chemin vers un fichier audio .wav.
    :param with_timestamps: Si True, retourne aussi les timestamps (version 2).
    :param to_french: Si True, traduit directement en français.
    :return: Texte transcrit ou traduit.
    """
    try:
        if to_french:
            result = client.transcribe(wav_path, translate_to_french=True)
            return result["text"] if isinstance(result, dict) else str(result)

        if with_timestamps:
            segments = client.transcribe(wav_path, version=2)
            return "\n".join([seg["text"] for seg in segments])

        result = client.transcribe(wav_path)
        return result["text"] if isinstance(result, dict) else str(result)

    except Exception as e:
        st.error(f"Erreur lors de la transcription du fichier .wav avec DJELIA : {e}")
        return ""

# def translate_text(text: str) -> str:
#     """
#     Traduit du bambara vers le français en utilisant le modèle HF local.
#     """
#     # 1) Tokenize
#     inputs = tokenizer(text, return_tensors="pt")
#     # 2) Génération
#     outputs = translator_model.generate(**inputs)
#     # 3) Décodage
#     translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
#     return translated
def translate_djelia(text: str) -> str:
    """
    Traduit un texte du français vers le bambara en utilisant l'API Djelia.

    :param text: Texte en français à traduire.
    :return: Traduction en bambara.
    """
    try:
        result = client.translate(
            text=text,
            source="bam",   # français
            target="fr"   # bambara
        )
        return result["text"]
    except Exception as e:
        st.error(f"Erreur de traduction avec DJELIA : {e}")
        return ""


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

    transcription = transcribe_djelia_file(st.session_state['audio_data'])
    #transcription_text = transcription["chunks"][0]["text"]
    # st.write(transcription)
    # print(transcription)
    #transcription = transcribe_djelia_file(st.session_state['audio_data'])
    # Si transcription est une chaîne JSON, convertis-la en liste Python
    #transcription_list = json.loads(transcription)

    # Ensuite tu peux extraire le texte comme prévu
    # transcription_text = transcription_list[0]['text']
    # st.write(transcription_text)
    # st.write(transcription_text)
    #print("Transcription brute :", transcription)
    #print("Type de transcription :", type(transcription))
    raw = transcription  
    # Nouvelle extraction fiable :
    text = extract_text_from_transcription(raw)
    #st.write(text)

    #transcription = "Aw ni sɔgɔma.ça va ?"
    # transcription = transcribe_with_whisper(st.session_state['audio_data'])
    audio_bytes = st.session_state['audio_data'].getvalue()
    #transcription = transcribe_with_whisper(audio_bytes)
    if transcription:
        #translation = translate_djelia(transcription)
        # translation = translate_djelia(text)
        translation = "Bonjour. ça va ?"
        placeholder.empty()

        st.session_state['transcribed_text'] = transcription
        st.session_state['translated_text'] = translation

        st.success("Transcription et traduction terminées !")

        # Affichage côte à côte
        text_col, trans_col = st.columns(2)
        with text_col:
            st.markdown("** Transcription :**")
            st.write(transcription)
            # st.write(text)
        with trans_col:
            st.markdown("** Traduction  :**")
            st.write(translation)

        # Boutons
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        with btn_col1:
            if st.button("🔊 Écouter en Français"):
                if st.session_state['translated_text']:
                    try:
                        # Version avec TTS moderne
                        audio = text_to_audio(st.session_state['translated_text'], language='fr')
                        st.audio(audio, format="audio/wav")
                    except Exception as e:
                        # Fallback simple
                        st.audio(f"data:audio/wav;base64,{generate_base64_audio(st.session_state['translated_text'], 'fr')}", 
                                format="audio/wav")

        with btn_col2:
            if st.button("🔊 Écouter en Bambara (DJELIA)"):
                if st.session_state['transcribed_text']:
                    try:
                        # Utilisation correcte de l'API DJELIA
                        audio_bytes = client.text_to_speech(
                            #text=st.session_state['transcribed_text'],
                            text= transcription,  
                            speaker=2
                        )
                        st.audio(audio_bytes, format="audio/mp3")
                    except DjeliaError as e:
                        st.error(f"Erreur DJELIA: {str(e)}")
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
