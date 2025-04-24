# Application de Transcription et Traduction Bambara ↔ Français

Cette application permet de :

- Enregistrer un message audio en Bambara depuis le navigateur
- Le transcrire automatiquement avec deux systèmes : Whisper (local) et DJELIA (API)
- Le traduire en français avec deux modèles : un modèle local et l'API DJELIA
- Écouter toutes les versions textuelles et audio générées

## Fonctionnalités

- Enregistrement audio
- Transcription automatique (Whisper et DJELIA)
- Traduction vers le français (modèle local et DJELIA)
- Synthèse vocale (Google Text-to-Speech et DJELIA)
- Comparaison visuelle des résultats

## Lancer l’application

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Téléchargement des modèles

Télécharger les modèles ici : [Google Drive - Modèles](https://drive.google.com/drive/folders/14qsNtBHF_rs_-FoSRa0wzHcZND1uTGFW?usp=drive_link)

## Arborescence attendue

```
├── app.py
├── requirements.txt
├── modele_whisper_transcription/
├── modele_nllb_traduction/
```
