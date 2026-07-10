# AI Learning Assistant

AI Learning Assistant is a Streamlit-based web application for student emotion detection and learning guidance. It combines BiLSTM and BERT emotion models with personalized support from Google Gemini AI.

## Project Overview

This project analyzes study-related text inputs from students and predicts emotions such as Bored, Confident, Confused, Curious, and Frustrated. It offers:

- dual-model emotion detection using BiLSTM and BERT
- ensemble emotion prediction with mixed emotion breakdown
- Gemini-powered learning guidance and study tips
- analytics dashboard for dataset and model behavior
- CSV logging of predictions and Gemini responses

## Features

- Emotion detection page with BiLSTM and BERT outputs
- Personalized learning support page using Gemini fallback logic
- Analytics dashboard with Plotly charts
- Interaction logging to `logs/interactions.csv`
- Graceful handling for missing models and API dependencies

## Technology Stack

- Python 3.10+
- Streamlit
- TensorFlow / Keras
- Hugging Face Transformers
- PyTorch
- NLTK
- Plotly
- Pandas, NumPy
- scikit-learn
- python-dotenv
- Google Gemini API integration

## Installation

1. Clone or download this repository.

2. Create a Python virtual environment:

   ```bash
   python -m venv venv
   ```

3. Activate the environment:

   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS / Linux:
     ```bash
     source venv/bin/activate
     ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Copy the environment file and configure keys:

   ```bash
   copy .env.example .env   # Windows
   # cp .env.example .env   # macOS / Linux
   ```

   Then open `.env` and add your Gemini API key if available.

## Usage

Run the application with Streamlit:

```bash
streamlit run app.py
```

If your shell does not expose the `streamlit` command, use the Python launcher:

```bash
py -m streamlit run app.py
```

Open the app in your browser at `http://localhost:8501`.

## Project Structure

```
app.py                      # Streamlit entry point
requirements.txt            # Python dependencies
.env.example                # Environment variable template
assets/                     # Generated assets and visualizations
configs/                    # Training configuration examples
dataset/                    # Raw and processed dataset files
logs/                       # CSV interaction logs
models/                     # Saved model artifacts and checkpoints
notebooks/                  # Jupyter notebooks for exploration
pages/                      # Streamlit multipage app modules
scripts/                    # Dataset and model training scripts
utils/                      # Shared project utilities
```

## Notes

- The app can use BERT checkpoints located under `models/bert_checkpoints` if the main `models/bert_emotion` directory is not present.
- Gemini AI features require the `google-generativeai` package and a valid API key in `.env`.
- Prediction logs are written to `logs/interactions.csv` automatically.

## Model Artifacts and BERT Fallback

- Required model artifact locations:
  - BiLSTM model: `models/bilstm_emotion.keras`
  - BiLSTM tokenizer: `models/bilstm_tokenizer.joblib`
  - Label encoder: `models/label_encoder.joblib`
  - BiLSTM max-length config: `models/bilstm_max_len.joblib`
  - Full BERT model directory: `models/bert_emotion/`
  - Alternative BERT checkpoints: `models/bert_checkpoints/`

- BERT checkpoint fallback behavior:
  - The app first attempts to load the full BERT model from `models/bert_emotion/`.
  - If that directory is unavailable, it falls back to `models/bert_checkpoints/` and loads the latest saved checkpoint there.
  - This fallback allows the app to keep using BERT-based emotion prediction even when the packaged full model directory is missing.

- Running the application without the full BERT model:
  - Ensure the BiLSTM artifacts are present in `models/`.
  - Ensure `models/bert_checkpoints/` contains valid Hugging Face checkpoint files.
  - Start the app normally with:

    ```bash
    py -m streamlit run app.py
    ```

  - The app will still run and use the fallback BERT checkpoints when `models/bert_emotion/` is not found.

## License

Add your license here.
