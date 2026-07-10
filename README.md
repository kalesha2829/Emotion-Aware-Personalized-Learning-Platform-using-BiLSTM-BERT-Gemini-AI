# 🎓 Emotion-Aware Personalized Learning Platform using BiLSTM, BERT & Gemini AI

---

## 📖 Project Overview

The Emotion-Aware Personalized Learning Platform is an intelligent, web-based educational assistant designed to identify and support student emotional states during learning. Students frequently experience a wide range of emotions—such as curiosity, confusion, confidence, frustration, and boredom—while studying. By analyzing student reflections using a hybrid deep learning ensemble (combining BiLSTM and BERT models), the platform predicts the student's current emotional state. It then leverages Google Gemini AI to generate personalized study advice, tailored encouragement, and actionable next steps. This integration creates an empathetic, supportive study environment that enhances student engagement and learning efficiency.

---

## ✨ Key Features

- **Emotion Detection using BiLSTM**: Processes sequential text input using a Bidirectional LSTM network to capture sequential context and sentiment.
- **Emotion Detection using BERT**: Extracts deep semantic representations using a pre-trained transformer model to recognize complex emotional nuances in study text.
- **Ensemble Emotion Prediction**: Integrates predictions from both BiLSTM and BERT models to determine a final resolved emotional state and compute confidence scores.
- **Google Gemini AI Learning Support**: Requests personalized, empathetic guidance, study tips, and recommended next steps based on the student's resolved emotion.
- **Analytics Dashboard**: Visualizes model performance, confidence trends, emotion distributions, and model agreement frequencies.
- **Emotion Visualization**: Renders predictions, confidence scores, and mixed emotion breakdowns in an easy-to-read interface.
- **CSV Interaction Logging**: Saves user queries, classification results, confidence scores, and Gemini recommendations into `logs/interactions.csv` for tracking.
- **Streamlit Web Interface**: Provides a responsive, clean, and intuitive multi-page web application for student interaction.

---

## 🛠 Technology Stack

| Category | Technology |
| --- | --- |
| Frontend | Streamlit |
| Backend | Python |
| Machine Learning | scikit-learn, joblib |
| Deep Learning | TensorFlow, Keras, PyTorch |
| Natural Language Processing | Transformers (Hugging Face), NLTK, Tokenizers |
| Generative AI | Google Gemini API (google-generativeai SDK) |
| Visualization | Plotly, Matplotlib, Seaborn |
| Development Tools | python-dotenv, Git |

---

## 📂 Project Structure

```text
.
├── Project Documentation/
│   ├── 1. Brainstorming & Ideation/
│   ├── 2.Requirement Analysis/
│   ├── 3. Project Design Phase/
│   ├── 4. Project Planning Phase/
│   ├── 5. Project Development Phase/
│   ├── 6. Project Testing/
│   ├── 7. Project Documentation/
│   └── 8. Project Demonstration/
├── assets/
├── configs/
├── dataset/
├── logs/
├── models/
├── notebooks/
├── pages/
│   ├── 1_Emotion_Detection.py
│   ├── 2_Learning_Support.py
│   └── 3_Analytics.py
├── scripts/
├── utils/
├── .env.example
├── .gitignore
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/kalesha2829/Emotion-Aware-Personalized-Learning-Platform-using-BiLSTM-BERT-Gemini-AI.git
   cd Emotion-Aware-Personalized-Learning-Platform-using-BiLSTM-BERT-Gemini-AI
   ```

2. **Create a Virtual Environment**
   - Windows:
     ```bash
     python -m venv venv311
     ```
   - macOS / Linux:
     ```bash
     python3 -m venv venv311
     ```

3. **Activate the Virtual Environment**
   - Windows:
     ```bash
     venv311\Scripts\activate
     ```
   - macOS / Linux:
     ```bash
     source venv311/bin/activate
     ```

4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure Environment Variables**
   - Copy the example environment template:
     ```bash
     copy .env.example .env
     ```
   - Open the `.env` file and insert your Google Gemini API key:
     ```text
     GEMINI_API_KEY=your_actual_api_key_here
     ```

---

## ▶️ How to Run

Launch the Streamlit web application from the terminal:
```bash
streamlit run app.py
```

Open your web browser and navigate to `http://localhost:8501`.

---

## 🤖 AI Models Used

- **BiLSTM (Bidirectional Long Short-Term Memory)**: A recurrent neural network that processes text in both forward and backward directions. This model is trained to recognize emotional patterns based on the sequential and structural relationships of words in student text.
- **BERT (Bidirectional Encoder Representations from Transformers)**: A pre-trained transformer model fine-tuned on study-related text data. It captures deep contextual and semantic representations, enabling the system to understand complex language structure and subtle emotional nuances.
- **Google Gemini AI**: A large language model integrated via the Google Generative AI API. It analyzes the resolved student emotion along with their text input to act as an empathetic tutor, generating customized study guidance, constructive tips, and learning steps.

---

## 📑 Project Documentation

The repository includes a comprehensive set of documents organized across the project lifecycle:

- **1. Brainstorming & Ideation**
- **2. Requirement Analysis**
- **3. Project Design Phase**
- **4. Project Planning Phase**
- **5. Project Development Phase**
- **6. Project Testing**
- **7. Project Documentation**
- **8. Project Demonstration**

---

## 🎥 Demo

The demo video link is available in:
Project Documentation → 8. Project Demonstration.

---

## 🚀 Future Scope

- **Multi-Modal Analysis**: Incorporate facial expression recognition and speech analysis to capture a broader range of affective states.
- **Personalized Profile Tracking**: Implement user authentication and database storage to track student emotional trends and study progress over time.
- **Support for Special Domains**: Expand Gemini prompting to supply specific, subject-matter coaching for different learning fields (e.g., Mathematics, Coding, Literature).

---

## 👨💻 Author

**Kalesha Shaik**

Individual College AI/ML Project

---

## 📄 License

This project was developed for academic and educational purposes.

