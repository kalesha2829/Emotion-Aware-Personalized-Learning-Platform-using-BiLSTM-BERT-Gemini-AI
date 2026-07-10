from pathlib import Path
from utils.predict import EmotionDetectionPipeline
from utils.constants import BERT_MODEL_DIR
pipeline = EmotionDetectionPipeline()
result = pipeline.predict('I am confused about recursion.')
print('bilstm_prediction', result.bilstm_prediction)
print('bert_prediction', result.bert_prediction)
print('final_emotion', result.final_emotion)
print('confidence_score', result.confidence_score)
print('bilstm_confidence', result.bilstm_confidence)
print('bert_confidence', result.bert_confidence)
print('mixed_emotion_breakdown', result.mixed_emotion_breakdown)
print('bert_dir files', sorted([p.name for p in BERT_MODEL_DIR.iterdir() if p.is_file()]))
