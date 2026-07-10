import traceback
from pathlib import Path
from utils.bert_model import load_bert_artifacts
from utils.constants import BERT_MODEL_DIR, BERT_CHECKPOINT_DIR
print('cwd', Path.cwd())
print('bert model dir', BERT_MODEL_DIR.exists(), BERT_MODEL_DIR)
print('bert checkpoint dir', BERT_CHECKPOINT_DIR.exists(), BERT_CHECKPOINT_DIR)
try:
    art = load_bert_artifacts()
    print('loaded artifacts', type(art['model']), type(art['tokenizer']), art['label_mapping'].keys())
except Exception as e:
    print('ERROR', type(e).__name__, e)
    traceback.print_exc()
