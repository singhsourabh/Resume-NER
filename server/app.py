import io
import argparse
import torch
from transformers import BertTokenizerFast, BertForTokenClassification
from flask import Flask, jsonify, request
from utils import preprocess_data, predict, idx2tag

parser = argparse.ArgumentParser(description='Train Bert-NER')
parser.add_argument('-p', type=str, help='path of trained model state dict')
args = parser.parse_args().__dict__


app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

MAX_LEN = 500
NUM_LABELS = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'bert-base-uncased'
STATE_DICT = torch.load(args['p'], map_location=DEVICE)
TOKENIZER = BertTokenizerFast(MODEL_PATH, lowercase=True)

model = BertForTokenClassification.from_pretrained(
    MODEL_PATH, state_dict=STATE_DICT['model_state_dict'], num_labels=NUM_LABELS)
model.to(DEVICE)


@app.route('/predict', methods=['POST'])
def predict_api():
    if request.method == 'POST':
        data = io.BytesIO(request.files.get('resume').read())
        resume_text = preprocess_data(data)
        entities = predict(model, TOKENIZER, idx2tag,
                           DEVICE, resume_text, MAX_LEN)
        return jsonify({'entities': entities})


if __name__ == '__main__':
    app.run()
