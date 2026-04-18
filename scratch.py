from app.core.config import ml_resources
import pickle, onnxruntime as ort
ml_resources["tokenizer"] = pickle.load(open('weights/emotion_bilstm/tokenizer.pkl', 'rb'))
ml_resources["labels"] = pickle.load(open('weights/emotion_bilstm/labels.pkl', 'rb'))
session = ort.InferenceSession('weights/emotion_bilstm/model.onnx')
ml_resources["session"] = session
ml_resources["input_name"] = session.get_inputs()[0].name
from app.ml.emotion_bilstm import run_onnx_inference
try:
    print(run_onnx_inference("hello"))
except Exception as e:
    import traceback
    traceback.print_exc()
