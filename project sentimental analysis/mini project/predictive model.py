import os
import sys
from tensorflow.keras.models import model_from_json
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
model_json_path = "model.json"
if not os.path.exists(model_json_path):
    print(f"Error: {model_json_path} not found")
    sys.exit()

with open(model_json_path, "r") as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)

model_weights_path = "C:\\Users\\admin\\Downloads\\model_weights.weights.h5"
loaded_model.load_weights(model_weights_path)

tokenizer_path = os.path.join(os.getcwd(), "tokenizer.pickle")
if not os.path.exists(tokenizer_path):
    print(f"Error: {tokenizer_path} not found")
    sys.exit()

with open(tokenizer_path, "rb") as tokenizer_file:
    loaded_tokenizer, maxlen = pickle.load(tokenizer_file)

def analyze_sentiment(review, loaded_model, loaded_tokenizer):
    review_sequence = loaded_tokenizer.texts_to_sequences([review])
    review_padded = pad_sequences(review_sequence, maxlen=maxlen)
    result = loaded_model.predict(review_padded)[0]
    sentiment = np.argmax(result)
    if sentiment == 0:
        return "Negative"
    elif sentiment == 1:
        return "Neutral"
    else:
        return "Positive"

new_review = input("Enter the review:")
sentiment = analyze_sentiment(new_review, loaded_model, loaded_tokenizer)
print("Sentiment:", sentiment)
