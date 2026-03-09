import streamlit as st
import json
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.config import enable_unsafe_deserialization

enable_unsafe_deserialization()

def last_token(x):
    return x[:, -1, :]

@st.cache_resource
def load_all():
    model = load_model(
        "next_quote_model.keras",
        custom_objects={"last_token": last_token}
    )

    with open("config.json", "r") as f:
        config = json.load(f)

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer, config["max_len"]

model, tokenizer, max_len = load_all()

id_to_word = {idx: word for word, idx in tokenizer.word_index.items()}

def sample_with_temperature(preds, temperature=0.7, top_k=8):

    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    probs = exp_preds / np.sum(exp_preds)

    if top_k is not None:
        top_indices = np.argpartition(probs, -top_k)[-top_k:]
        top_probs = probs[top_indices]
        top_probs /= top_probs.sum()
        next_token = np.random.choice(top_indices, p=top_probs)
    else:
        next_token = np.random.choice(len(probs), p=probs)

    return next_token

def generate_quote(seed_text, next_tokens=30, temperature=0.7):

    start_token = tokenizer.word_index.get("<START>")
    end_token = tokenizer.word_index.get("<END>")

    token_list = [start_token] + tokenizer.texts_to_sequences([seed_text])[0]
    current_tokens = token_list[-(max_len-1):]

    output_words = seed_text.split()

    for _ in range(next_tokens):

        padded = pad_sequences([current_tokens], maxlen=max_len-1, padding='pre')
        preds = model.predict(padded, verbose=0)[0]

        next_token_id = sample_with_temperature(preds, temperature)

        if next_token_id == end_token:
            break

        next_word = id_to_word.get(next_token_id, "")
        output_words.append(next_word)

        current_tokens.append(next_token_id)
        current_tokens = current_tokens[-(max_len-1):]

    return " ".join(output_words)

st.title("🧠 Next Quote Prediction")

st.write("Enter a text prompt and the model will generate the rest of the quote.")

seed_text = st.text_input("Enter starting text:", "beauty is")

temperature = st.slider(
    "Creativity (Temperature)",
    min_value=0.2,
    max_value=1.2,
    value=0.7,
    step=0.1
)

length = st.slider(
    "Prediction Length",
    min_value=10,
    max_value=60,
    value=30
)

if st.button("Generate Quote"):

    if seed_text.strip() == "":
        st.warning("Please enter some text")

    else:
        with st.spinner("Generating..."):
            result = generate_quote(seed_text, length, temperature)

        st.subheader("Generated Quote")
        st.success(result + ".")