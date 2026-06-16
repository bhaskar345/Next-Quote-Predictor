import streamlit as st
import json
import pickle
import numpy as np
import onnxruntime as ort
from tensorflow.keras.preprocessing.sequence import pad_sequences

@st.cache_resource
def load_all():

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    session = ort.InferenceSession(
        "next_quote_model.onnx",
        sess_options=session_options,
        providers=["CPUExecutionProvider"]
    )

    with open("config.json", "r") as f:
        config = json.load(f)

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    return session, tokenizer, config["max_len"]


session, tokenizer, max_len = load_all()

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

id_to_word = {
    idx: word
    for word, idx in tokenizer.word_index.items()
}

def sample_with_temperature(preds, temperature=0.7, top_k=8):

    preds = np.asarray(preds).astype(np.float64)

    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds - np.max(preds))
    probs = exp_preds / np.sum(exp_preds)

    if top_k is not None:

        top_indices = np.argpartition(
            probs,
            -top_k
        )[-top_k:]

        top_probs = probs[top_indices]
        top_probs /= top_probs.sum()

        next_token = np.random.choice(
            top_indices,
            p=top_probs
        )

    else:

        next_token = np.random.choice(
            len(probs),
            p=probs
        )

    return int(next_token)

def generate_quote(
    seed_text,
    next_tokens=30,
    temperature=0.7
):

    start_token = tokenizer.word_index.get("<START>")
    end_token = tokenizer.word_index.get("<END>")

    seed_tokens = tokenizer.texts_to_sequences(
        [seed_text]
    )[0]

    token_list = [start_token] + seed_tokens

    current_tokens = token_list[-(max_len - 1):]

    output_words = seed_text.split()

    for _ in range(next_tokens):

        padded = pad_sequences(
            [current_tokens],
            maxlen=max_len - 1,
            padding="pre"
        )

        padded = padded.astype(np.int32)

        preds = session.run(
            [output_name],
            {input_name: padded}
        )[0]

        preds = preds[0]

        next_token_id = sample_with_temperature(
            preds,
            temperature=temperature
        )

        if next_token_id == end_token:
            break

        next_word = id_to_word.get(
            next_token_id,
            ""
        )

        if next_word:
            output_words.append(next_word)

        current_tokens.append(next_token_id)

        current_tokens = current_tokens[-(max_len - 1):]

    return " ".join(output_words)

st.set_page_config(
    page_title="Next Quote Prediction",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Next Quote Prediction")

st.write(
    "Enter a text prompt and the model will generate the rest of the quote."
)

seed_text = st.text_input(
    "Enter starting text:",
    value="beauty is"
)

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

    if not seed_text.strip():

        st.warning(
            "Please enter some text."
        )

    else:

        with st.spinner(
            "Generating quote..."
        ):

            result = generate_quote(
                seed_text,
                next_tokens=length,
                temperature=temperature
            )

        st.subheader(
            "Generated Quote"
        )

        st.success(
            result + "."
        )