# 🧠 Next Quote Predictor

> A Transformer-based deep learning app that generates inspirational quotes from a seed phrase — powered by a custom-trained model and served with a Streamlit UI.

---

[Download Model](https://drive.google.com/file/d/15R92LBPEkgj49AvgKYLNeP1GOp35OX0e/view?usp=sharing)


## 🧠 Model Architecture

The model is a **single-layer Transformer** trained on the [Kaggle Quotes Dataset](https://www.kaggle.com/datasets/akmittal/quotes-dataset):

| Component            | Detail                          |
|---------------------|---------------------------------|
| Embedding dim        | 256                             |
| Attention heads      | 4                               |
| Feedforward dim      | 512                             |
| Vocabulary size      | ~21,671 tokens                  |
| Max sequence length  | 25 tokens                       |
| Total parameters     | ~12.4 million (47 MB)           |
| Training hardware    | Tesla P100-PCIE-16GB (Kaggle)   |

Generation uses **temperature sampling with top-k filtering** (default: k=8, temp=0.7) for diverse, human-like outputs.

---
## 🗂️ Project Structure

```
Quote Predictor/
├── app.py                  # Streamlit web application
├── training.ipynb          # Model training notebook (Kaggle/GPU)
├── next_quote_model.keras  # Trained Transformer model
├── tokenizer.pkl           # Fitted Keras tokenizer
├── config.json             # Model config (max sequence length)
└── requirements.txt        # Python dependencies
```

---

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- pip

### Steps

**1. Clone the repository**

```bash
git clone https://github.com/your-username/quote-predictor.git
cd quote-predictor
```

**2. (Recommended) Create a virtual environment**

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

> ⚠️ TensorFlow installation may vary by platform. If you face issues, install it manually:
> ```bash
> pip install tensorflow
> ```

---

## ▶️ Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

> **Note:** On first launch, the model loads into memory — this may take a few seconds.

---

## 🖥️ Usage

Once the app is running:

1. **Enter a seed phrase** in the text box (e.g., `"the purpose of life"`)
2. **Adjust the Creativity slider** (Temperature: 0.2 – 1.2)
   - Low values → more predictable, focused completions
   - High values → more surprising, creative completions
3. **Adjust the Prediction Length** (10 – 60 words)
4. Click **"Generate Quote"** and see your quote appear

### Example Seeds to Try

| Seed Text               | Sample Output                                                       |
|------------------------|---------------------------------------------------------------------|
| `beauty is`            | beauty is not in the face it is a light in the heart               |
| `the purpose of life`  | the purpose of life is to live it to taste experience to the utmost |
| `in the end`           | in the end it is not the years in your life that count             |
| `success is`           | success is not final failure is not fatal it is the courage to continue |

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for:

- Improved model architectures (multi-layer Transformer, LSTM hybrid)
- Better decoding strategies (beam search, nucleus sampling)
- UI enhancements or new features

---
