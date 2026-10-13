"""Streamlit dashboard for the leakage-safe GloVe + BiGRU + attention model."""
from pathlib import Path
import re

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


MODEL_PATH = Path("bigru_attention.keras")
VOCAB_PATH = Path("vocabulary.txt")
MAX_SEQUENCE_LENGTH = 256

DATELINE_RE = re.compile(
    r"^\s*(?:[A-Z][A-Z .'-]{2,40}(?:\s*\([^)]{2,30}\))?\s*(?:[-:|]|\u2014|\u2013)\s*)"
)
URL_RE = re.compile(
    r"(?:https?://|www\.)\S+|\b[\w.-]+\.(?:com|org|net|io|co\.[a-z]{2})\S*", re.I
)
HTML_RE = re.compile(r"<[^>]+>")
HANDLE_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")


def clean_text(text: str) -> str:
    """Apply the same artifact-removal rules used during notebook training."""
    if not isinstance(text, str):
        return ""
    text = text.replace("\u2014", " - ").replace("\u2013", " - ")
    text = HTML_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = HANDLE_RE.sub(" ", text)
    text = DATELINE_RE.sub(" ", text)
    text = re.sub(r"\b(?:RT|via)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-zA-Z0-9\s'!?.,-]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


@tf.keras.utils.register_keras_serializable(package="FakeNews")
class Attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        feature_dim = int(input_shape[-1])
        self.score = self.add_weight(
            name="score", shape=(feature_dim, 1), initializer="glorot_uniform"
        )
        self.bias = self.add_weight(name="bias", shape=(1,), initializer="zeros")
        super().build(input_shape)

    def call(self, inputs, mask=None):
        logits = tf.squeeze(tf.matmul(tf.tanh(inputs), self.score) + self.bias, axis=-1)
        if mask is not None:
            logits = tf.where(mask, logits, tf.fill(tf.shape(logits), tf.cast(-1e9, logits.dtype)))
        weights = tf.nn.softmax(logits, axis=1)
        return tf.reduce_sum(inputs * tf.expand_dims(weights, -1), axis=1)

    def compute_mask(self, inputs, mask=None):
        return None


@st.cache_resource
def load_artifacts(model_path: str, vocab_path: str):
    vocabulary = Path(vocab_path).read_text(encoding="utf-8").splitlines()
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=len(vocabulary),
        output_mode="int",
        output_sequence_length=MAX_SEQUENCE_LENGTH,
        standardize=None,
        vocabulary=vocabulary,
    )
    model = tf.keras.models.load_model(model_path, custom_objects={"Attention": Attention})
    return model, vectorizer


def predict_dataframe(dataframe, model, vectorizer):
    texts = (
        dataframe.get("title", pd.Series("", index=dataframe.index)).fillna("").astype(str)
        + " "
        + dataframe.get("text", pd.Series("", index=dataframe.index)).fillna("").astype(str)
    ).map(clean_text)
    sequences = vectorizer(tf.constant(texts.tolist())).numpy()
    probabilities = model.predict(sequences, verbose=0).ravel()
    return probabilities


st.set_page_config(page_title="Fake News Detection", page_icon="N", layout="wide")
st.title("Fake News Detection")
st.caption("Traditional deep learning: GloVe embeddings + BiGRU + trainable attention")

with st.sidebar:
    st.header("Model artifacts")
    model_path = st.text_input("Keras model", str(MODEL_PATH))
    vocab_path = st.text_input("Vocabulary", str(VOCAB_PATH))
    st.divider()
    st.subheader("Architecture")
    st.write("Embedding (GloVe) -> SpatialDropout1D -> BiGRU -> Attention -> Dense -> Sigmoid")

try:
    model, vectorizer = load_artifacts(model_path, vocab_path)
except Exception as error:
    st.error(f"Could not load model artifacts: {error}")
    st.info("Run the notebook export cell first, then start Streamlit from the Fake News Detection folder.")
    st.stop()

left, right = st.columns([1.5, 1])
with left:
    st.subheader("Classify an article")
    title = st.text_input("Headline")
    article = st.text_area("Article text", height=280)
    if st.button("Analyze article", type="primary"):
        if not (title.strip() or article.strip()):
            st.warning("Enter a headline or article body.")
        else:
            cleaned = clean_text(f"{title} {article}")
            sequence = vectorizer(tf.constant([cleaned])).numpy()
            fake_probability = float(model.predict(sequence, verbose=0)[0][0])
            label = "FAKE" if fake_probability >= 0.5 else "REAL"
            st.metric("Prediction", label)
            st.progress(fake_probability, text=f"Fake probability: {fake_probability:.1%}")
            st.caption("The score is a model probability, not a fact-check or source-of-truth verdict.")

with right:
    st.subheader("Metric guide")
    with st.expander("Precision", expanded=True):
        st.write("Of articles predicted fake, the proportion that are actually fake.")
    with st.expander("Recall"):
        st.write("Of truly fake articles, the proportion the model catches.")
    with st.expander("False Positive Rate"):
        st.write("Real articles incorrectly flagged as fake: FP / (FP + TN).")
    with st.expander("Why attention?"):
        st.write("Attention learns which BiGRU time steps matter most when forming the article representation.")

st.divider()
st.subheader("Batch evaluation")
st.write("Upload a CSV with `title`, `text`, and `label` columns to inspect test-set metrics.")
upload = st.file_uploader("Evaluation CSV", type="csv")
if upload is not None:
    evaluation_data = pd.read_csv(upload)
    if "label" not in evaluation_data.columns:
        st.error("The evaluation CSV must include a label column where 0 = real and 1 = fake.")
    else:
        probabilities = predict_dataframe(evaluation_data, model, vectorizer)
        predictions = (probabilities >= 0.5).astype("int32")
        labels = evaluation_data["label"].astype("int32").to_numpy()
        report = classification_report(labels, predictions, target_names=["real", "fake"], output_dict=True)
        matrix = confusion_matrix(labels, predictions, labels=[0, 1])
        false_positive_rate = matrix[0, 1] / max(matrix[0].sum(), 1)
        metric_a, metric_b, metric_c = st.columns(3)
        metric_a.metric("Accuracy", f"{report['accuracy']:.2%}")
        metric_b.metric("Fake precision", f"{report['fake']['precision']:.2%}")
        metric_c.metric("False positive rate", f"{false_positive_rate:.2%}")
        st.dataframe(pd.DataFrame(report).T.style.format("{:.3f}"))
        st.write("Confusion matrix: rows are true labels [real, fake], columns are predictions [real, fake].")
        st.dataframe(pd.DataFrame(matrix, index=["true real", "true fake"], columns=["pred real", "pred fake"]))
