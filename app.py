import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model import NeuralNetwork
from evaluation import evaluate_classification


st.set_page_config(
    page_title="Neural Network Lab",
    page_icon="🧠",
    layout="wide"
)


@st.cache_resource
def train_model():
    dataset = load_breast_cancer()
    X = dataset.data
    y = dataset.target.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    network = NeuralNetwork(
        input_size=X_train.shape[1],
        hidden_size=16,
        output_size=1,
        seed=42
    )

    history = network.fit(
        X_train,
        y_train,
        epochs=1000,
        learning_rate=0.01,
        verbose=False
    )

    probabilities = network.forward(X_test)
    metrics = evaluate_classification(
        y_test, probabilities, threshold=0.5
    )

    return dataset, network, scaler, X_test, y_test, history, metrics


st.title("🧠 Neural Network Lab")
st.caption("A feedforward neural network built from scratch using Python and NumPy.")

with st.spinner("Preparing the neural network..."):
    dataset, network, scaler, X_test, y_test, history, metrics = train_model()

st.success("Model trained and ready for demonstration.")

st.header("Model Performance")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
c2.metric("Precision", f"{metrics['precision']:.2%}")
c3.metric("Recall", f"{metrics['recall']:.2%}")
c4.metric("F1 Score", f"{metrics['f1_score']:.2%}")

st.divider()
st.header("Try a Prediction")
st.write(
    "Select a sample from the held-out test dataset. "
    "The app will show the network's prediction and the actual dataset label."
)

sample_index = st.slider(
    "Test sample index",
    min_value=0,
    max_value=len(X_test) - 1,
    value=0
)

sample = X_test[sample_index:sample_index + 1]
probability = float(network.forward(sample).ravel()[0])
predicted_class = int(probability >= 0.5)
actual_class = int(y_test[sample_index, 0])

labels = {0: "Malignant", 1: "Benign"}

p1, p2, p3 = st.columns(3)
p1.metric("Predicted class", labels[predicted_class])
p2.metric("Probability of benign", f"{probability:.2%}")
p3.metric("Actual dataset label", labels[actual_class])

if predicted_class == actual_class:
    st.success("Prediction matches the dataset label.")
else:
    st.warning("Prediction differs from the dataset label.")

st.caption(
    "Educational demonstration only. This model is not a medical diagnostic tool."
)

with st.expander("View sample feature values"):
    feature_table = pd.DataFrame({
        "Feature": dataset.feature_names,
        "Standardized value": X_test[sample_index]
    })
    st.dataframe(feature_table, width="stretch", hide_index=True)

st.divider()
st.header("Training Loss")

chart_data = pd.DataFrame({
    "Epoch": range(1, len(history) + 1),
    "Loss": history
}).set_index("Epoch")

st.line_chart(chart_data, y="Loss")

with st.expander("Confusion matrix counts"):
    st.write({
        "True positives": metrics["true_positive"],
        "True negatives": metrics["true_negative"],
        "False positives": metrics["false_positive"],
        "False negatives": metrics["false_negative"]
    })

st.divider()
st.caption("Built with Python, NumPy, scikit-learn, Matplotlib and Streamlit.")


