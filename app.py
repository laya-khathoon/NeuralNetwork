

from __future__ import annotations



import io

import html

import numpy as np

import pandas as pd

import streamlit as st

import matplotlib.pyplot as plt



from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    confusion_matrix,

)



# ---------------------------------------------------------

# PAGE CONFIGURATION

# ---------------------------------------------------------

st.set_page_config(

    page_title="Neural Network Studio",

    page_icon="◈",

    layout="wide",

    initial_sidebar_state="expanded",

)



BG = "#0B0B10"

PANEL = "#13131C"

BORDER = "#282838"

VIOLET = "#9B8AFB"

MUTED = "#9999B1"

TEXT = "#F4F3FA"





# ---------------------------------------------------------

# GLOBAL STYLING

# ---------------------------------------------------------

st.markdown(

    f"""

    <style>

    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600;700;800&display=swap');



    html, body, [class*="css"] {{

        font-family: 'Inter', sans-serif;

    }}



    .stApp {{

        background: {BG};

        color: {TEXT};

    }}



    [data-testid="stSidebar"] {{

        background: #101018;

        border-right: 1px solid {BORDER};

    }}



    [data-testid="stHeader"] {{

        background: rgba(11,11,16,0.92);

    }}



    h1, h2, h3 {{

        color: {TEXT};

        letter-spacing: -0.035em;

    }}



    .brand {{

        color: {VIOLET};

        font-family: 'DM Mono', monospace;

        font-size: 12px;

        letter-spacing: 2px;

        font-weight: 500;

    }}



    .hero-title {{

    font-size: clamp(28px, 3vw, 38px);

    font-weight: 800;

    line-height: 1.08;

    letter-spacing: -1.5px;

    margin: 6px 0;

    }}



    .subtle {{

        color: {MUTED};

        font-size: 14px;

        line-height: 1.7;

    }}



    .panel {{

        background: {PANEL};

        border: 1px solid {BORDER};

        border-radius: 16px;

        padding: 20px;

        margin-bottom: 14px;

    }}



    .panel-label {{

        color: {MUTED};

        font-size: 11px;

        font-weight: 700;

        letter-spacing: 1.6px;

        text-transform: uppercase;

        margin-bottom: 12px;

    }}



    .metric-card {{

        background: {PANEL};

        border: 1px solid {BORDER};

        border-radius: 14px;

        padding: 18px;

        min-height: 105px;

    }}



    .metric-label {{

        color: {MUTED};

        font-size: 12px;

        margin-bottom: 10px;

    }}



    .metric-value {{

        color: {TEXT};

        font-size: 27px;

        font-weight: 800;

        letter-spacing: -1px;

    }}



    .metric-accent {{

        color: {VIOLET};

    }}



    .stButton > button {{

        background: {VIOLET};

        color: #100D20;

        border: none;

        border-radius: 10px;

        font-weight: 800;

        min-height: 44px;

        transition: all .2s ease;

    }}



    .stButton > button:hover {{

        background: #B2A7FF;

        color: #100D20;

        border: none;

        box-shadow: 0 0 24px #9B8AFB33;

    }}



    .stTextInput input, .stNumberInput input,

    [data-baseweb="select"] > div {{

        background: #191925;

        border-color: {BORDER};

        border-radius: 9px;

    }}



    div[data-testid="stFileUploader"] {{

        background: #151520;

        border: 1px dashed #494565;

        border-radius: 12px;

        padding: 12px;

    }}



    .stTabs [data-baseweb="tab-list"] {{

        gap: 10px;

        border-bottom: 1px solid {BORDER};

    }}



    .stTabs [data-baseweb="tab"] {{

        color: {MUTED};

        padding: 12px 16px;

    }}



    .stTabs [aria-selected="true"] {{

        color: {VIOLET} !important;

    }}



    hr {{

        border-color: {BORDER};

    }}



    [data-testid="stMetricValue"] {{

        color: {VIOLET};

    }}



    .status-pill {{

        display: inline-block;

        border: 1px solid #50477B;

        color: #C6BCFF;

        background: #211C35;

        border-radius: 20px;

        padding: 5px 10px;

        font-size: 11px;

        font-weight: 700;

    }}

    </style>

    """,

    unsafe_allow_html=True,

)





# ---------------------------------------------------------

# NUMPY MULTI-LAYER NETWORK

# A small educational MLP trained with backpropagation.

# ---------------------------------------------------------

class NumpyMLP:

    def __init__(self, layer_sizes, seed=42):

        if len(layer_sizes) < 3:

            raise ValueError(

                "Configure an input layer, at least one hidden layer, "

                "and an output layer."

            )



        if any(int(size) < 1 for size in layer_sizes):

            raise ValueError("Every layer must contain at least one neuron.")



        self.sizes = [int(x) for x in layer_sizes]

        rng = np.random.default_rng(seed)



        self.weights = []

        self.biases = []



        for fan_in, fan_out in zip(self.sizes[:-1], self.sizes[1:]):

            self.weights.append(

                rng.normal(

                    0,

                    np.sqrt(2.0 / fan_in),

                    size=(fan_in, fan_out),

                )

            )

            self.biases.append(np.zeros((1, fan_out)))



    @staticmethod

    def sigmoid(x):

        x = np.clip(x, -40, 40)

        return 1.0 / (1.0 + np.exp(-x))



    def forward(self, X):

        activations = [X]

        current = X



        for i, (weight, bias) in enumerate(

            zip(self.weights, self.biases)

        ):

            z = current @ weight + bias



            if i == len(self.weights) - 1:

                current = self.sigmoid(z)

            else:

                current = np.maximum(0, z)



            activations.append(current)



        return activations



    def loss(self, y, probabilities):

        eps = 1e-8

        p = np.clip(probabilities, eps, 1 - eps)

        return float(

            -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

        )



    def fit(self, X, y, epochs=300, learning_rate=0.01):

        history = []

        n = len(X)



        for _ in range(epochs):

            activations = self.forward(X)

            prediction = activations[-1]

            history.append(self.loss(y, prediction))



            # Binary cross-entropy + sigmoid output gradient.

            delta = (prediction - y) / n



            grad_w = [None] * len(self.weights)

            grad_b = [None] * len(self.biases)



            for layer in reversed(range(len(self.weights))):

                grad_w[layer] = (

                    activations[layer].T @ delta

                )

                grad_b[layer] = np.sum(delta, axis=0, keepdims=True)



                if layer > 0:

                    delta = delta @ self.weights[layer].T

                    delta *= (activations[layer] > 0)



            for layer in range(len(self.weights)):

                self.weights[layer] -= learning_rate * grad_w[layer]

                self.biases[layer] -= learning_rate * grad_b[layer]



        history.append(self.loss(y, self.forward(X)[-1]))

        return history



    def predict_proba(self, X):

        return self.forward(X)[-1]



    def predict(self, X, threshold=0.5):

        return (self.predict_proba(X) >= threshold).astype(int)





# ---------------------------------------------------------

# DATASET PREPARATION

# ---------------------------------------------------------

@st.cache_data(show_spinner=False)

def builtin_dataset(name):

    if name == "Breast Cancer (demo)":

        data = load_breast_cancer()

        frame = pd.DataFrame(data.data, columns=data.feature_names)

        frame["target"] = data.target

        return frame



    if name == "OR Logic":

        return pd.DataFrame({

            "input_a": [0, 0, 1, 1],

            "input_b": [0, 1, 0, 1],

            "target": [0, 1, 1, 1],

        })



    if name == "XOR Logic":

        return pd.DataFrame({

            "input_a": [0, 0, 1, 1],

            "input_b": [0, 1, 0, 1],

            "target": [0, 1, 1, 0],

        })



    raise ValueError("Unknown built-in dataset.")





def prepare_data(frame, target_column, test_fraction, seed):

    if frame.empty:

        raise ValueError("The dataset is empty.")



    if target_column not in frame.columns:

        raise ValueError("Select a valid target column.")



    if frame[target_column].isna().any():

        raise ValueError(

            "The target column contains missing values. "

            "Remove those rows before training."

        )



    features = frame.drop(columns=[target_column]).copy()

    target = frame[target_column].copy()



    # This version uses numeric features only.

    non_numeric = features.select_dtypes(exclude=[np.number]).columns.tolist()

    if non_numeric:

        raise ValueError(

            "Non-numeric feature columns are not supported yet: "

            + ", ".join(map(str, non_numeric))

        )



    features = features.replace([np.inf, -np.inf], np.nan)

    features = features.fillna(features.median(numeric_only=True))



    if features.shape[1] == 0:

        raise ValueError("Add at least one numeric feature column.")



    if target.nunique() != 2:

        raise ValueError(

            "The target must contain exactly two classes for binary "

            "classification. Select a binary target column."

        )



    classes = list(pd.unique(target))

    mapping = {classes[0]: 0, classes[1]: 1}

    y = target.map(mapping).to_numpy(dtype=float).reshape(-1, 1)

    X = features.to_numpy(dtype=float)



    if len(X) < 5:

        raise ValueError(

            "At least five rows are recommended for a train/test split."

        )



    indices = np.arange(len(X))

    try:

        train_idx, test_idx = train_test_split(

            indices,

            test_size=test_fraction,

            random_state=seed,

            stratify=y.ravel(),

        )

    except ValueError as exc:

        raise ValueError(

            "Could not split the data by class. Ensure each class has "

            "enough examples, or adjust the test fraction."

        ) from exc



    scaler = StandardScaler()

    X_train = scaler.fit_transform(X[train_idx])

    X_test = scaler.transform(X[test_idx])



    return (

        X_train,

        X_test,

        y[train_idx],

        y[test_idx],

        scaler,

        features.columns.tolist(),

        mapping,

    )





# ---------------------------------------------------------

# INTERACTIVE NETWORK DIAGRAM

# ---------------------------------------------------------

def render_network_diagram(layer_sizes):

    width = 900

    height = 330

    left = 65

    right = 835

    top = 48

    bottom = 275



    max_nodes = 8

    visible_sizes = [min(n, max_nodes) for n in layer_sizes]

    x_positions = np.linspace(left, right, len(layer_sizes))



    node_positions = []

    for x, count in zip(x_positions, visible_sizes):

        if count == 1:

            ys = [(top + bottom) / 2]

        else:

            ys = np.linspace(top, bottom, count).tolist()

        node_positions.append([(x, y) for y in ys])



    parts = [

        f'<svg xmlns="http://www.w3.org/2000/svg" '

        f'viewBox="0 0 {width} {height}" width="100%">',

        f'<rect width="{width}" height="{height}" rx="16" fill="#101018"/>',

    ]



    # Connections first, behind nodes.

    for layer in range(len(node_positions) - 1):

        for x1, y1 in node_positions[layer]:

            for x2, y2 in node_positions[layer + 1]:

                parts.append(

                    f'<line x1="{x1:.1f}" y1="{y1:.1f}" '

                    f'x2="{x2:.1f}" y2="{y2:.1f}" '

                    f'stroke="#6554B8" stroke-opacity="0.30" '

                    f'stroke-width="1.1"/>'

                )



    labels = ["INPUT"] + [

        f"HIDDEN {i}" for i in range(1, len(layer_sizes) - 1)

    ] + ["OUTPUT"]



    for i, (nodes, count, x, label) in enumerate(

        zip(node_positions, layer_sizes, x_positions, labels)

    ):

        fill = "#8D7AF5" if i not in (0, len(layer_sizes) - 1) else "#C4B5FD"

        for x_node, y_node in nodes:

            parts.append(

                f'<circle cx="{x_node:.1f}" cy="{y_node:.1f}" r="9" '

                f'fill="{fill}" stroke="#E8E1FF" stroke-width="1.1"/>'

            )



        shown = visible_sizes[i]

        if count > shown:

            parts.append(

                f'<text x="{x:.1f}" y="{bottom + 23}" text-anchor="middle" '

                f'fill="#C7BFFF" font-size="11">+{count - shown} more</text>'

            )



        parts.append(

            f'<text x="{x:.1f}" y="25" text-anchor="middle" '

            f'fill="#F4F3FA" font-size="12" font-weight="700">{label}</text>'

        )

        parts.append(

            f'<text x="{x:.1f}" y="313" text-anchor="middle" '

            f'fill="#AAA6C4" font-size="12">{count} neurons</text>'

        )



    parts.append("</svg>")

    st.markdown("".join(parts), unsafe_allow_html=True)





def metric_card(label, value, note=""):

    st.markdown(

        f"""

        <div class="metric-card">

            <div class="metric-label">{html.escape(label)}</div>

            <div class="metric-value metric-accent">{html.escape(str(value))}</div>

            <div class="subtle" style="font-size:11px">{html.escape(note)}</div>

        </div>

        """,

        unsafe_allow_html=True,

    )





# ---------------------------------------------------------

# SIDEBAR

# ---------------------------------------------------------

with st.sidebar:

    st.markdown('<div class="brand">◈ NN / STUDIO</div>', unsafe_allow_html=True)

    st.markdown("## Workspace")

    st.caption("Configure · Train · Inspect")



    st.markdown("---")

    st.markdown("**01 — Dataset**")



    dataset_source = st.radio(

        "Dataset source",

        ["Built-in dataset", "Upload CSV"],

        label_visibility="collapsed",

    )



    uploaded = None

    if dataset_source == "Built-in dataset":

        dataset_name = st.selectbox(

            "Choose dataset",

            ["XOR Logic", "OR Logic", "Breast Cancer (demo)"],

        )

        frame = builtin_dataset(dataset_name)

    else:

        uploaded = st.file_uploader(

            "Upload a CSV file",

            type=["csv"],

            help="Use numeric feature columns and a binary target column.",

        )

        if uploaded is None:

            frame = None

        else:

            try:

                frame = pd.read_csv(uploaded)

                dataset_name = uploaded.name

            except Exception as exc:

                st.error(f"Could not read CSV: {exc}")

                frame = None



    target_column = None

    if frame is not None:

        target_column = st.selectbox(

            "Target column",

            options=list(frame.columns),

            index=len(frame.columns) - 1,

        )



    st.markdown("---")

    st.markdown("**02 — Architecture**")



    hidden_layer_count = st.number_input(

        "Hidden layers",

        min_value=1,

        max_value=5,

        value=2,

        step=1,

    )



    hidden_sizes = []

    for i in range(int(hidden_layer_count)):

        hidden_sizes.append(

            st.number_input(

                f"Layer {i + 1} neurons",

                min_value=1,

                max_value=128,

                value=16 if i == 0 else 8,

                step=1,

                key=f"hidden_size_{i}",

            )

        )



    st.caption("Hidden layers use ReLU. Output uses sigmoid.")



    st.markdown("---")

    st.markdown("**03 — Training**")



    epochs = st.slider("Epochs", 10, 2000, 300, step=10)

    learning_rate = st.select_slider(

        "Learning rate",

        options=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1],

        value=0.01,

        format_func=lambda x: f"{x:g}",

    )

    test_fraction = st.slider(

        "Test split",

        min_value=0.1,

        max_value=0.4,

        value=0.2,

        step=0.05,

    )

    seed = st.number_input("Random seed", min_value=0, max_value=99999, value=42)



    train_clicked = st.button(

        "▶  Train network",

        use_container_width=True,

        type="primary",

    )



    st.markdown("---")

    st.caption("NUMPY ENGINE · EDUCATIONAL USE")





# ---------------------------------------------------------

# MAIN HEADER

# ---------------------------------------------------------

st.markdown('<div class="brand">MACHINE LEARNING / PLAYGROUND</div>', unsafe_allow_html=True)

st.markdown('<div class="hero-title">Neural Network<br>Studio.</div>', unsafe_allow_html=True)

st.markdown(

    '<div class="subtle">Design your architecture. Train from scratch. '

    'Understand every layer.</div>',

    unsafe_allow_html=True,

)

st.write("")



tab_design, tab_train, tab_predictions, tab_data = st.tabs(

    ["◈  Architecture", "↗  Training", "⌕  Predictions", "▤  Dataset"]

)



# Architecture shown from the currently selected dataset.

if frame is not None and target_column is not None:

    input_count = len([c for c in frame.columns if c != target_column])

else:

    input_count = 2



layer_sizes = [input_count] + [int(x) for x in hidden_sizes] + [1]



with tab_design:

    left_col, right_col = st.columns([1.65, 1], gap="large")



    with left_col:

        st.markdown('<div class="panel-label">NETWORK CANVAS</div>', unsafe_allow_html=True)

        render_network_diagram(layer_sizes)

        st.markdown(

            '<div class="subtle">Connections represent trainable weights. '

            'The displayed network is the current configuration.</div>',

            unsafe_allow_html=True,

        )





    with right_col:

        st.markdown(

            '<div class="panel-label">ARCHITECTURE OVERVIEW</div>',

            unsafe_allow_html=True,

        )



        total_neurons = sum(layer_sizes)

        total_weights = sum(

            a * b for a, b in zip(layer_sizes[:-1], layer_sizes[1:])

        )

        total_biases = sum(layer_sizes[1:])



        # Compact summary cards

        summary_cols = st.columns(2)

        with summary_cols[0]:

            metric_card("LAYERS", len(layer_sizes), "Input to output")

        with summary_cols[1]:

            metric_card("NEURONS", total_neurons, "Across all layers")



        summary_cols = st.columns(2)

        with summary_cols[0]:

            metric_card("WEIGHTS", total_weights, "Trainable parameters")

        with summary_cols[1]:

            metric_card("BIASES", total_biases, "Trainable parameters")



        st.markdown("---")

        st.markdown("#### Layer inspector")



        layer_options = []

        for index, count in enumerate(layer_sizes):

            if index == 0:

                name = "Input layer"

            elif index == len(layer_sizes) - 1:

                name = "Output layer"

            else:

                name = f"Hidden layer {index}"



            layer_options.append(f"{index:02d} · {name} ({count} neurons)")



        selected_layer_label = st.selectbox(

            "Select a layer",

            layer_options,

            key="architecture_layer_inspector",

        )

        selected_layer = layer_options.index(selected_layer_label)

        selected_neurons = layer_sizes[selected_layer]



        if selected_layer == 0:

            activation = "Input / identity"

            incoming_weights = 0

            biases = 0

        else:

            activation = (

                "Sigmoid"

                if selected_layer == len(layer_sizes) - 1

                else "ReLU"

            )

            incoming_weights = (

                layer_sizes[selected_layer - 1] * selected_neurons

            )

            biases = selected_neurons



        st.markdown(

            f"""

            <div class="panel">

                <div class="panel-label">SELECTED LAYER</div>

                <div style="font-size:22px;font-weight:800">

                    {html.escape(layer_options[selected_layer].split(" · ")[1])}

                </div>

                <hr style="border-color:#282838">

                <div style="display:flex;justify-content:space-between;padding:7px 0">

                    <span style="color:#9999B1">Neurons</span>

                    <b>{selected_neurons}</b>

                </div>

                <div style="display:flex;justify-content:space-between;padding:7px 0">

                    <span style="color:#9999B1">Activation</span>

                    <b>{activation}</b>

                </div>

                <div style="display:flex;justify-content:space-between;padding:7px 0">

                    <span style="color:#9999B1">Incoming weights</span>

                    <b>{incoming_weights}</b>

                </div>

                <div style="display:flex;justify-content:space-between;padding:7px 0">

                    <span style="color:#9999B1">Bias parameters</span>

                    <b>{biases}</b>

                </div>

            </div>

            """,

            unsafe_allow_html=True,

        )





# ---------------------------------------------------------

# TRAINING

# ---------------------------------------------------------

if train_clicked:

    if frame is None:

        st.error("Upload a CSV file before training.")

    elif target_column is None:

        st.error("Choose a target column.")

    else:

        try:

            (

                X_train, X_test, y_train, y_test, scaler,

                feature_names, class_mapping,

            ) = prepare_data(frame, target_column, test_fraction, int(seed))



            actual_layer_sizes = (

                [X_train.shape[1]]

                + [int(x) for x in hidden_sizes]

                + [1]

            )



            network = NumpyMLP(actual_layer_sizes, seed=int(seed))



            progress = st.progress(0, text="Training network...")

            history = []



            # Train in chunks so the UI can report progress.

            chunk_size = max(1, int(epochs) // 25)

            trained_epochs = 0



            while trained_epochs < int(epochs):

                step_epochs = min(chunk_size, int(epochs) - trained_epochs)

                chunk_history = network.fit(

                    X_train,

                    y_train,

                    epochs=step_epochs,

                    learning_rate=float(learning_rate),

                )

                history.extend(chunk_history[:-1])

                trained_epochs += step_epochs

                progress.progress(

                    min(trained_epochs / int(epochs), 1.0),

                    text=f"Training network... {trained_epochs}/{epochs} epochs",

                )



            history.append(network.loss(y_train, network.predict_proba(X_train)))

            progress.empty()



            probabilities = network.predict_proba(X_test)

            predictions = (probabilities >= 0.5).astype(int)



            metrics = {

                "Accuracy": accuracy_score(y_test, predictions),

                "Precision": precision_score(

                    y_test, predictions, zero_division=0

                ),

                "Recall": recall_score(

                    y_test, predictions, zero_division=0

                ),

                "F1 score": f1_score(

                    y_test, predictions, zero_division=0

                ),

            }



            st.session_state["studio_result"] = {

                "network": network,

                "history": history,

                "metrics": metrics,

                "X_test": X_test,

                "y_test": y_test,

                "probabilities": probabilities,

                "predictions": predictions,

                "scaler": scaler,

                "feature_names": feature_names,

                "class_mapping": class_mapping,

                "dataset_name": dataset_name,

                "layer_sizes": actual_layer_sizes,

                "epochs": int(epochs),

                "learning_rate": float(learning_rate),

                "target_column": target_column,

                "test_fraction": float(test_fraction),

            }



            st.success("Training completed. Open the Training and Predictions tabs.")

        except Exception as exc:

            st.error(f"Training failed: {exc}")





result = st.session_state.get("studio_result")



with tab_train:

    if result is None:

        st.markdown(

            '<div class="panel"><div class="panel-label">NO TRAINING RUN</div>'

            '<h3>Your experiment starts here.</h3>'

            '<p class="subtle">Choose a dataset, configure the architecture, '

            'and press Train network in the sidebar.</p></div>',

            unsafe_allow_html=True,

        )

    else:

        st.markdown(

            f'<span class="status-pill">TRAINING COMPLETE</span> '

            f'<span class="subtle">{html.escape(result["dataset_name"])}</span>',

            unsafe_allow_html=True,

        )

        st.write("")



        metric_cols = st.columns(4)

        for col, (name, value) in zip(metric_cols, result["metrics"].items()):

            with col:

                metric_card(name, f"{value:.3f}", "Test set")



        chart_col, info_col = st.columns([1.6, 1], gap="large")



        with chart_col:

            st.markdown("### Loss curve")

            fig, ax = plt.subplots(figsize=(8, 3.5))

            fig.patch.set_facecolor(PANEL)

            ax.set_facecolor(PANEL)

            ax.plot(

                np.arange(1, len(result["history"]) + 1),

                result["history"],

                color=VIOLET,

                linewidth=2.2,

            )

            ax.set_xlabel("Training step", color=MUTED)

            ax.set_ylabel("Binary cross-entropy", color=MUTED)

            ax.tick_params(colors=MUTED)

            ax.grid(alpha=0.15, color="#8580A0")

            for spine in ax.spines.values():

                spine.set_color(BORDER)

            st.pyplot(fig)

            plt.close(fig)



        with info_col:

            st.markdown("### Run details")


            st.markdown(f"**Dataset:** {result['dataset_name']}")

            st.markdown(f"**Layers:** {' → '.join(map(str, result['layer_sizes']))}")

            st.markdown(f"**Epochs:** {result['epochs']}")

            st.markdown(f"**Learning rate:** {result['learning_rate']:g}")

            st.markdown(f"**Target:** {result['target_column']}")

            st.markdown(f"**Test samples:** {len(result['y_test'])}")




        st.markdown("### Confusion matrix")

        cm = confusion_matrix(result["y_test"], result["predictions"], labels=[0, 1])

        cm_frame = pd.DataFrame(

            cm,

            index=["Actual 0", "Actual 1"],

            columns=["Predicted 0", "Predicted 1"],

        )

        st.dataframe(cm_frame, use_container_width=True)





with tab_predictions:

    if result is None:

        st.info("Train a network first to explore its predictions.")

    else:

        st.markdown("### Prediction explorer")

        st.markdown(

            '<div class="subtle">Inspect held-out samples and compare the '

            'predicted probability with the actual class.</div>',

            unsafe_allow_html=True,

        )



        count = len(result["y_test"])

        sample_index = st.slider(

            "Test sample",

            min_value=0,

            max_value=max(0, count - 1),

            value=0,

        )



        probability = float(result["probabilities"][sample_index, 0])

        predicted_class = int(probability >= 0.5)

        actual_class = int(result["y_test"][sample_index, 0])



        a, b, c = st.columns(3)

        with a:

            metric_card("Predicted class", predicted_class, "Threshold: 0.50")

        with b:

            metric_card("Probability of class 1", f"{probability:.3f}", "Sigmoid output")

        with c:

            metric_card(

                "Actual class",

                actual_class,

                "Correct" if predicted_class == actual_class else "Incorrect",

            )



        st.markdown("#### Scaled feature values")

        st.dataframe(

            pd.DataFrame(

                result["X_test"][sample_index].reshape(1, -1),

                columns=result["feature_names"],

            ),

            hide_index=True,

        )



        st.markdown("#### All test predictions")

        prediction_table = pd.DataFrame({

            "Actual": result["y_test"].ravel().astype(int),

            "Predicted": result["predictions"].ravel().astype(int),

            "Probability (class 1)": result["probabilities"].ravel(),

        })

        prediction_table["Correct"] = (

            prediction_table["Actual"] == prediction_table["Predicted"]

        )

        st.dataframe(prediction_table, hide_index=True)





with tab_data:

    st.markdown("### Dataset preview")

    if frame is None:

        st.info("Choose a built-in dataset or upload a CSV file in the sidebar.")

    else:

        st.markdown(

            f'<div class="subtle">{len(frame):,} rows · {len(frame.columns)} columns</div>',

            unsafe_allow_html=True,

        )

        st.dataframe(frame.head(100), hide_index=True)



        buffer = io.StringIO()

        frame.to_csv(buffer, index=False)

        st.download_button(

            "Download current dataset as CSV",

            data=buffer.getvalue(),

            file_name="dataset_export.csv",

            mime="text/csv",

        )



        st.markdown("#### Data requirements")

        st.markdown(

            "- Numeric feature columns only.\n"

            "- Exactly two classes in the target column.\n"

            "- Missing numeric feature values are filled with column medians.\n"

            "- Missing target values are rejected.\n"

            "- The test set is held out from gradient-based training."

        )



st.markdown("---")

st.markdown(

    '<div class="subtle">NEURAL NETWORK STUDIO · NUMPY BACKPROPAGATION · '

    'Educational use only. The built-in breast cancer dataset is a demonstration, '

    'not a medical diagnostic tool.</div>',

    unsafe_allow_html=True,

)
