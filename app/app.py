from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import streamlit as st
import torch
import torchaudio
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================
# PATHS AND CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))

import train_cnn14 as training


CHECKPOINT_PATH = (
    ROOT / "models" / "cnn14_optimisation"
    / "FINAL-123" / "best_model.pt"
)

TEMPERATURE_PATH = (
    ROOT / "models" / "cnn14_final123_temperature_scaling"
    / "calibration_metrics.json"
)

CONFORMAL_PATH = (
    ROOT / "models" / "cnn14_final123_conformal"
    / "conformal_parameters.npz"
)

DEVICE = training.DEVICE


# =========================================================
# STREAMLIT PAGE
# =========================================================

st.set_page_config(
    page_title="DataSEC Sound Classifier",
    page_icon="ðŸ”Š",
    layout="wide",
)

st.title("DataSEC Environmental Sound Classification")

st.markdown(
    """
    An uncertainty-aware and explainable neural network
    for environmental sound classification.

    **Model:** PANNs CNN14 - FINAL-123
    """
)

st.info(
    "This application provides research decision support only. "
    "Predictions are not regulatory measurements or proof of "
    "environmental noise compliance."
)


# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource
def load_model():

    required_files = [
        CHECKPOINT_PATH,
        TEMPERATURE_PATH,
        CONFORMAL_PATH,
        ROOT / "logs" / "hierarchical_label_mapping.json",
    ]

    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    main_map, sub_map = training.load_label_maps()

    if checkpoint["main_mapping"] != main_map:
        raise ValueError("Main-class mapping does not match checkpoint.")

    if checkpoint["subclass_mapping"] != sub_map:
        raise ValueError("Subclass mapping does not match checkpoint.")

    model = training.DataSECCNN14().to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    with open(TEMPERATURE_PATH, "r", encoding="utf-8") as file:
        calibration = json.load(file)

    conformal = np.load(CONFORMAL_PATH)

    temperature = float(calibration["temperature"])
    conformal_temperature = float(conformal["temperature"])
    conformal_threshold = float(conformal["threshold"])

    id_to_main = {
        index: name for name, index in main_map.items()
    }

    return (
        model,
        temperature,
        conformal_temperature,
        conformal_threshold,
        id_to_main,
    )


# =========================================================
# AUDIO PREPROCESSING
# =========================================================

def preprocess_audio(audio_bytes: bytes):
    if not audio_bytes:
        raise ValueError("The uploaded audio file is empty.")

    try:
        audio, sample_rate = sf.read(
            io.BytesIO(audio_bytes),
            dtype="float32",
            always_2d=True,
        )

    except Exception as error:
        raise ValueError(
            f"Unable to read this audio file: {error}"
        ) from error

    if audio.size == 0:
        raise ValueError("The uploaded audio file is empty.")

    if not np.isfinite(audio).all():
        raise ValueError("Audio contains invalid numerical values.")

    original_duration = len(audio) / sample_rate

    # Convert stereo or multichannel audio to mono.
    audio = audio.mean(axis=1)

    waveform = torch.from_numpy(
        audio.copy()
    ).float().unsqueeze(0)

    # Resample to the model's required sample rate.
    if sample_rate != training.SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=sample_rate,
            new_freq=training.SAMPLE_RATE,
        )

    original_resampled_length = waveform.shape[-1]

    # Centre crop long recordings, matching validation preprocessing.
    if original_resampled_length > training.NUM_SAMPLES:

        start = (
            original_resampled_length - training.NUM_SAMPLES
        ) // 2

        waveform = waveform[
            :, start:start + training.NUM_SAMPLES
        ]

    # Zero-pad short recordings.
    elif original_resampled_length < training.NUM_SAMPLES:

        waveform = F.pad(
            waveform,
            (0, training.NUM_SAMPLES - original_resampled_length),
        )

    if waveform.shape != (1, training.NUM_SAMPLES):
        raise ValueError("Unexpected processed audio dimensions.")

    if float(waveform.abs().max()) < 1e-7:
        raise ValueError(
            "The uploaded audio is silent or contains near-zero "
            "amplitude throughout."
        )

    return waveform.squeeze(0), original_duration


# =========================================================
# PREDICTION
# =========================================================

def predict_audio(model, waveform, temperature,
                  conformal_temperature, conformal_threshold):

    waveform = waveform.unsqueeze(0).to(DEVICE)

    with torch.inference_mode():

        main_logits, subclass_logits = model(waveform)

        # Temperature-scaled probabilities for displayed confidence.
        calibrated_probabilities = torch.softmax(
            main_logits / temperature,
            dim=1,
        )

        # Separate temperature for conformal prediction.
        conformal_probabilities = torch.softmax(
            main_logits / conformal_temperature,
            dim=1,
        )

    probabilities = calibrated_probabilities[0].cpu().numpy()
    conformal_probs = conformal_probabilities[0].cpu().numpy()

    predicted_id = int(np.argmax(probabilities))

    # Global conformal prediction set.
    prediction_set = [
        index
        for index, probability in enumerate(conformal_probs)
        if (1.0 - float(probability)) <= conformal_threshold
    ]

    return predicted_id, probabilities, prediction_set



# =========================================================
# GRAD-CAM EXPLANATION
# =========================================================

def generate_gradcam(model, waveform, predicted_id, id_to_main):

    captured = {}

    def capture_activations(module, inputs, output):
        captured["activations"] = output
        output.register_hook(
            lambda gradient: captured.__setitem__(
                "gradients", gradient
            )
        )

    hook = model.backbone.conv_block6.register_forward_hook(
        capture_activations
    )

    try:
        model.zero_grad(set_to_none=True)

        input_waveform = waveform.unsqueeze(0).to(DEVICE)

        with torch.enable_grad():
            main_logits, _ = model(input_waveform)

            target_score = main_logits[0, predicted_id]
            target_score.backward()

        activations = captured["activations"].detach()
        gradients = captured["gradients"].detach()

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        cam = torch.relu(
            (weights * activations).sum(dim=1, keepdim=True)
        )

        cam_min = cam.min()
        cam_max = cam.max()

        is_constant = (
            (cam_max - cam_min).item() <= 1e-8
        )

        if not is_constant:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        with torch.no_grad():
            spectrogram = model.backbone.spectrogram_extractor(
                input_waveform
            )
            logmel = model.backbone.logmel_extractor(
                spectrogram
            )

        logmel_image = (
            logmel.squeeze().detach().cpu().numpy()
        )

        if logmel_image.ndim != 2:
            raise RuntimeError(
                f"Unexpected log-mel shape: {logmel_image.shape}"
            )

        cam_resized = F.interpolate(
            cam,
            size=logmel_image.shape,
            mode="bilinear",
            align_corners=False,
        )

        cam_image = cam_resized.squeeze().cpu().numpy()

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(12, 8),
            sharex=True,
            constrained_layout=True,
        )

        time_extent = [0, training.CLIP_SECONDS]
        frequency_extent = [0, training.MEL_BINS]

        axes[0].imshow(
            logmel_image.T,
            origin="lower",
            aspect="auto",
            extent=time_extent + frequency_extent,
            cmap="magma",
        )
        axes[0].set_title("Log-mel spectrogram")
        axes[0].set_ylabel("Mel-bin index")

        axes[1].imshow(
            logmel_image.T,
            origin="lower",
            aspect="auto",
            extent=time_extent + frequency_extent,
            cmap="gray",
        )

        if not is_constant:
            axes[1].imshow(
                cam_image.T,
                origin="lower",
                aspect="auto",
                extent=time_extent + frequency_extent,
                cmap="jet",
                alpha=0.45,
                vmin=0,
                vmax=1,
            )

        axes[1].set_title(
            f"Grad-CAM for predicted class: "
            f"{id_to_main[predicted_id]}"
        )
        axes[1].set_xlabel("Time (seconds)")
        axes[1].set_ylabel("Mel-bin index")

        fig.suptitle(
            f"Grad-CAM explanation - "
            f"{id_to_main[predicted_id]}",
            fontsize=12,
        )

        return fig, is_constant

    finally:
        hook.remove()

# =========================================================
# LOAD MODEL
# =========================================================

try:

    with st.spinner("Loading the trained neural network..."):

        (
            model,
            temperature,
            conformal_temperature,
            conformal_threshold,
            id_to_main,
        ) = load_model()

    st.success(
        f"FINAL-123 model loaded successfully on {DEVICE}."
    )

except Exception as error:

    st.error(f"Model loading failed: {error}")
    st.stop()


# =========================================================
# MODEL INFORMATION
# =========================================================

with st.expander("Model and processing information"):

    col1, col2, col3 = st.columns(3)

    col1.metric("Main classes", len(id_to_main))
    col2.metric("Input sample rate", "32 kHz")
    col3.metric("Input duration", "10 seconds")

    st.write(
        f"Confidence calibration temperature: {temperature:.7f}"
    )

    st.write(
        f"Conformal temperature: {conformal_temperature:.7f}"
    )

    st.write(
        f"Conformal threshold: {conformal_threshold:.7f}"
    )

    st.write(
        "Audio is converted to mono, resampled, centre-cropped "
        "or zero-padded to match the training preprocessing."
    )


# =========================================================
# AUDIO UPLOAD
# =========================================================

st.header("1. Upload an audio recording")

uploaded_file = st.file_uploader(
    "Choose a WAV audio file",
    type=["wav"],
    help="Upload a WAV recording for environmental sound classification.",
)

if uploaded_file is not None:

    st.audio(
        uploaded_file.getvalue(),
        format="audio/wav",
    )

    if st.button("Classify audio", type="primary"):

        try:

            with st.spinner("Processing audio and generating predictions..."):

                waveform, original_duration = preprocess_audio(
                    uploaded_file.getvalue()
                )

                predicted_id, probabilities, prediction_set = predict_audio(
                    model,
                    waveform,
                    temperature,
                    conformal_temperature,
                    conformal_threshold,
                )

            predicted_class = id_to_main[predicted_id]
            confidence = float(probabilities[predicted_id])

            st.header("2. Classification result")

            col1, col2 = st.columns(2)

            col1.metric(
                "Predicted environmental sound",
                predicted_class,
            )

            col2.metric(
                "Calibrated confidence",
                f"{confidence:.2%}",
            )

            if original_duration > 10:
                st.warning(
                    f"The original recording is {original_duration:.2f} "
                    "seconds long. Only its central 10 seconds were "
                    "classified. Results may not represent the entire "
                    "recording."
                )

            elif original_duration < 10:
                st.info(
                    f"The recording is {original_duration:.2f} seconds "
                    "long. Zero-padding was applied to reach 10 seconds."
                )

            # -------------------------------------------------
            # TOP FIVE PREDICTIONS
            # -------------------------------------------------

            st.header("3. Top-five class predictions")

            top_indices = np.argsort(probabilities)[::-1][:5]

            top_data = {
                id_to_main[int(index)]: float(probabilities[index])
                for index in top_indices
            }

            st.bar_chart(top_data)

            for class_name, probability in top_data.items():

                st.write(
                    f"**{class_name}:** {probability:.2%}"
                )

            # -------------------------------------------------
            # CONFORMAL PREDICTION SET
            # -------------------------------------------------

            st.header("4. Uncertainty and deferral")

            if len(prediction_set) == 0:

                st.warning(
                    "DEFER: The conformal prediction set is empty. "
                    "The model does not provide an accepted class "
                    "under the configured conformal rule."
                )

            else:

                prediction_names = [
                    id_to_main[index]
                    for index in prediction_set
                ]

                st.write("Conformal prediction set:")

                for name in prediction_names:
                    st.write(f"- {name}")

                if len(prediction_set) == 1:

                    st.success(
                        "Singleton prediction set: one class "
                        "is included by the conformal rule."
                    )

                else:

                    st.warning(
                        "Multiple possible classes: manual listening "
                        "or further assessment is recommended."
                    )

            st.caption(
                "The conformal procedure uses a target calibration "
                "coverage of 90%. This is not a guarantee of 90% "
                "coverage on future recordings or out-of-distribution data."
            )


            # -------------------------------------------------
            # GRAD-CAM EXPLANATION
            # -------------------------------------------------

            st.header("5. Explainability - Grad-CAM")

            try:
                with st.spinner("Generating Grad-CAM explanation..."):
                    fig, is_constant = generate_gradcam(
                        model, waveform, predicted_id, id_to_main
                    )

                st.pyplot(fig)
                plt.close(fig)

                if is_constant:
                    st.warning(
                        "The Grad-CAM heatmap is constant for this recording. "
                        "No meaningful highlighted regions could be generated."
                    )
                else:
                    st.caption(
                        "Highlighted regions indicate areas that influenced "
                        "the predicted class. They are not causal evidence."
                    )

            except Exception as gradcam_error:
                st.warning(
                    f"Grad-CAM could not be generated: {gradcam_error}"
                )

        except Exception as error:

            st.error(f"Audio processing or prediction failed: {error}")


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "DataSEC research application | FINAL-123 PANNs CNN14 | "
    "Predictions are intended for research decision support."
)




