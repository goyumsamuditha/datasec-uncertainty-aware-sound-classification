from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import train_cnn14 as training

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models/cnn14_initial/best_model.pt"
OUT = ROOT / "models/cnn14_occlusion"
N_EXAMPLES = 30
N_TIME = 10
N_FREQ = 8

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    main_map, sub_map = training.load_label_maps()

    if checkpoint["main_mapping"] != main_map:
        raise ValueError("Main-class mapping mismatch.")

    model = training.DataSECCNN14().to(training.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    manifest = training.load_manifest()
    val = manifest[manifest["partition"] == "validation"].copy()
    val = val.sample(n=N_EXAMPLES, random_state=42).reset_index(drop=True)
    dataset = training.DataSECDataset(val, main_map, sub_map, training=False)
    id_to_name = {v: k for k, v in main_map.items()}

    results = []

    for i in range(len(dataset)):
        waveform, actual_id, _ = dataset[i]
        waveform = waveform.unsqueeze(0).to(training.DEVICE)

        with torch.no_grad():
            original_logits, _ = model(waveform)
            original_probs = F.softmax(original_logits, dim=1)
            predicted_id = int(original_probs.argmax(1).item())
            original_prob = float(original_probs[0, predicted_id].item())

        captured = {}

        def capture(module, inputs, output):
            captured["shape"] = output.shape
            return output

        hook = model.backbone.logmel_extractor.register_forward_hook(capture)

        # Obtain the unmodified log-mel representation.
        with torch.no_grad():
            model(waveform)

        hook.remove()
        time_steps, mel_bins = captured["shape"][2:]
        mask_baseline = -100.0

        for axis, count in [("time", N_TIME), ("frequency", N_FREQ)]:
            for segment in range(count):
                start = segment * (time_steps if axis == "time" else mel_bins) // count
                end = (segment + 1) * (time_steps if axis == "time" else mel_bins) // count

                def mask_logmel(module, inputs, output):
                    masked = output.clone()
                    if axis == "time":
                        masked[:, :, start:end, :] = mask_baseline
                    else:
                        masked[:, :, :, start:end] = mask_baseline
                    return masked

                hook = model.backbone.logmel_extractor.register_forward_hook(mask_logmel)
                with torch.no_grad():
                    masked_logits, _ = model(waveform)
                    masked_prob = float(
                        F.softmax(masked_logits, dim=1)[0, predicted_id].item()
                    )
                hook.remove()

                results.append({
                    "relative_path": str(val.iloc[i]["relative_path"]),
                    "actual_class": id_to_name[int(actual_id.item())],
                    "predicted_class": id_to_name[predicted_id],
                    "original_probability": original_prob,
                    "mask_axis": axis,
                    "segment": segment,
                    "masked_probability": masked_prob,
                    "probability_drop": original_prob - masked_prob,
                    "prediction_correct": int(actual_id.item()) == predicted_id,
                })

        print(f"[{i + 1}/{len(dataset)}] {id_to_name[predicted_id]} completed")

    output = pd.DataFrame(results)
    output.to_csv(OUT / "occlusion_results.csv", index=False)

    summary = (
        output.groupby("mask_axis")["probability_drop"]
        .agg(["mean", "median", "std"])
        .reset_index()
    )
    summary.to_csv(OUT / "occlusion_summary.csv", index=False)

    print(f"Examples: {len(dataset)}")
    print(f"Occlusion tests: {len(output)}")
    print(f"Results: {OUT}")
    print("Test partition was not accessed.")

if __name__ == "__main__":
    main()
