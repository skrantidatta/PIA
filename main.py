# main.py
import os
# Unbuffered output + quieter logs for pyannote/speechbrain/tokenizers
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYANNOTE_AUDIO_LOG_LEVEL"] = "ERROR"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import torch
import argparse
import torch.nn.functional as F
from datetime import datetime

from src.model import CNNMultistreamModel
from src.utils import (
    get_device, process_single_video, build_single_video_tensors_fromdisk,
    default_phoneme_vocab, IMAGES_PER_PHON
)

# ==== weights/model hyperparams ====
WEIGHTS_PATH = "checkpoints/best_model.pth"
EMBED_DIM    = 128
NUM_HEADS    = 4
FPS          = 25.0
LABEL_FOLDER = "testvideo"  # subfolder under --outdir
# ===============================================

def load_weights(model: torch.nn.Module, ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    return ckpt

def main():
    ap = argparse.ArgumentParser(description="Single-video deepfake test (disk crops)")
    ap.add_argument("--video", required=True, help="Path to input video (.mp4/.mov)")
    ap.add_argument("--outdir", required=True, help="Where to save crops/json/results")
    args = ap.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    device = get_device()
    print(f"[INFO] Using device: {device}", flush=True)

 
    aligned_json, label_dir = process_single_video(
        video_path=args.video,
        output_dir=outdir,
        label=LABEL_FOLDER,
        fps=FPS,
        whisper_model_name="large-v2",
        language="en",
        compute_type="float32"
    )
    print(f"[OK] Aligned JSON: {aligned_json}", flush=True)


    phoneme_vocab = default_phoneme_vocab()  # replace with saved vocab loader if needed
    geoms, imgs, arcs, mask = build_single_video_tensors_fromdisk(
        aligned_json_path=aligned_json,
        label_dir=label_dir,
        phoneme_vocab=phoneme_vocab,
        device=torch.device(device)
    )


    model = CNNMultistreamModel(
        arcface_dim=512, geo_dim=1,
        embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
        frames_per_phon=IMAGES_PER_PHON
    ).to(device)
    load_weights(model, WEIGHTS_PATH, device)
    model.eval()

    # with torch.no_grad():
    #     logits = model(geoms, imgs, arcs, mask)
    #     probs = F.softmax(logits, dim=1).cpu().numpy().tolist()[0]
    #     pred_label = int(probs[1] > 0.5)  # 1=Fake, 0=Real
    
    with torch.no_grad():
        logits, _ = model(geoms, imgs, arcs, mask)  
        probs = F.softmax(logits, dim=1).cpu().numpy().tolist()[0]
        pred_label = int(probs[1] > 0.5)  # 1=Fake, 0=Real


    label_map = {0: "Real", 1: "Fake"}
    result = {
        "model": "PIA",
        "code_reference": "https://github.com/skrantidatta/PIA",
        "paper_reference": "https://arxiv.org/pdf/2510.14241",
        "video": os.path.abspath(args.video),
        "predicted_label": pred_label,
        "predicted_label_name": label_map[pred_label],
        "probs": {"real": probs[0], "fake": probs[1]},
        "weights": WEIGHTS_PATH,
        "aligned_json": aligned_json,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    print(f"[RESULT] {result['predicted_label_name']}  (P(fake)={probs[1]:.4f})", flush=True)

    results_path = os.path.join(outdir, "results.json")
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[SAVED] {results_path}", flush=True)

if __name__ == "__main__":
    main()
