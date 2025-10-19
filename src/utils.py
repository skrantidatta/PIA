# src/utils.py
import os
import gc
import cv2
import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# quiet down noisy loggers
import logging
logging.getLogger("speechbrain").setLevel(logging.ERROR)

import mediapipe as mp
import whisperx
import insightface
from phonemizer import phonemize
from PIL import Image
from torchvision import transforms


IMAGES_PER_PHON = 5
IMAGE_SIZE = (112, 112)
KEEP_PHONEMES = {"i", "æ", "k", "ɹ", "b", "w", "t", "m", "p", "o", "f", "v", "ʃ", "s"}
IGNORED_PHONEMES = {"<sil>", "<SIL>", "sp", "<sp>", "", " "}
CLOSURE_PHONEMES = {"m", "b", "p"}
CLOSURE_THRESHOLD = 0.3



def get_device() -> str:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    return dev


# ----- MediaPipe lip geometry -----
TOP_CENTER = [13, 14, 15]
BOTTOM_CENTER = [17, 18, 87]
LIP_INDICES = list(set([
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 78,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    13, 15, 17, 18, 291
]))


def extract_mouth_patch(frame, landmarks, margin=10):
    h, w, _ = frame.shape
    xs = [int(landmarks[i].x * w) for i in LIP_INDICES]
    ys = [int(landmarks[i].y * h) for i in LIP_INDICES]
    min_x, max_x = max(min(xs) - margin, 0), min(max(xs) + margin, w)
    min_y, max_y = max(min(ys) - margin, 0), min(max(ys) + margin, h)
    return frame[min_y:max_y, min_x:max_x], (min_x, min_y, max_x, max_y)


def extract_mouth_states(video_path: str, save_crops: bool, output_crop_dir: str) -> List[Dict]:
    print("[STEP] Viseme/geometry...", flush=True)
    if save_crops and output_crop_dir:
        os.makedirs(output_crop_dir, exist_ok=True)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
    )
    cap = cv2.VideoCapture(video_path)
    frame_idx, mouth_data = 0, []
    t0 = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            h, w, _ = frame.shape
            top_lip_y    = np.mean([landmarks[i].y for i in TOP_CENTER]) * h
            bottom_lip_y = np.mean([landmarks[i].y for i in BOTTOM_CENTER]) * h
            lip_height   = abs(top_lip_y - bottom_lip_y)
            left_corner  = landmarks[61].x * w
            right_corner = landmarks[291].x * w
            lip_width    = abs(right_corner - left_corner)
            aspect_ratio  = lip_height / (lip_width + 1e-6)
            closure_score = 1.0 - min(lip_height / 10.0, 1.0)

            entry = {
                "frame": frame_idx,
                "lip_height": float(lip_height),
                "lip_width": float(lip_width),
                "aspect_ratio": float(aspect_ratio),
                "closure_score": float(closure_score),
            }

            if save_crops and output_crop_dir:
                crop_img, _ = extract_mouth_patch(frame, landmarks)  # np.ndarray BGR
                crop_path = os.path.join(output_crop_dir, f"mouth_{frame_idx:04d}.png")
                cv2.imwrite(crop_path, crop_img)
                entry["crop_path"] = crop_path

            mouth_data.append(entry)
        frame_idx += 1
    cap.release()
    face_mesh.close()
    print(f"[OK] Viseme/geometry in {time.time()-t0:.1f}s (frames={len(mouth_data)})", flush=True)
    return mouth_data


# ----- ArcFace per-frame embeddings -----
def extract_arcface_features(video_path: str) -> List[List[float]]:
    print("[STEP] ArcFace embeddings...", flush=True)
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
    face_app = insightface.app.FaceAnalysis(name='buffalo_l', providers=providers)
    face_app.prepare(ctx_id=0 if torch.cuda.is_available() else -1)

    cap = cv2.VideoCapture(video_path)
    feats = []
    t0 = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        faces = face_app.get(frame)
        # embedding = faces[0].embedding if faces else np.zeros(512)
        embedding = faces[0].embedding if len(faces) > 0 and hasattr(faces[0], "embedding") else np.zeros(512)

        feats.append(embedding.tolist())
    cap.release()
    print(f"[OK] ArcFace in {time.time()-t0:.1f}s (frames={len(feats)})", flush=True)
    return feats


# ----- WhisperX + phonemizer alignment -----
def phonemize_segments(segments: List[Dict]) -> List[Dict]:
    def split_ipa_units(ipa_str: str): return list(ipa_str.strip())
    phone_transcript = []
    for seg in segments:
        ipa_str = phonemize(seg["text"])
        ipa_units = split_ipa_units(ipa_str)
        dur = (seg["end"] - seg["start"]) / max(len(ipa_units), 1)
        for i, p in enumerate(ipa_units):
            p = p.strip()
            if len(p) == 1 and p not in {"<", ">", "", " "}:
                phone_transcript.append({
                    "text": p,
                    "start": seg["start"] + i * dur,
                    "end": seg["start"] + (i + 1) * dur
                })
    return phone_transcript


def align_phonemes(audio, align_model, metadata, device, phone_transcript):
    result = whisperx.align(
        phone_transcript, align_model, metadata, audio, device, return_char_alignments=True
    )
    if not result.get("segments"):
        print("[WARN] No alignment segments returned.")
        return []
    phonemes = []
    for seg in result["segments"]:
        for word in seg.get("words", []):
            if all(k in word for k in ("word", "start", "end")):
                p = word["word"].strip()
                if len(p) == 1 and p not in {"<", ">", "", " "}:
                    phonemes.append({"phoneme": p, "start": word["start"], "end": word["end"]})
    return phonemes


def process_single_video(
    video_path: str,
    output_dir: str,
    label: str = "testvideo",
    fps: float = 25.0,
    whisper_model_name: str = "large-v2",
    language: str = "en",
    compute_type: str = "float32",
):
    device = get_device()
    video_name = Path(video_path).stem
    label_dir = os.path.join(output_dir, label)
    crop_dir  = os.path.join(label_dir, f"{video_name}_mouth_crops")
    os.makedirs(crop_dir, exist_ok=True)

    # 1) Viseme geometry + save crops
    viseme_data = extract_mouth_states(video_path, save_crops=True, output_crop_dir=crop_dir)

    # 2) ArcFace per frame
    arcface_data = extract_arcface_features(video_path)

    # 3) WhisperX models & audio
    print("[STEP] Load WhisperX ASR model...", flush=True)
    t0 = time.time()
    whisper_model = whisperx.load_model(whisper_model_name, device=device, compute_type=compute_type, language=language)
    print(f"[OK] ASR loaded in {time.time()-t0:.1f}s", flush=True)

    print("[STEP] Load align model...", flush=True)
    t0 = time.time()
    # align_model, metadata = whisperx.load_align_model(model_name="facebook/wav2vec2-xlsr-53-espeak-cv-ft", language_code=language, device=device)
    try:
        align_model, metadata = whisperx.load_align_model(
        # model_name="facebook/wav2vec2-large-xlsr-53-espeak-cv-ft",
        model_name="ntsema/wav2vec2-xlsr-53-espeak-cv-ft-xas2-ntsema-colab",
        language_code="en",
        device=device
        )

    except Exception as e:
        print("[WARN] espeak-based aligner unavailable, falling back to default multilingual model.")

        align_model, metadata = whisperx.load_align_model(
        model_name="jonatasgrosman/wav2vec2-large-xlsr-53-english",
        language_code=language,
        device=device)


    print(f"[OK] Align model loaded in {time.time()-t0:.1f}s", flush=True)

    print("[STEP] Read audio from video...", flush=True)
    audio = whisperx.load_audio(video_path)

    print("[STEP] Transcribe...", flush=True)
    t0 = time.time()
    segments = whisper_model.transcribe(audio, batch_size=2)["segments"]
    print(f"[OK] Transcribed in {time.time()-t0:.1f}s (segments={len(segments)})", flush=True)

    print("[STEP] Phonemize + align...", flush=True)
    phone_transcript = phonemize_segments(segments)
    phonemes = align_phonemes(audio, align_model, metadata, device, phone_transcript)

    # 4) Align phonemes to frames
    print("[STEP] Build aligned frames & save JSON...", flush=True)
    num_frames = len(viseme_data)
    frame_ts = [i / fps for i in range(num_frames)]

    def match_phoneme_to_frames(phonemes, timestamps):
        out = []
        for ts in timestamps:
            match = next((p["phoneme"] for p in phonemes if p["start"] <= ts <= p["end"]), "<SIL>")
            out.append(match)
        return out

    phoneme_labels = match_phoneme_to_frames(phonemes, frame_ts)

    aligned = []
    for i in range(num_frames):
        vis = viseme_data[i]
        arc = arcface_data[i] if i < len(arcface_data) else [0.0] * 512
        ph  = phoneme_labels[i].lower()
        closure = vis.get("closure_score", 0.0)
        mismatch = (ph in CLOSURE_PHONEMES) and (closure < CLOSURE_THRESHOLD)
        frame_info = {
            "frame_idx": i,
            "timestamp": frame_ts[i],
            "phoneme": ph,
            "arcface": list(arc),
            "closure_score": float(closure),
            "aspect_ratio": float(vis.get("aspect_ratio", 0.0)),
            "lip_height": float(vis.get("lip_height", 0.0)),
            "lip_width": float(vis.get("lip_width", 0.0)),
            "mismatch": bool(mismatch),
        }
        label_dir = os.path.abspath(label_dir)
        # attach relative crop path if present
        if "crop_path" in vis:
            frame_info["crop_path"] = os.path.relpath(vis["crop_path"], start=label_dir)
        aligned.append(frame_info)

    aligned_path = os.path.join(label_dir, f"{video_name}_aligned.json")
    os.makedirs(label_dir, exist_ok=True)
    with open(aligned_path, "w") as f:
        json.dump(aligned, f, indent=2)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return aligned_path, label_dir


# ----- Build tensors from DISK (uses crop PNGs) -----
_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

def build_single_video_tensors_fromdisk(
    aligned_json_path: str,
    label_dir: str,
    phoneme_vocab: List[str],
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with open(aligned_json_path, "r") as f:
        aligned = json.load(f)

    P, F = len(phoneme_vocab), IMAGES_PER_PHON
    H, W = IMAGE_SIZE

    geoms = torch.zeros(1, P, F, 1, device=device)
    imgs  = torch.zeros(1, P, F, 3, H, W, device=device)
    arcs  = torch.zeros(1, P, F, 512, device=device)
    mask  = torch.zeros(1, P, F, dtype=torch.bool, device=device)

    by_ph = {p: [] for p in phoneme_vocab}
    for e in aligned:
        p = e.get("phoneme", "").lower().strip()
        if p in KEEP_PHONEMES and p not in IGNORED_PHONEMES:
            by_ph[p].append(e)

    for pi, p in enumerate(phoneme_vocab):
        entries = by_ph.get(p, [])[:F]
        for fi, e in enumerate(entries):
            geoms[0, pi, fi, 0] = float(e.get("aspect_ratio", 0.0))
            vec = e.get("arcface", None)
            if isinstance(vec, list) and len(vec) == 512:
                arcs[0, pi, fi] = torch.tensor(vec, device=device, dtype=torch.float32)

            crop_rel = e.get("crop_path", "")
            crop_abs = os.path.join(label_dir, crop_rel) if crop_rel else ""
            if crop_abs and os.path.exists(crop_abs):
                pil = Image.open(crop_abs).convert("RGB")
                imgs[0, pi, fi] = _transform(pil).to(device)
                mask[0, pi, fi] = True

    return geoms, imgs, arcs, mask


def default_phoneme_vocab() -> List[str]:
    return sorted(list(KEEP_PHONEMES))
