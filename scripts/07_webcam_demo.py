"""
treshold 0.47 olarak belirlendi CelebA datasetine göre
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.dataset import build_transform
from src.eval_utils import load_checkpoint_model
from src.face_crop import crop_with_margin


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Egitilmis checkpoint'i webcam ile canli test eder.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.4724,
                         help="Spoof karari icin sabit esik (varsayilan: internal val EER esigi, "
                              "bkz. docs/generalization_report.md).")
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument("--margin", type=float, default=0.2,
                         help="Yuz kutusuna eklenecek pay orani (egitimdeki preprocess_face ile ayni varsayilan).")
    parser.add_argument("--infer_every", type=int, default=1,
                         help="Her N karede bir inference calistir (CPU'da akiciligi artirmak icin >1 verin).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print(f"Checkpoint yukleniyor: {args.checkpoint}")
    model, ckpt = load_checkpoint_model(args.checkpoint, device)
    print(f"unfreeze_blocks={ckpt['args'].get('unfreeze_blocks', 0)}, esik={args.threshold:.4f}, cihaz={device}")

    transform = build_transform(train=False)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Kamera acilamadi (index={args.camera_index}).")

    print("Canli demo basladi. Cikmak icin 'q' tusuna basin.")

    frame_idx = 0
    last_label, last_score, last_bbox = None, None, None

    fps, inference_fps = 0.0, 0.0
    fps_window_start = time.time()
    fps_frame_count, fps_inference_count = 0, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Kare okunamadi, cikiliyor.")
            break
        fps_frame_count += 1

        frame = cv2.flip(frame, 1)  # ayna gorunumu icin
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])  # en buyuk (alan) yuz
            last_bbox = (x, y, x + w, y + h)

            if frame_idx % args.infer_every == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                cropped = crop_with_margin(pil_img, last_bbox, margin=args.margin)
                if cropped.width > 0 and cropped.height > 0:
                    face_resized = cropped.resize((224, 224), Image.BILINEAR)
                    tensor = transform(face_resized).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = model(tensor)
                        score = torch.softmax(logits, dim=1)[0, 1].item()
                    last_score = score
                    last_label = "SPOOF" if score >= args.threshold else "LIVE"
                    fps_inference_count += 1
        else:
            last_bbox = None

        if last_bbox is not None:
            x1, y1, x2, y2 = last_bbox
            color = (0, 0, 255) if last_label == "SPOOF" else (0, 200, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if last_score is not None:
                cv2.putText(frame, f"{last_label} ({last_score:.2%})", (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            cv2.putText(frame, "Yuz bulunamadi", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        elapsed = time.time() - fps_window_start
        if elapsed >= 1.0:
            fps = fps_frame_count / elapsed
            inference_fps = fps_inference_count / elapsed
            fps_frame_count, fps_inference_count = 0, 0
            fps_window_start = time.time()

        cv2.putText(frame, f"FPS: {fps:.1f}  Inference/s: {inference_fps:.1f}",
                    (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Passive Liveness Demo (q = cikis)", frame)
        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
