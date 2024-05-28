import argparse
import cv2
import mss
import numpy as np
import signal
import time
import torch

from PIL import Image
from pytube import YouTube
from threading import Thread
from transformers import TextIteratorStreamer, AutoTokenizer, AutoModelForCausalLM

try:
    from moondream import detect_device, LATEST_REVISION
except ImportError:
    def detect_device():
        if torch.cuda.is_available():
            return torch.device("cuda"), torch.float16
        if torch.backends.mps.is_available():
            return torch.device("mps"), torch.float16
        return torch.device("cpu"), torch.float32

    LATEST_REVISION = "main"

parser = argparse.ArgumentParser()
parser.add_argument("--cpu", action="store_true")
parser.add_argument("--mode", choices=["screenshot", "webcam", "youtube"], required=True,
                    help="Capture mode: 'screenshot', 'webcam' or 'youtube'")
parser.add_argument("--url", type=str, help="YouTube video URL (required for youtube mode)")
args = parser.parse_args()  # Corrected function name

if args.cpu:
    device = torch.device("cpu")
    dtype = torch.float32
else:
    device, dtype = detect_device()
    if device != torch.device("cpu"):
        print("Using device:", device)
        print("If you run into issues, pass the `--cpu` flag to this script.")
        print()

model_id = "vikhyatk/moondream2"
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=LATEST_REVISION)
moondream = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=LATEST_REVISION
).to(device=device, dtype=dtype)
moondream.eval()


def cleanup(camera=None):
    if camera and camera.isOpened():
        camera.release()
    cv2.destroyAllWindows()
    print("Cleanup complete.")


def signal_handler(sig, frame):
    cleanup()
    exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def answer_question(img, prompt):
    image_embeds = moondream.encode_image(img)
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
    thread = Thread(
        target=moondream.answer_question,
        kwargs={
            "image_embeds": image_embeds,
            "question": prompt,
            "tokenizer": tokenizer,
            "streamer": streamer,
        },
    )
    thread.start()
    thread.join()

    buffer = ''.join(streamer)
    return buffer.strip()


def capture_screenshot(sct, monitor):
    screenshot = sct.grab(monitor)
    return Image.frombytes("RGB", (screenshot.width, screenshot.height), screenshot.rgb)


def capture_webcam_frame(cap):
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Could not read frame.")
        return None
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def download_youtube_video(url):
    yt = YouTube(url)
    stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    return stream.download()


def extract_frames(video_path, frame_interval=10):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    extracted_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            extracted_frames.append(img)
        frame_count += 1
    cap.release()
    return extracted_frames


def main():
    frame_count = 0
    prompt = "What's going on? Respond with a single sentence."
    capture_interval = 5

    if args.mode == "screenshot":
        with mss.mss() as sct:
            monitor = sct.monitors[0]
            try:
                while True:
                    start_time = time.time()
                    img = capture_screenshot(sct, monitor)

                    frame_count += 1
                    width, height = img.size
                    print(f"📷 Captured frame {frame_count}: resolution {width}x{height}")

                    response = answer_question(img, prompt)
                    print(f"🌙 Moondream analysis: {response}")

                    elapsed_time = time.time() - start_time
                    time.sleep(max(0, capture_interval - elapsed_time))
            finally:
                cleanup()

    elif args.mode == "webcam":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Could not open webcam.")
            return

        try:
            while True:
                start_time = time.time()
                img = capture_webcam_frame(cap)

                if img is None:
                    break

                frame_count += 1
                width, height = img.size
                print(f"📷 Captured frame {frame_count}: resolution {width}x{height}")

                response = answer_question(img, prompt)
                print(f"🌙 Moondream analysis: {response}")

                elapsed_time = time.time() - start_time
                time.sleep(max(0, capture_interval - elapsed_time))
        finally:
            cleanup(cap)


    elif args.mode == "webcam":

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ Error: Could not open webcam.")

            return

        try:

            while True:
                start_time = time.time()
                img = capture_webcam_frame(cap)

                if img is None:
                    break

                frame_count += 1
                width, height = img.size
                print(f"📷 Captured frame {frame_count}: resolution {width}x{height}")

                response = answer_question(img, prompt)
                print(f"🌙 Moondream analysis: {response}")

                elapsed_time = time.time() - start_time
                time.sleep(max(0, capture_interval - elapsed_time))
        finally:
            cleanup(cap)


    elif args.mode == "youtube":

        if not args.url:
            print("❌ Error: YouTube URL is required for youtube mode.")

            return

        video_path = download_youtube_video(args.url)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps)
        cap.release()

        frames = extract_frames(video_path, frame_interval)

        try:

            for frame_count, img in enumerate(frames, start=1):
                start_time = time.time()

                width, height = img.size
                print(f"📷 Captured frame {frame_count}: resolution {width}x{height}")

                response = answer_question(img, prompt)
                print(f"🌙 Moondream analysis: {response}")

                elapsed_time = time.time() - start_time
                time.sleep(max(0, capture_interval - elapsed_time))
        finally:
            cleanup()


if __name__ == "__main__":
    main()
