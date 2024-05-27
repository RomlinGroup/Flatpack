import argparse
import cv2
import numpy as np
import signal
import time
import torch

from PIL import Image
from threading import Thread
from transformers import TextIteratorStreamer, AutoTokenizer, AutoModelForCausalLM

# Attempt to import detect_device and LATEST_REVISION from moondream
try:
    from moondream import detect_device, LATEST_REVISION
except ImportError:
    # Define a custom detect_device function if import fails
    def detect_device():
        if torch.cuda.is_available():
            return torch.device("cuda"), torch.float16
        else:
            return torch.device("cpu"), torch.float32


    LATEST_REVISION = "main"

# Argument parser for CPU option
parser = argparse.ArgumentParser()
parser.add_argument("--cpu", action="store_true")
args = parser.parse_args()

# Determine device
if args.cpu:
    device = torch.device("cpu")
    dtype = torch.float32
else:
    device, dtype = detect_device()
    if device != torch.device("cpu"):
        print("Using device:", device)
        print("If you run into issues, pass the `--cpu` flag to this script.")
        print()

# Load model and tokenizer
model_id = "vikhyatk/moondream2"
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=LATEST_REVISION)
moondream = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=LATEST_REVISION
).to(device=device, dtype=dtype)
moondream.eval()

# Initialize the webcam
cap = cv2.VideoCapture(0)


def cleanup(cap):
    """Release the webcam and close all OpenCV windows."""
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    print("Webcam and OpenCV windows have been released cleanly.")


def signal_handler(signum, frame, cap):
    cleanup(cap)
    exit(0)


# Register signal handler for graceful termination
signal.signal(signal.SIGINT, lambda signum, frame: signal_handler(signum, frame, cap))
signal.signal(signal.SIGTERM, lambda signum, frame: signal_handler(signum, frame, cap))


def answer_question(img, prompt):
    # Convert frame to PIL image
    pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    image_embeds = moondream.encode_image(pil_image)
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
    thread.join()  # Wait for the thread to complete

    buffer = ""
    for new_text in streamer:
        buffer += new_text
    return buffer.strip()


def main():
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    frame_count = 0
    prompt = "What's going on? Respond with a single sentence."

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            frame_count += 1
            height, width, _ = frame.shape
            print(f"Captured frame {frame_count}: resolution {width}x{height}")

            response = answer_question(frame, prompt)
            print(f"Moondream analysis: {response}")

            time.sleep(5)  # Capture one frame every fifth second
    finally:
        cleanup(cap)


if __name__ == "__main__":
    main()
