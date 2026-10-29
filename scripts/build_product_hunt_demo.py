from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "product-hunt"
OUTPUT = ASSET_DIR / "btc2g-demo-28s.mp4"
WIDTH, HEIGHT = 1270, 760
FPS = 24
SECONDS_PER_SCENE = 7
FADE_FRAMES = 12

SCENES = [
    ("01-home.jpg", "Bitcoin versus gold in one calm daily dashboard"),
    ("02-forecast.jpg", "Historical prices and clearly labelled model scenarios"),
    ("03-calculator.jpg", "A reference calculator with source, date, and limitations"),
    ("04-premium.jpg", "A Premium research roadmap without trading promises"),
]


def fitted_frame(image: np.ndarray, progress: float) -> np.ndarray:
    image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    scale = 1.0 + 0.025 * progress
    scaled = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    x = (scaled.shape[1] - WIDTH) // 2
    y = (scaled.shape[0] - HEIGHT) // 2
    return scaled[y : y + HEIGHT, x : x + WIDTH].copy()


def add_caption(frame: np.ndarray, caption: str) -> np.ndarray:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, HEIGHT - 92), (WIDTH, HEIGHT), (4, 15, 29), -1)
    frame = cv2.addWeighted(overlay, 0.86, frame, 0.14, 0)
    cv2.putText(
        frame,
        caption,
        (48, HEIGHT - 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (245, 247, 250),
        2,
        cv2.LINE_AA,
    )
    return frame


def main() -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT), fourcc, FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not initialize the MP4 writer")

    previous = None
    frames_per_scene = FPS * SECONDS_PER_SCENE
    try:
        for filename, caption in SCENES:
            image = cv2.imread(str(ASSET_DIR / filename), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(ASSET_DIR / filename)

            for frame_number in range(frames_per_scene):
                progress = frame_number / max(frames_per_scene - 1, 1)
                frame = add_caption(fitted_frame(image, progress), caption)
                if previous is not None and frame_number < FADE_FRAMES:
                    alpha = (frame_number + 1) / (FADE_FRAMES + 1)
                    frame = cv2.addWeighted(previous, 1.0 - alpha, frame, alpha, 0)
                writer.write(frame)
                previous = frame
    finally:
        writer.release()

    capture = cv2.VideoCapture(str(OUTPUT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / FPS
    capture.release()
    if frame_count != FPS * SECONDS_PER_SCENE * len(SCENES):
        raise RuntimeError(f"Unexpected frame count: {frame_count}")
    print(f"Created {OUTPUT} ({duration:.1f}s, {WIDTH}x{HEIGHT}, {FPS}fps)")


if __name__ == "__main__":
    main()
