from __future__ import annotations

import argparse
import asyncio
import subprocess
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "product-hunt"
SILENT_OUTPUT = ASSET_DIR / "btc2g-demo-motion-silent.mp4"
VOICE_OUTPUT = ASSET_DIR / "btc2g-demo-voice.mp3"
OUTPUT = ASSET_DIR / "btc2g-demo-42s-female.mp4"
REFERENCE_MUSIC = ROOT.parent / "Voice" / "___Simona____.m4a"
WIDTH, HEIGHT = 1270, 760
FPS = 30
FADE_FRAMES = 24
INTRO_SECONDS = 3.6
VOICE_DELAY_MS = 2200
VOICE_NAME = "en-US-AvaMultilingualNeural"

VOICEOVER = (
    "BTC2G answers one simple question: how much gold does one Bitcoin represent? "
    "The dashboard shows the published ratio, the seven-day move, and market date at a glance. "
    "Explore historical Bitcoin and gold data, plus clearly labelled long-term scenarios, not price promises. "
    "Use the reference calculator to convert Bitcoin into estimated troy ounces, with sources and limitations beside the result. "
    "Premium is an early-access research roadmap for deeper history, model notes, and weekly context. "
    "BTC2G: Bitcoin versus gold, with the uncertainty left in."
)

SCENES = [
    {
        "file": "01-home.jpg",
        "label": "DAILY DASHBOARD",
        "caption": "Published ratio, 7-day move, and market date",
        "start": (0.50, 0.48, 1.00),
        "end": (0.69, 0.37, 1.12),
        "cursor": ((0.79, 0.36), (0.72, 0.48)),
        "seconds": 12,
    },
    {
        "file": "02-forecast.jpg",
        "label": "HISTORY + SCENARIOS",
        "caption": "Historical prices with clearly labelled extrapolations",
        "start": (0.50, 0.50, 1.02),
        "end": (0.66, 0.42, 1.13),
        "cursor": ((0.28, 0.66), (0.74, 0.35)),
        "seconds": 10,
    },
    {
        "file": "03-calculator.jpg",
        "label": "REFERENCE CALCULATOR",
        "caption": "Convert BTC to gold with sources and limitations visible",
        "start": (0.50, 0.43, 1.02),
        "end": (0.52, 0.54, 1.12),
        "cursor": ((0.47, 0.43), (0.63, 0.58)),
        "seconds": 10,
    },
    {
        "file": "04-premium.jpg",
        "label": "PREMIUM EARLY ACCESS",
        "caption": "A research roadmap for deeper history and weekly context",
        "start": (0.50, 0.50, 1.00),
        "end": (0.36, 0.54, 1.10),
        "cursor": ((0.82, 0.52), (0.20, 0.78)),
        "seconds": 10,
    },
]
TOTAL_DURATION = sum(scene["seconds"] for scene in SCENES)


def ease(value: float) -> float:
    return value * value * (3.0 - 2.0 * value)


def interpolate(start: float, end: float, progress: float) -> float:
    return start + (end - start) * ease(progress)


def motion_frame(image: np.ndarray, scene: dict, progress: float) -> np.ndarray:
    source = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    sx, sy, start_scale = scene["start"]
    ex, ey, end_scale = scene["end"]
    scale = interpolate(start_scale, end_scale, progress)
    center_x = interpolate(sx, ex, progress) * WIDTH
    center_y = interpolate(sy, ey, progress) * HEIGHT
    crop_width = WIDTH / scale
    crop_height = HEIGHT / scale
    left = int(np.clip(center_x - crop_width / 2, 0, WIDTH - crop_width))
    top = int(np.clip(center_y - crop_height / 2, 0, HEIGHT - crop_height))
    right = min(WIDTH, int(left + crop_width))
    bottom = min(HEIGHT, int(top + crop_height))
    crop = source[top:bottom, left:right]
    return cv2.resize(crop, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)


def add_overlays(frame: np.ndarray, scene: dict, progress: float, overall: float) -> np.ndarray:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (WIDTH, 82), (4, 15, 29), -1)
    cv2.rectangle(overlay, (0, HEIGHT - 98), (WIDTH, HEIGHT), (4, 15, 29), -1)
    frame = cv2.addWeighted(overlay, 0.88, frame, 0.12, 0)
    cv2.putText(frame, "BTC2G", (42, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.075, (55, 175, 212), 3, cv2.LINE_AA)
    cv2.putText(frame, scene["label"], (WIDTH - 365, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (212, 175, 55), 2, cv2.LINE_AA)
    cv2.putText(frame, scene["caption"], (46, HEIGHT - 43), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 247, 250), 2, cv2.LINE_AA)
    cv2.rectangle(frame, (0, HEIGHT - 6), (int(WIDTH * overall), HEIGHT), (247, 147, 26), -1)

    (start_x, start_y), (end_x, end_y) = scene["cursor"]
    cursor_x = int(interpolate(start_x, end_x, progress) * WIDTH)
    cursor_y = int(interpolate(start_y, end_y, progress) * HEIGHT)
    pulse = 8 + int(4 * abs(np.sin(progress * np.pi * 5)))
    cv2.circle(frame, (cursor_x, cursor_y), pulse, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, (cursor_x, cursor_y), 3, (247, 147, 26), -1, cv2.LINE_AA)
    return frame


def transition_frame(previous: np.ndarray, current: np.ndarray, alpha: float) -> np.ndarray:
    """Blend scenes with a soft zoom blur and a diagonal light sweep."""
    eased = ease(alpha)
    blur_strength = int(2 + 14 * np.sin(np.pi * eased))
    if blur_strength % 2 == 0:
        blur_strength += 1
    outgoing = cv2.GaussianBlur(previous, (blur_strength, blur_strength), 0)
    incoming = cv2.GaussianBlur(current, (blur_strength, blur_strength), 0)
    blended = cv2.addWeighted(outgoing, 1.0 - eased, incoming, eased, 0)

    sweep = np.zeros_like(blended)
    center = int((-0.25 + 1.5 * eased) * WIDTH)
    points = np.array(
        [[center - 95, 0], [center + 5, 0], [center + 145, HEIGHT], [center + 45, HEIGHT]],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(sweep, points, (255, 238, 200))
    glow = 0.16 * np.sin(np.pi * eased)
    return cv2.addWeighted(blended, 1.0, sweep, glow, 0)


def logo_intro_frame(home_frame: np.ndarray, logo: np.ndarray, elapsed: float) -> np.ndarray:
    """Show the BTC2G mark, then dissolve it into the live dashboard."""
    y = np.linspace(0, 1, HEIGHT, dtype=np.float32)[:, None]
    top = np.array([75, 44, 19], dtype=np.float32)
    bottom = np.array([29, 15, 4], dtype=np.float32)
    background = np.repeat(((1 - y[..., None]) * top + y[..., None] * bottom), WIDTH, axis=1).astype(np.uint8)
    content = background.copy()

    logo_size = 413
    logo_square = cv2.resize(logo, (logo_size, logo_size), interpolation=cv2.INTER_AREA)
    left = (WIDTH - logo_size) // 2
    top_y = 62
    opacity = min(1.0, elapsed / 0.8)
    if elapsed > 2.2:
        opacity = max(0.0, 1.0 - (elapsed - 2.2) / (INTRO_SECONDS - 2.2))
    content[top_y : top_y + logo_size, left : left + logo_size] = logo_square
    title = "BTC2G"
    title_scale = 1.85
    title_thickness = 4
    title_width = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, title_scale, title_thickness)[0][0]
    cv2.putText(
        content,
        title,
        ((WIDTH - title_width) // 2, 550),
        cv2.FONT_HERSHEY_SIMPLEX,
        title_scale,
        (55, 175, 212),
        title_thickness,
        cv2.LINE_AA,
    )
    slogan = "From era of gold to era of crypto."
    slogan_scale = 0.81
    slogan_thickness = 2
    slogan_width = cv2.getTextSize(slogan, cv2.FONT_HERSHEY_SIMPLEX, slogan_scale, slogan_thickness)[0][0]
    cv2.putText(
        content,
        slogan,
        ((WIDTH - slogan_width) // 2, 620),
        cv2.FONT_HERSHEY_SIMPLEX,
        slogan_scale,
        (247, 247, 250),
        slogan_thickness,
        cv2.LINE_AA,
    )
    intro = cv2.addWeighted(background, 1.0 - opacity, content, opacity, 0)
    if elapsed <= 2.2:
        return intro
    reveal = ease(min(1.0, (elapsed - 2.2) / (INTRO_SECONDS - 2.2)))
    return cv2.addWeighted(intro, 1.0 - reveal, home_frame, reveal, 0)


def build_motion_video() -> None:
    writer = cv2.VideoWriter(str(SILENT_OUTPUT), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not initialize the MP4 writer")
    previous = None
    total_frames = FPS * TOTAL_DURATION
    logo = cv2.imread(str(ASSET_DIR / "btc2g-logo-512.png"), cv2.IMREAD_COLOR)
    if logo is None:
        raise FileNotFoundError(ASSET_DIR / "btc2g-logo-512.png")
    written = 0
    try:
        for scene_index, scene in enumerate(SCENES):
            frames_per_scene = FPS * scene["seconds"]
            image = cv2.imread(str(ASSET_DIR / scene["file"]), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(ASSET_DIR / scene["file"])
            for frame_number in range(frames_per_scene):
                elapsed = frame_number / FPS
                if scene_index == 0:
                    active_frame = max(0, frame_number - int(INTRO_SECONDS * FPS))
                    active_total = max(1, frames_per_scene - int(INTRO_SECONDS * FPS) - 1)
                    progress = min(1.0, active_frame / active_total)
                else:
                    progress = frame_number / max(frames_per_scene - 1, 1)
                frame = add_overlays(motion_frame(image, scene, progress), scene, progress, (written + 1) / total_frames)
                if scene_index == 0 and elapsed < INTRO_SECONDS:
                    frame = logo_intro_frame(frame, logo, elapsed)
                if scene_index > 0 and previous is not None and frame_number < FADE_FRAMES:
                    alpha = (frame_number + 1) / (FADE_FRAMES + 1)
                    frame = transition_frame(previous, frame, alpha)
                writer.write(frame)
                previous = frame
                written += 1
    finally:
        writer.release()


async def build_voiceover() -> None:
    import edge_tts

    speech = edge_tts.Communicate(VOICEOVER, VOICE_NAME, rate="+0%", pitch="+1Hz", volume="+0%")
    await speech.save(str(VOICE_OUTPUT))


def mux_audio(music_path: Path) -> None:
    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise RuntimeError("Install imageio-ffmpeg to create the narrated MP4") from error
    if not music_path.exists():
        raise FileNotFoundError(f"Reference music not found: {music_path}")
    duration = TOTAL_DURATION
    audio_filter = (
        f"[2:a]atrim=0:{duration},asetpts=PTS-STARTPTS,volume=0.333,"
        "aformat=channel_layouts=stereo[music];"
        f"[1:a]adelay={VOICE_DELAY_MS}|{VOICE_DELAY_MS},volume=3.10,highpass=f=95,"
        "acompressor=threshold=-18dB:ratio=2.5:attack=10:release=220,"
        "aformat=channel_layouts=stereo[voice];"
        "[music][voice]amix=inputs=2:duration=longest:dropout_transition=2:normalize=0,"
        "alimiter=limit=0.95[audio]"
    )
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(SILENT_OUTPUT), "-i", str(VOICE_OUTPUT),
            "-ss", "112", "-t", str(duration), "-i", str(music_path),
            "-filter_complex", audio_filter, "-map", "0:v:0", "-map", "[audio]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-c:a", "aac", "-b:a", "192k",
            "-ar", "48000",
            "-t", str(duration), "-movflags", "+faststart", str(OUTPUT),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the narrated BTC2G Product Hunt demo")
    parser.add_argument("--keep-intermediates", action="store_true")
    parser.add_argument("--music", type=Path, default=REFERENCE_MUSIC)
    args = parser.parse_args()
    build_motion_video()
    asyncio.run(build_voiceover())
    mux_audio(args.music)
    if not args.keep_intermediates:
        SILENT_OUTPUT.unlink(missing_ok=True)
        VOICE_OUTPUT.unlink(missing_ok=True)
    print(f"Created {OUTPUT} ({TOTAL_DURATION}s, {WIDTH}x{HEIGHT}, {FPS}fps, narrated)")


if __name__ == "__main__":
    main()
