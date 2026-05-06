import os
import subprocess
import json
from PIL import Image

def get_video_duration(video_path):
    command = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        video_path
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    data = json.loads(result.stdout)
    return float(data["format"]["duration"])

def extract_frames(video_path, num_frames):
    output_dir = "FFmpeg/FFmpeg Images"
    os.makedirs(output_dir, exist_ok=True)

    duration = get_video_duration(video_path)

    # calculate FPS needed to get exact number of frames
    fps = num_frames / duration

    output_pattern = os.path.join(output_dir, "frame_%04d.png")

    command = [
        "ffmpeg",
        "-i", video_path,
        #"-vf", f"fps={fps}",  # if you want all the frames just remove this line -> "-vf", f"fps={fps}"
        output_pattern
    ]

    try:
        subprocess.run(command, check=True)
        print(f"✅ Extracted {num_frames} frames into '{output_dir}'")
    except subprocess.CalledProcessError as e:
        print("❌ Error while extracting frames:", e)

def get_frames(video_name):
    extract_frames(video_name, num_frames=12)

    square_and_resize(
        "FFmpeg/FFmpeg Images",
        "FFmpeg/FFmpeg Images"  # overwrite in place
    )

def square_and_resize(input_dir, output_dir, size=768):
    os.makedirs(output_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(input_dir) if f.endswith(".png")])

    for file in files:
        path = os.path.join(input_dir, file)
        img = Image.open(path).convert("RGB")
        w, h = img.size

        # CASE 1: Image is wider than it is tall (Landscape video frame)
        # It likely already has black bars on the sides.
        # We crop the center horizontally so the height dictates the square.
        if w > h:
            left = (w - h) // 2
            # Crop bounds: (left, top, right, bottom)
            img = img.crop((left, 0, left + h, h))

        # CASE 2: Image is taller than it is wide (True portrait)
        # We add black bars to the left and right sides to make it a square.
        elif h > w:
            square_img = Image.new("RGB", (h, h), (0, 0, 0))
            upper_left_x = (h - w) // 2
            # Paste at Y=0 so there are no top/bottom bars
            square_img.paste(img, (upper_left_x, 0))
            img = square_img
            
        # (If w == h, it is already a perfect square, so we do nothing here)

        # 3. Final resize
        # The image is now a perfect square bounding the exact top and bottom of the frame
        final_img = img.resize((size, size), Image.Resampling.LANCZOS)
        final_img.save(os.path.join(output_dir, file))

    print(f"✅ Frames perfectly squared to {size}x{size} (Zero top/bottom bars)")

if __name__ == "__main__":
    get_frames("FFmpeg/videos/input.mp4")