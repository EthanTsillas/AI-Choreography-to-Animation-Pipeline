import os
from PIL import Image
from controlnet_aux import LineartDetector

def get_lineart():
    Width = 768
    Height = 768

    lineart_detector = LineartDetector.from_pretrained("lllyasviel/Annotators").to("cuda")

    source_dir = "FFmpeg/FFmpeg Images"
    lineart_dir = "lineart_frames"
    os.makedirs(lineart_dir, exist_ok=True)


    # Generate lineart from the original extracted frames
    source_dir = "FFmpeg/FFmpeg Images"
    lineart_dir = "lineart_frames"
    os.makedirs(lineart_dir, exist_ok=True)

    source_files = sorted([f for f in os.listdir(source_dir)])
    for filename in source_files:
        img = Image.open(os.path.join(source_dir, filename)).convert("RGB")
        img = img.resize((Width, Height))
        lineart = lineart_detector(img, coarse=False)  # coarse=True for thicker lines
        lineart.save(os.path.join(lineart_dir, filename))

    print(f"Generated {len(source_files)} lineart frames")

if __name__ == "__main__":
    get_lineart()