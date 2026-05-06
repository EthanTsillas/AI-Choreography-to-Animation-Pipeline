import os
import torch
from PIL import Image, ImageFilter
from rembg import remove, new_session
from FFmpeg.FFmpeg_frames_to_video import get_video


def process_backgrounds_v2(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # BiRefNet is used for high-accuracy segmentation
    model_name = "birefnet-general" 
    session = new_session(model_name)

    files = sorted([f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg'))])
    print(f"Waiting {model_name} for high-precision removal...")

    for filename in files:
        img = Image.open(os.path.join(input_folder, filename)).convert("RGBA")
        
        no_bg = remove(
            img, 
            session=session,
            alpha_matting=True,
            post_process_mask=True 
        )

        # Create the black background
        black_bg = Image.new("RGB", no_bg.size, (0, 0, 0))
        black_bg.paste(no_bg, (0, 0), mask=no_bg)

        black_bg.save(os.path.join(output_folder, filename))
        print(f"Processed {filename}")
        
    get_video("black_bg_frames")


if __name__ == "__main__":
    process_backgrounds_v2("generated_frames", "black_bg_frames")