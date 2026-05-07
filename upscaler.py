import os
from PIL import Image
from FFmpeg.FFmpeg_frames_to_video import get_video

def batch_upscale(input_folder, output_folder, size=2048):
    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Common image extensions to look for
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(valid_extensions):
            img_path = os.path.join(input_folder, filename)
            
            try:
                with Image.open(img_path) as img:
                    # Ensure image is in RGB (avoids issues with transparency/palettes)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    # Your specific resizing logic
                    final_img = img.resize((size, size), Image.Resampling.LANCZOS)
                    
                    # Construct the final path and save
                    save_path = os.path.join(output_folder, filename)
                    final_img.save(save_path, quality=95)
                    print(f"Processed: {filename}")
                    
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    source_dir = 'black_bg_frames'
    target_dir = 'upscaled_images'
    # ---------------------

    batch_upscale(source_dir, target_dir)
    get_video(target_dir)
    print("\nProcessing complete!")