import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Configuration
IMAGE_FOLDER = "img/5_ratman"
PREFIX = "ratman, "

# Load the BLIP model (if its not downloaded it will automatically)
print("Loading BLIP model...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")

def caption_images():
    for filename in os.listdir(IMAGE_FOLDER):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(IMAGE_FOLDER, filename)
            txt_path = os.path.splitext(img_path)[0] + ".txt"
            
            # Process Image
            raw_image = Image.open(img_path).convert('RGB')
            inputs = processor(raw_image, return_tensors="pt").to("cuda")
            
            # Generate Caption
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)
            
            # Save with Prefix
            with open(txt_path, "w") as f:
                f.write(f"{PREFIX}{caption}")
                
            print(f"Generated: {filename}.txt")

if __name__ == "__main__":
    caption_images()
    print("✅ Done! All 53 images have been captioned.")