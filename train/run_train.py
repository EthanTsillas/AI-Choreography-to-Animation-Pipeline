import subprocess
import sys
import os


TRAIN_DATA_DIR = "img" 
OUTPUT_DIR = "model"
OUTPUT_NAME = "Ratman_v1"


accelerate_path = os.path.join(os.path.dirname(sys.executable), "Scripts", "accelerate.exe")


train_script_path = r"sd-scripts\train_network.py"

train_cmd = [
    accelerate_path, "launch", train_script_path,
    "--pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5",
    f"--train_data_dir={TRAIN_DATA_DIR}",
    f"--output_dir={OUTPUT_DIR}",
    f"--output_name={OUTPUT_NAME}",
    "--resolution=768,768",      
    "--enable_bucket", 
    "--min_bucket_reso=512",
    "--max_bucket_reso=768",
    "--bucket_reso_steps=64",
    "--save_model_as=safetensors",
    "--caption_extension=.txt",
    "--network_module=networks.lora",
    "--network_dim=64",
    "--network_alpha=32",
    "--train_batch_size=1",
    "--learning_rate=1e-4",
    "--lr_scheduler=cosine",
    "--mixed_precision=fp16",
    "--save_every_n_epochs=1",
    "--max_train_epochs=10",
    "--optimizer_type=AdamW8bit",
    "--sdpa",
    "--gradient_checkpointing"
]

print("Starting Training...")


subprocess.run(train_cmd, shell=True)