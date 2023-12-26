# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import yaml
import torch
import subprocess
from diffusers import DDIMScheduler, VideoToVideoSDPipeline

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        model = VideoToVideoSDPipeline.from_pretrained("cerspense/zeroscope_v2_576w", torch_dtype=torch.float16)
        os.chdir("motion")

    def predict(
        self,
        video: Path = Input(description="Input video"),
        video_prompt: str = Input(
            description="Prompt used to describe the input video",
            default="Amazing quality, masterpiece, A locomotive rides in a forest",
        ),
        n_timesteps: int = Input(
            description="Number of times to preprocess the video",
            default=999,
        ),
        target_prompt: str = Input(
            description="Prompt used to describe the output video",
            default="Amazing quality, masterpiece, A motorbike driving in a forest",
        ),
        negative_prompt: str = Input(
            description="Negative prompt to use for editing",
            default="bad quality, distortions, unrealistic, distorted image, watermark, signature"
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        print(f"Using seed: {seed}")

        # Clear tmp input & output images folder
        input_path = "data/input"
        # os.system("rm -rf " + input_path)
        os.makedirs(input_path, exist_ok=True)

        # Convert input video to images
        print("Converting input video to images")
        command = ["ffmpeg", "-i", str(video), input_path+"/%05d.png"]
        subprocess.run(command, check=True)

        latent_path = "data/input/ddim_latents"
        config_data = f"""video_path: {input_path}
save_dir: {latent_path}
max_number_of_frames: 24

n_timesteps: {n_timesteps}
prompt: {video_prompt}
negative_prompt: ""
save_ddim_reconstruction: False
        """
        with open("configs/config.yaml", "w") as file:
            file.write(config_data)

        # Run preprocessor script
        print("Running preprocessor script")
        command = ["python", "preprocess_video_ddim.py", "--config_path", "configs/config.yaml"]
        subprocess.run(command)

        output_path = "result/output"
        # Edit editing script
        with open('configs/guidance_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        # Change the output_path value
        config['output_path'] = output_path
        config['data_path'] = input_path
        config['latents_path'] = latent_path
        config['source_prompt'] = video_prompt
        config['target_prompt'] = target_prompt
        config['negative_prompt'] = negative_prompt
        config['seed'] = seed
        # Write the modified configuration back to the YAML file
        with open('configs/guidance.yaml', 'w') as f:
            yaml.dump(config, f)

        # Clear result output folder
        
        # os.system("rm -rf result")
        # Run editing script
        print("Running editing script")
        command = ["python", "run.py", "--config_path", "configs/guidance.yaml"]
        subprocess.run(command)

        output_path = "/src/motion/result/output/result.mp4"
        # if file exists return path
        if os.path.isfile(output_path):
            return Path(output_path)
        else:
            raise RuntimeError("Error fetching output video")