from diffusers import DDIMScheduler, DDIMInverseScheduler
from diffusers import ControlNetModel
from diffusers.utils import make_image_grid
from pipeline_controlnet import StableDiffusionControlNetPipeline
from PIL import Image
import torch

controlnet = ControlNetModel.from_pretrained("thibaud/controlnet-sd21-scribble-diffusers", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base",
    controlnet=controlnet,
    torch_dtype=torch.float16, use_safetensors=True
).to("cuda")

sketch = Image.open("/data2/kietngt00/score-distillation-via-inversion/data/sketch/a_baby_penguin_wearing_a_blue_hat.png").convert('RGB').resize((512, 512))

output = pipe(
    "a baby penguin wearing a hat and a shirt", image=sketch
).images[0]