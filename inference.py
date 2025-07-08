# inference.py

import torch
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from insightface.app import FaceAnalysis
import cv2
import numpy as np
import os

# Load face analysis (CPU-only for Streamlit Cloud)
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(512, 512))

# Load ControlNet (face guidance)
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose",
    torch_dtype=torch.float32
)

# Load Stable Diffusion pipeline (CPU-compatible config)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float32
)
pipe.to("cpu")


def extract_face_landmarks(pil_image):
    np_img = np.array(pil_image)
    faces = face_app.get(np_img)
    if not faces:
        return None
    landmarks = faces[0].kps  # facial keypoints
    return landmarks


def run_instantid(face_image: Image.Image, prompt: str) -> Image.Image:
    try:
        # Resize and prepare image
        face_image = face_image.resize((512, 512))
        landmarks = extract_face_landmarks(face_image)
        if landmarks is None:
            return None

        # Create dummy ControlNet image (pose map or similar)
        control_map = np.zeros((512, 512, 3), dtype=np.uint8)
        for (x, y) in landmarks:
            cv2.circle(control_map, (int(x), int(y)), 4, (255, 255, 255), -1)
        control_tensor = transforms.ToTensor()(Image.fromarray(control_map)).unsqueeze(0).to("cpu")

        # Generate portrait
        output = pipe(prompt, image=control_tensor, num_inference_steps=30).images[0]
        return output

    except Exception as e:
        print("Error in InstantID pipeline:", e)
        return None
