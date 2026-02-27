"""
SD-Turbo Web Interface — Real Photo Generation
Author: imroshan18 (https://github.com/imroshan18)
"""

import io
import base64
import torch
from flask import Flask, render_template, request, jsonify
from diffusers import AutoPipelineForText2Image
from PIL import Image

app = Flask(__name__)

# Global pipeline
pipe = None


def load_pipeline():
    global pipe
    if pipe is None:
        print("Loading SD-Turbo model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sd-turbo",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            variant="fp16" if device == "cuda" else None,
        ).to(device)
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        print(f"Model ready on {device}.")
    return pipe


def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()
    steps = int(data.get("steps", 2))
    width = int(data.get("width", 512))
    height = int(data.get("height", 512))
    seed = data.get("seed")

    if not prompt:
        return jsonify({"error": "Prompt is required."}), 400

    steps = max(1, min(4, steps))
    width = min(width, 1024)
    height = min(height, 1024)

    # Enhance prompt for photorealism
    enhanced = f"{prompt}, photorealistic, ultra-detailed, 4K, sharp focus"

    pipeline = load_pipeline()

    generator = None
    if seed is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(int(seed))

    result = pipeline(
        prompt=enhanced,
        num_inference_steps=steps,
        guidance_scale=0.0,
        width=width,
        height=height,
        generator=generator,
    )

    img_b64 = image_to_base64(result.images[0])
    return jsonify({"image": img_b64, "prompt": enhanced})


if __name__ == "__main__":
    load_pipeline()
    app.run(host="0.0.0.0", port=5000, debug=False)
