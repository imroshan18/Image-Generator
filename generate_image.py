"""
SD-Turbo Real Photo Generator
Author: imroshan18 (https://github.com/imroshan18)
Description: Ultra-fast photorealistic image generation using Stable Diffusion Turbo
"""

import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image
import time
import os


class SDTurboGenerator:
    """
    Fast photorealistic image generator using SD-Turbo.
    Supports single and batch generation with configurable quality/speed tradeoffs.
    """

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = None
        self._load_model()

    def _load_model(self):
        """Load SD-Turbo model with FP16 optimization."""
        print("Loading SD-Turbo model...")
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sd-turbo",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            variant="fp16" if self.device == "cuda" else None,
        )
        self.pipe = self.pipe.to(self.device)

        # Enable memory-efficient attention if available
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("xFormers memory-efficient attention enabled.")
        except Exception:
            print("xFormers not available, using default attention.")

        print(f"Model loaded on {self.device}.")

    def generate(
        self,
        prompt: str,
        num_inference_steps: int = 1,
        width: int = 512,
        height: int = 512,
        seed: int = None,
    ) -> Image.Image:
        """
        Generate a photorealistic image from a text prompt.

        Args:
            prompt: Text description of the desired image
            num_inference_steps: Number of denoising steps (1-4 recommended)
            width: Output image width in pixels
            height: Output image height in pixels
            seed: Random seed for reproducibility

        Returns:
            PIL Image object
        """
        # Append photorealism keywords automatically
        enhanced_prompt = f"{prompt}, photorealistic, ultra-detailed, 4K, sharp focus"

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        start = time.time()
        result = self.pipe(
            prompt=enhanced_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=0.0,  # SD-Turbo works without guidance
            width=width,
            height=height,
            generator=generator,
        )
        elapsed = time.time() - start
        print(f"Image generated in {elapsed:.2f}s")

        return result.images[0]

    def generate_batch(
        self,
        prompts: list,
        num_inference_steps: int = 2,
        width: int = 512,
        height: int = 512,
    ) -> list:
        """
        Generate multiple photorealistic images from a list of prompts.

        Args:
            prompts: List of text descriptions
            num_inference_steps: Number of denoising steps
            width: Output image width
            height: Output image height

        Returns:
            List of PIL Image objects
        """
        enhanced_prompts = [
            f"{p}, photorealistic, ultra-detailed, 4K, sharp focus" for p in prompts
        ]

        print(f"Generating batch of {len(prompts)} images...")
        start = time.time()
        results = self.pipe(
            prompt=enhanced_prompts,
            num_inference_steps=num_inference_steps,
            guidance_scale=0.0,
            width=width,
            height=height,
        )
        elapsed = time.time() - start
        print(f"Batch generated in {elapsed:.2f}s ({elapsed/len(prompts):.2f}s per image)")

        return results.images


def main():
    """Demo: Generate sample photorealistic images."""
    os.makedirs("outputs", exist_ok=True)

    generator = SDTurboGenerator()

    # --- Single Image ---
    print("\n--- Single Image Generation ---")
    image = generator.generate(
        prompt="A golden hour portrait of a woman standing in a wheat field",
        num_inference_steps=2,
        seed=42,
    )
    image.save("outputs/portrait_golden_hour.png")
    print("Saved: outputs/portrait_golden_hour.png")

    # --- Batch Generation ---
    print("\n--- Batch Generation ---")
    prompts = [
        "A misty forest path in the morning light",
        "A modern city skyline at night with reflections on wet pavement",
        "A cozy kitchen with natural light streaming through windows",
    ]
    images = generator.generate_batch(prompts, num_inference_steps=2)
    for i, img in enumerate(images):
        path = f"outputs/batch_{i+1}.png"
        img.save(path)
        print(f"Saved: {path}")

    print("\nAll images saved to ./outputs/")


if __name__ == "__main__":
    main()
