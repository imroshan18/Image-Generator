"""
Real Photo Generation Use Case
Author: imroshan18 (https://github.com/imroshan18)
Description: Demonstrates photorealistic image generation for professional workflows
             including portraits, landscapes, architecture, products, and more.
"""

import os
from generate_image import SDTurboGenerator


PHOTO_ASSETS = {
    "portraits": [
        "A professional headshot of a businessman in a suit, soft studio lighting",
        "A candid portrait of a young woman laughing in a park, golden hour",
        "An elderly man with a weathered face, black and white portrait photography",
    ],
    "landscapes": [
        "A misty mountain valley at dawn, long exposure, photorealistic",
        "A tropical beach at sunset with crystal-clear water and soft waves",
        "A snow-covered forest trail in winter, diffused natural light",
    ],
    "architecture": [
        "A modern glass skyscraper reflecting clouds, architectural photography",
        "A rustic wooden cabin surrounded by autumn foliage",
        "A minimalist interior living room with floor-to-ceiling windows",
    ],
    "products": [
        "A luxury watch on a marble surface, macro product photography",
        "A perfume bottle with bokeh background, commercial photography",
        "A coffee cup with latte art on a wooden table, warm lighting",
    ],
    "street": [
        "A rainy city street at night with neon light reflections",
        "A bustling outdoor market in the morning light",
        "A quiet European alley with cobblestones and flower pots",
    ],
}


def generate_photo_assets(
    categories: list = None,
    num_inference_steps: int = 2,
    output_dir: str = "outputs/photo_assets",
):
    """
    Generate photorealistic images across multiple categories.

    Args:
        categories: List of category names to generate (default: all)
        num_inference_steps: Quality/speed tradeoff (1-4)
        output_dir: Directory to save generated images
    """
    if categories is None:
        categories = list(PHOTO_ASSETS.keys())

    os.makedirs(output_dir, exist_ok=True)
    generator = SDTurboGenerator()

    total = sum(len(PHOTO_ASSETS[c]) for c in categories if c in PHOTO_ASSETS)
    generated = 0

    for category in categories:
        if category not in PHOTO_ASSETS:
            print(f"Category '{category}' not found, skipping.")
            continue

        cat_dir = os.path.join(output_dir, category)
        os.makedirs(cat_dir, exist_ok=True)
        prompts = PHOTO_ASSETS[category]

        print(f"\n📸 Generating {category} ({len(prompts)} images)...")
        images = generator.generate_batch(prompts, num_inference_steps=num_inference_steps)

        for i, (img, prompt) in enumerate(zip(images, prompts)):
            filename = f"{category}_{i+1:02d}.png"
            path = os.path.join(cat_dir, filename)
            img.save(path)
            generated += 1
            print(f"  [{generated}/{total}] Saved: {path}")
            print(f"         Prompt: {prompt[:60]}...")

    print(f"\n✅ Done! {generated} photorealistic images saved to {output_dir}/")
    return output_dir


def quick_demo():
    """Quick demo — generates one image per category."""
    os.makedirs("outputs/demo", exist_ok=True)
    generator = SDTurboGenerator()

    demo_prompts = {
        "portrait": "A professional headshot of a woman with natural studio lighting",
        "landscape": "A dramatic mountain sunrise with golden clouds",
        "architecture": "A modern minimalist house with large windows in a forest",
        "product": "A sleek smartphone on a reflective black surface",
        "street": "A vibrant street market scene in warm evening light",
    }

    print("\n🚀 Quick Demo — One image per category\n")
    for name, prompt in demo_prompts.items():
        print(f"Generating: {name}...")
        image = generator.generate(prompt=prompt, num_inference_steps=2, seed=42)
        path = f"outputs/demo/{name}.png"
        image.save(path)
        print(f"  Saved: {path}")

    print("\n✅ Demo complete! Check outputs/demo/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Real Photo Asset Generator by imroshan18")
    parser.add_argument("--demo", action="store_true", help="Run quick demo (1 image per category)")
    parser.add_argument("--categories", nargs="+", help="Categories to generate", default=None)
    parser.add_argument("--steps", type=int, default=2, help="Inference steps (1-4)")
    args = parser.parse_args()

    if args.demo:
        quick_demo()
    else:
        generate_photo_assets(categories=args.categories, num_inference_steps=args.steps)
