from PIL import Image

from experts.experts import Experts

if __name__ == "__main__":
    all_experts = Experts()

    image_path = "test_img.jpg"
    rgb_frames = [Image.open(image_path)]

    output_maps = all_experts.rgb_inference(rgb_frames)
