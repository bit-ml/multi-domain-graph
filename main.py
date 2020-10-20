import glob

from PIL import Image

from experts.experts import Experts

if __name__ == "__main__":
    all_experts = Experts()

    image_path = "test_img.jpg"
    rgb_frames = [Image.open(image_path)]

    # videopath = "test_video"
    # rgb_frames = [
    #     Image.open(image_path)
    #     for image_path in glob.glob("%s/*.jpg" % videopath)
    # ]

    output_maps = all_experts.rgb_inference(rgb_frames)
