import cv2
import numpy as np

DOMAIN_EXPERTS_GT = ["normals_xtc", "depth_n_1_xtc"]

SAVED_VIDEOS_PATH = "calitatives/"
# video
FPS = 0.5
# FOURCC = cv2.VideoWriter_fourcc(*'MJPG')  # colored
FOURCC = cv2.VideoWriter_fourcc(*'XVID')  # colored


def init_video(video_name, save_video, shape, title=None):
    h, w = shape
    if save_video:
        if title:
            save_video_path = "%s/%s.mp4" % (SAVED_VIDEOS_PATH, title)
        else:
            save_video_path = "%s/%s.mp4" % (SAVED_VIDEOS_PATH, video_name)

        print("w, h", w, h)
        out_video = cv2.VideoWriter(save_video_path,
                                    FOURCC,
                                    FPS, (w, h),
                                    isColor=True)
        print("save_video_path", save_video_path)
        return out_video
    return None


def close_video(out_video, save_video, video_name):
    if save_video:
        out_video.release()
        #         cv2.destroyAllWindows()
        print("Video %s saved" % video_name)


def add_frames(out_stream, frame_ids, domain):
    # save them in video
    for idx in frame_ids:
        frame_path = "calitatives/video/%s/%d.npy" % (domain, idx)
        np_image = np.load(frame_path)[:, :, [2, 1, 0]]
        shape = np_image.shape[0] + TEXT_H, np_image.shape[1], 3
        np_result = np.zeros(shape, dtype=np.uint8)

        # print("w, h", shape[1], shape[0])
        write_me = cv2.resize(np_result,
                              dsize=(shape[1], shape[0]),
                              interpolation=cv2.INTER_CUBIC)
        write_me[TEXT_H:] = np_image
        text_color = (255, 255, 255)
        cv2.putText(write_me, "RGB", (110, TEXT_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)
        cv2.putText(write_me, "Expert", (90 + SIZE, TEXT_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)
        cv2.putText(write_me, "CShift (Ours)", (35 + SIZE * 2, TEXT_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)
        if domain in DOMAIN_EXPERTS_GT:
            cv2.putText(write_me, "GT", (110 + SIZE * 3, TEXT_H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2,
                        cv2.LINE_AA)
            cv2.putText(write_me, "Expert vs CShift",
                        (20 + SIZE * 4, TEXT_H - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        1, text_color, 2, cv2.LINE_AA)

        out_stream.write(write_me)


sz = 4
SIZE = 256
TEXT_H = 40

# # 1. GT video
# video_name = "with_GT"

# shape = 2 * sz + SIZE + TEXT_H, (SIZE + sz) * 5 + sz
# out_stream = init_video(video_name, True, shape, title=video_name)

# # read images
# frame_ids_normals = [
#     173, 187, 333, 386, 467, 486, 491, 492, 506, 532, 552, 576
# ]
# frame_ids_depth = [171, 201, 316, 318, 425, 426, 436, 470, 496]

# add_frames(out_stream, frame_ids_depth, "depth_n_1_xtc")
# add_frames(out_stream, frame_ids_normals, "normals_xtc")

# close_video(out_stream, True, video_name)

# 2. NO GT video
video_name = "no_GT"

shape = 2 * sz + SIZE + TEXT_H, (SIZE + sz) * 3 + sz
out_stream = init_video(video_name, True, shape, title=video_name)

# read images
frame_ids_edges = [50, 46, 86, 90, 115, 166, 185, 252]
frame_ids_super_pixel = [276, 338, 363, 407, 590, 631]
frame_ids_cartoon = [4, 40, 43, 45, 104, 123, 161]
frame_ids_sem_seg = []

add_frames(out_stream, frame_ids_super_pixel, "superpixel_fcn")
add_frames(out_stream, frame_ids_edges, "edges_dexined")
add_frames(out_stream, frame_ids_cartoon, "cartoon_wb")
# add_frames(out_stream, frame_ids_sem_seg, "sem_seg_hrnet")

close_video(out_stream, True, video_name)
