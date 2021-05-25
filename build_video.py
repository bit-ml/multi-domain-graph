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


def add_frames(out_stream, frame_ids, domain, message):
    # save them in video
    for idx in frame_ids:
        frame_path = "calitatives/video/%s/%d.npy" % (domain, idx)
        np_image = np.load(frame_path)[:, :, [2, 1, 0]]
        if domain in DOMAIN_EXPERTS_GT:
            shape = np_image.shape[0] + 5 * TEXT_H, np_image.shape[1], 3
        else:
            shape = np_image.shape[0] + 4 * TEXT_H, np_image.shape[1], 3

        np_result = np.zeros(shape, dtype=np.uint8)

        print("w, h", shape[1], shape[0])
        write_me = cv2.resize(np_result,
                              dsize=(shape[1], shape[0]),
                              interpolation=cv2.INTER_CUBIC)
        if domain in DOMAIN_EXPERTS_GT:
            write_me[3 * TEXT_H:-2 * TEXT_H] = np_image
        else:
            write_me[3 * TEXT_H:-TEXT_H] = np_image

        text_color = (255, 255, 255)

        if domain in DOMAIN_EXPERTS_GT:
            w_center_title = (SIZE * 5 - len(message) * 32) // 2
        else:
            w_center_title = (SIZE * 3 - len(message) * 32) // 2

        cv2.putText(write_me, message, (w_center_title, TEXT_H + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255), 3, cv2.LINE_AA)

        write_me[sz + 3 * TEXT_H:sz + SIZE + 3 * TEXT_H, sz:(SIZE + sz),
                 1] = np_image[sz:sz + SIZE, sz + (SIZE + sz):(SIZE + sz) * 2,
                               0]
        write_me[sz + 3 * TEXT_H:sz + SIZE + 3 * TEXT_H, sz:(SIZE + sz),
                 2] = np_image[sz:sz + SIZE,
                               sz + (SIZE + sz) * 2:(SIZE + sz) * 3, 0]

        cv2.putText(write_me, "RGB input", (55, 3 * TEXT_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)
        cv2.putText(write_me, "Expert", (90 + SIZE, 3 * TEXT_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)
        cv2.putText(write_me, "CShift (Ours)",
                    (35 + SIZE * 2, 3 * TEXT_H - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    1, text_color, 2, cv2.LINE_AA)
        if domain in DOMAIN_EXPERTS_GT:
            cv2.putText(write_me, "Ground-truth",
                        (40 + SIZE * 3, 3 * TEXT_H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2,
                        cv2.LINE_AA)
            cv2.putText(write_me, "Expert vs CShift",
                        (20 + SIZE * 4, 3 * TEXT_H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2,
                        cv2.LINE_AA)

            # add legend
            legend_w = 145
            cv2.putText(
                write_me,
                "CShift is better than Expert, when compared against ground-truth",
                (legend_w, shape[0] - 95), cv2.FONT_HERSHEY_SIMPLEX, 1,
                text_color, 1, cv2.LINE_AA)
            cv2.putText(
                write_me,
                "CShift is worse than Expert, when compared against ground-truth",
                (legend_w, shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                text_color, 1, cv2.LINE_AA)

            cv2.rectangle(write_me, (shape[1] - 80, shape[0] - 110),
                          (shape[1] - 20, shape[0] - 95), (0, 255, 0), -1)
            cv2.rectangle(write_me, (shape[1] - 80, shape[0] - 75),
                          (shape[1] - 20, shape[0] - 60), (0, 0, 255), -1)
        out_stream.write(write_me)
        # break


def title_slide(out_stream, shape, message):
    np_result = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)

    write_me = cv2.resize(np_result,
                          dsize=(shape[1], shape[0]),
                          interpolation=cv2.INTER_CUBIC)
    text_color = (255, 255)
    cv2.putText(write_me, message,
                ((shape[1] - len(message) * 35) // 2 - 4, shape[0] // 2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 2, text_color, 2, cv2.LINE_AA)
    out_stream.write(write_me)


sz = 4
SIZE = 256
TEXT_H = 60

# # 1. GT video
# video_name = "with_ground-truth"

# shape = 2 * sz + SIZE + 5 * TEXT_H, (SIZE + sz) * 5 + sz
# out_stream = init_video(video_name, True, shape, title=video_name)

# # read images
# frame_ids_normals = [
#     173, 187, 333, 386, 467, 486, 491, 492, 506, 532, 552, 576
# ]
# frame_ids_depth = [171, 201, 316, 318, 425, 426, 436, 470, 496]

# title_slide(out_stream, shape, "Tasks with ground-truth")
# add_frames(out_stream, frame_ids_depth, "depth_n_1_xtc", "Task: depth")
# add_frames(out_stream, frame_ids_normals, "normals_xtc", "Task: normals")
# close_video(out_stream, True, video_name)

# 2. NO GT video
video_name = "no_ground-truth"

shape = 2 * sz + SIZE + 4 * TEXT_H, (SIZE + sz) * 3 + sz
out_stream = init_video(video_name, True, shape, title=video_name)

# read images
frame_ids_edges = [50, 46, 86, 90, 115, 166, 185, 252]
frame_ids_super_pixel = [276, 338, 363, 407, 590, 631]
frame_ids_cartoon = [4, 40, 43, 45, 104, 123, 161]

# title_slide(out_stream, shape, "Tasks w/o ground-truth")
# add_frames(out_stream, frame_ids_super_pixel, "superpixel_fcn",
#            "Task: super-pixel")
add_frames(out_stream, frame_ids_edges, "edges_dexined", "Task: edges")
# add_frames(out_stream, frame_ids_cartoon, "cartoon_wb", "Task: cartoon")
close_video(out_stream, True, video_name)
