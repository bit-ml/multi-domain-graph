import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "tracking/pytracking"))

#from experts.tracking_expert_iface import TrackingModel

W, H = 256, 256


#class Tracking1Model(TrackingModel):
class Tracking1Model():
    def __init__(self, full_expert=True):
        tracker_name, tracker_param = "dimp", "prdimp18"
        # tracker_name, tracker_param = "dimp", "dimp18"
        # param config should be in pytracking/parameter
        # checkpoint .pth should be in pytracking/networks folder
        # self.model = Tracker(tracker_name, tracker_param)
        #super().__init__(tracker_name, tracker_param)
        self.domain_name = "tracking"
        self.n_maps = 1
        self.str_id = "tracking_prdimp18"
        self.identifier = "tracking_prdimp18"

    def apply_expert(self, rgb_frames, start_bbox=(80, 60, 160, 250)):
        '''
        x, y, w, h = start_bbox
        '''
        output_boxes = []
        show_video = False
        start_frame = np.array(rgb_frames[0].resize((W, H)))

        # init bbox
        sx, sy, sw, sh = start_bbox
        rW = rgb_frames[0].size[0] / W
        rH = rgb_frames[0].size[1] / H
        start_bbox = (sx / rW, sy / rH, sw / rW, sh / rH)
        output_boxes.append(start_bbox)

        # init tracker
        video_tracker = self.init_video(start_frame, start_bbox)

        if show_video:
            display_name = 'Display: ' + self.model.name
            cv2.namedWindow(display_name,
                            cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.imshow(display_name, start_frame)

        # frame by frame tracking
        for frame in rgb_frames[1:]:
            if frame is None:
                break
            frame = np.array(frame.resize((W, H)))
            bbox = self.apply_one_step(video_tracker, frame)
            output_boxes.append(bbox)

            # Draw box
            if show_video:
                frame_disp = frame.copy()
                cv2.rectangle(frame_disp, (bbox[0], bbox[1]),
                              (bbox[2] + bbox[0], bbox[3] + bbox[1]),
                              (0, 255, 0), 5)
                cv2.imshow(display_name, frame_disp)
                cv2.waitKey(300)

        # # save output
        # tracked_bb = np.array(output_boxes).astype(int)
        # bbox_file = "tracking_test.txt"
        # np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')

        if show_video:
            cv2.destroyAllWindows()
        return output_boxes

    def apply_expert_for_last_map(self, rgb_frames, start_bbox):
        bboxes = self.apply_expert(rgb_frames, start_bbox)
        out_map = np.zeros((1, H, W))
        x, y, bbox_w, bbox_h = bboxes[-1]
        x, y, bbox_w, bbox_h = round(x), round(y), round(bbox_w), round(bbox_h)

        # bbox: cv2.rectangle(im_np, (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]),
        out_map[:, y:y + bbox_h, x:x + bbox_w] = 1.

        # # for checking tracking output
        # res_img = np.array(rgb_frames[0].resize(
        #     (W, H)), dtype=np.float32).transpose(2, 0, 1) / 255.
        # out_map = (out_map + res_img).clip(min=0, max=1)
        # self.n_maps = 3

        # import matplotlib.pyplot as plt
        # plt.figure(1)
        # plt.imshow(out_map.transpose(1, 2, 0))

        # plt.figure(2)
        # crt = np.array(rgb_frames[0])
        # x, y, bbox_w, bbox_h = start_bbox
        # x, y, bbox_w, bbox_h = int(x), int(y), int(bbox_w), int(bbox_h)
        # crt[y:y + bbox_h, x:x + bbox_w, :] = 1.
        # plt.imshow(crt)
        # plt.show()

        return out_map
