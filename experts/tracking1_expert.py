import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "tracking/pytracking"))

from experts.tracking_expert_iface import TrackingModel


class Tracking1Model(TrackingModel):
    def __init__(self):
        tracker_name, tracker_param = "dimp", "prdimp18"
        # tracker_name, tracker_param = "dimp", "dimp18"
        # param config should be in pytracking/parameter
        # checkpoint .pth should be in pytracking/networks folder
        # self.model = Tracker(tracker_name, tracker_param)
        super().__init__(tracker_name, tracker_param)

    def apply_expert(self, rgb_frames):
        output_boxes = []
        show_video = False

        # init bbox
        W, H = 400, 400
        start_bbox = (80, 60, 160, 250)
        output_boxes.append(start_bbox)

        # init tracker
        start_frame = np.array(rgb_frames[0].resize((W, H)))
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

        # save output
        tracked_bb = np.array(output_boxes).astype(int)
        bbox_file = "tracking_test.txt"
        np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')

        if show_video:
            cv2.destroyAllWindows()
        return output_boxes
