import glob
import os
import sys
from collections import OrderedDict

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "tracking/pytracking"))

from experts.tracking.pytracking.pytracking.evaluation import Tracker
from experts.tracking.pytracking.pytracking.evaluation.multi_object_wrapper import \
    MultiObjectWrapper


def _build_init_info(box):
    return {
        'init_bbox': OrderedDict({1: box}),
        'init_object_ids': [
            1,
        ],
        'object_ids': [
            1,
        ],
        'sequence_object_ids': [
            1,
        ]
    }


class TrackingModel():
    def __init__(self, tracker_name, tracker_param):
        self.model = Tracker(tracker_name, tracker_param)

    def init_video(self, start_frame, start_bbox):
        params = self.model.get_parameters()
        video_tracker = MultiObjectWrapper(self.model.tracker_class,
                                           params,
                                           fast_load=True)
        video_tracker.initialize(start_frame, _build_init_info(start_bbox))

        return video_tracker

    def apply_one_step(self, video_tracker, rgb_frame):
        out = video_tracker.track(rgb_frame)
        state = [int(s) for s in out['target_bbox'][1]]
        return state
