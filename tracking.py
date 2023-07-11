from base_tracker.byte_tracker import BYTETracker
import cv2
from detection import TensorRtPredictor
from tqdm import tqdm
import numpy as np


color_base_class = {
    0: ['car', [0, 255, 0]], 
    1: ['bus', [0, 0, 255]],
    2: ['truck', [255, 0, 0]],
    3: ['motorcycle', [0, 255, 255]],
    4: ['bicycle', [255, 0, 255]],
}


class BytetrackSegment:
    def __init__(self, track_args, num_classes=5):
        self.detector = TensorRtPredictor()
        self.num_classes = num_classes
        for i in range(num_classes):    
            setattr(self, 'tracker_class_{}'.format(i), BYTETracker(track_args, frame_rate=track_args.frame_rate))
            # self.tracker = BYTETracker(track_args, frame_rate=track_args.frame_rate)

    def run(self, video_path, ouput_video_path):
        print('start tracking')
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(ouput_video_path, fourcc, fps, (width, height))
        frame_idx = 0
        pbar = tqdm(total=cap.get(cv2.CAP_PROP_FRAME_COUNT))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            boxes = self.detector(frame)
            boxes[:, 5] = boxes[:, 5].astype(np.int8)
            for i in range(self.num_classes):
                boxes_per_class_pos = np.where(boxes[:, 5]== i)
                boxes_per_class = boxes[boxes_per_class_pos, :5][0]
                track_boxes = getattr(self, 'tracker_class_{}'.format(i)).update(boxes_per_class)
                frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

                for t in track_boxes:
                    tlwh = t.tlwh
                    tid = t.track_id
                    cv2.putText(frame, f'{color_base_class[i][0]}:{tid}', (int(tlwh[0]), int(tlwh[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, color_base_class[i][1], 2)
                    cv2.rectangle(frame, (int(tlwh[0]), int(tlwh[1])), (int(tlwh[0]+tlwh[2]), int(tlwh[1]+tlwh[3])), color_base_class[i][1], 2)

            out.write(frame)    
            frame_idx += 1
            pbar.update(1)
        cap.release()
        out.release()
        print('\nfinished')

if __name__ == "__main__":
    
    from dotmap import DotMap

    args_reid_iou = {
        'frame_rate': 90,
        'track_buffer': 25,
        'track_thresh': 0.4,
        'match_thresh': 0.5,
        'type_fuse': 'iou'
    }

    args_reid_iou = DotMap(args_reid_iou)

    tracker = BytetrackSegment(args_reid_iou)
    video_p = 'video_test/car_bus_truck.mp4'
    out_video = 'test.mp4'
    tracker.run(video_p, out_video)