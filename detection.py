import os 
import torch
import tensorrt as trt
import numpy as np
import json
from collections import OrderedDict, namedtuple
from yolo_utils import non_max_suppression, scale_boxes
from glob import glob
import cv2

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


class TensorRtPredictor:
    def __init__(self, engine_path='checkpoints/best.engine',
                  img_size=(640, 640), conf_thres=0.35, iou_thres=0.45, classes=None, agnostic_nms=False, maxdet=1000):
        device = torch.device('cuda:0')
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        logger = trt.Logger(trt.Logger.INFO)
        with open(engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
            meta_len = int.from_bytes(f.read(4), byteorder='little')  # read metadata length
            metadata = json.loads(f.read(meta_len).decode('utf-8'))  # read metadata
            model = runtime.deserialize_cuda_engine(f.read())
        context = model.create_execution_context()
        bindings = OrderedDict()
        output_names = []
        fp16 = False  # default updated below
        dynamic = False
        for i in range(model.num_bindings):
            name = model.get_binding_name(i)
            dtype = trt.nptype(model.get_binding_dtype(i))
            if model.binding_is_input(i):
                if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                    dynamic = True
                    context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
                if dtype == np.float16:
                    fp16 = True
            else:  # output
                output_names.append(name)
            shape = tuple(context.get_binding_shape(i))
            im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
        binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
        batch_size = bindings['images'].shape[0]  # if dynamic, this is instead max batch size

        self.__dict__.update(locals())
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.maxdet = maxdet
        self.img_size = img_size

    def preprocess(self, im):
        processed_img, ratio, (dw, dh) = letterbox(im, new_shape=self.img_size, color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32)
        processed_img = processed_img.transpose(2, 0, 1).astype(np.float32)
        processed_img = np.ascontiguousarray(processed_img)
        # print(processed_img.shape)
        processed_img /= 255.0
        processed_img = torch.from_numpy(processed_img).unsqueeze(0).to(self.device)
        return processed_img
    
    def __call__(self, im):
        ori_shape = im.shape
        im = self.preprocess(im)

        if self.dynamic and im.shape != self.bindings['images'].shape:
            i = self.model.get_binding_index('images')
            self.context.set_binding_shape(i, im.shape)  # reshape if dynamic
            self.bindings['images'] = self.bindings['images']._replace(shape=im.shape)
            for name in self.output_names:
                i = self.model.get_binding_index(name)
                self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))
        s = self.bindings['images'].shape
        assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
        self.binding_addrs['images'] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        y = [self.bindings[x].data for x in sorted(self.output_names)]

        if isinstance(y, (list, tuple)):
            pred = self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            pred = self.from_numpy(y)

        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.maxdet)
        pred = pred[0]
        # return pred.cpu().numpy()
        if len(pred) == 0:
            return np.array([])
        else:
            pred[:, :4] = scale_boxes(self.img_size, pred[:, :4], ori_shape).round()
            return pred.cpu().numpy()

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x
    
def main_video():
    import cv2
    import shutil
    from tqdm import tqdm
    output_folder = 'output_folder'
    w = 'checkpoints/best.engine'
    predictor = TensorRtPredictor(w)

    video_path = 'video_test/car_video.mp4'
    out_video_path = os.path.join(output_folder, os.path.basename(video_path))
    cap = cv2.VideoCapture(video_path)

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    while cap.isOpened():
        ret, frame = cap.read()
        
        if ret:
            # new_frame = cv2.resize(frame, (640, 640))
            pred = predictor(frame)
            if len(pred) > 0:
                for p in pred:
                    x1, y1, x2, y2, conf, cls = p
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f'{conf:.2f}', (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            out.write(frame)
            pbar.update(1)

        else:
            break
    
    cap.release()
    out.release()

def main_paths():
    import cv2
    import shutil
    from tqdm import tqdm
    output_folder = 'output_folder'
    w = 'checkpoints/best.engine'
    predictor = TensorRtPredictor(w)

    img_pths = glob('/home/tanpv/fiftyone/coco-2017/validation/images/*')

    for img_pth in tqdm(img_pths):
        img = cv2.imread(img_pth)
        img_copy = img.copy()
        pred = predictor(img)
        print(pred)
        if len(pred) > 0:
            x1, y1, x2, y2, conf, cls = pred[0]
            cv2.rectangle(img_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(output_folder, os.path.basename(img_pth)), img_copy)
        # break


if __name__ == '__main__':
    main_paths()

