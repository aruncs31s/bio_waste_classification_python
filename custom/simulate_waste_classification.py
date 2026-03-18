import os
import pathlib
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Protocol

# Setup path BEFORE imports from local modules
plt = platform.system()
if plt != "Windows":
    pathlib.WindowsPath = pathlib.PosixPath

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory (project root, not custom/)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import numpy as np
import torch
from detect2 import parse_opt as parse_opt_og
from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend

from custom.servo_control import move_servo
from custom.csv_handler import write_to_csv
from custom.file_handler import save_results

from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode

from custom.config import ROI_X1, ROI_Y1, ROI_X2, ROI_Y2  

@dataclass(frozen=True)
class ROIConfig:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class ROIFrameData:
    tensor: np.ndarray
    roi_h: int
    roi_w: int
    origin: tuple[int, int]


class ServoActuator(Protocol):
    def __call__(self, duty: float) -> None:
        ...


class ROIProcessor:
    def __init__(self, roi: ROIConfig):
        self.roi = roi

    def draw_overlay(self, frame):
        cv2.rectangle(frame, (self.roi.x1, self.roi.y1), (self.roi.x2, self.roi.y2), (255, 0, 0), 2)
        cv2.putText(frame, "ROI", (self.roi.x1 + 4, self.roi.y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    def extract_tensor(self, raw_frame, imgsz) -> Optional[ROIFrameData]:
        h, w = raw_frame.shape[:2]
        x1 = max(0, min(self.roi.x1, w))
        x2 = max(0, min(self.roi.x2, w))
        y1 = max(0, min(self.roi.y1, h))
        y2 = max(0, min(self.roi.y2, h))

        if x2 <= x1 or y2 <= y1:
            return None

        roi_crop = raw_frame[y1:y2, x1:x2]
        roi_h, roi_w = roi_crop.shape[:2]
        roi_resized = cv2.resize(roi_crop, (imgsz[1], imgsz[0]))
        im_roi = roi_resized[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
        return ROIFrameData(np.ascontiguousarray(im_roi), roi_h, roi_w, (x1, y1))

    @staticmethod
    def scale_and_offset_boxes(det, model_input_shape, roi_h, roi_w, roi_origin):
        det[:, :4] = scale_boxes(model_input_shape, det[:, :4], (roi_h, roi_w, 3)).round()
        x1, y1 = roi_origin
        det[:, [0, 2]] += x1
        det[:, [1, 3]] += y1


class DetectionThrottle:
    def __init__(self, max_fps: float):
        self.frame_interval = 1.0 / max_fps if max_fps > 0 else 0.0
        self.last_det_time = 0.0

    def allow(self) -> bool:
        now = time.time()
        if now - self.last_det_time < self.frame_interval:
            return False
        self.last_det_time = now
        return True


class ServoPolicy:
    def __init__(self, actuator: ServoActuator):
        self.actuator = actuator
        self.rules: list[tuple[Callable[[str], bool], float, str]] = [
            (lambda cls: cls == "biodegradable", 3.5, "Biodegradable -> 45 deg"),
            (lambda cls: cls == "non biodegradable" or cls.startswith("non"), 10.0, "Non-Biodegradable -> 135 deg"),
        ]

    def handle(self, detected_class: str) -> None:
        for matcher, duty, message in self.rules:
            if matcher(detected_class):
                print(f"[SERVO] {message}")
                self.actuator(duty)
                return


from abc import ABC, abstractmethod
class UARTInterface(ABC):
    @abstractmethod
    def send(self, data: bytes) -> None:
        pass

class DummyUART(UARTInterface):
    def send(self, data: bytes) -> None:
        print(f"[UART] Sending data: {data.hex()}")


from enum import Enum
class ConvoyerBeltActions(Enum):
    STOP = 1
    START = 2
   

class PIUART(UARTInterface):
    def __init__(self, port: str, baudrate: int = 9600):
        try:
            import serial
            self.ser = serial.Serial(port, baudrate)
        except Exception as e:
            raise ImportError("Please install pyserial to use PIUART")
    def send(self, data: ConvoyerBeltActions) -> None:
        self.ser.write(data.value.to_bytes(1, byteorder='big'))


@smart_inference_mode()
def run(
    weights=str(ROOT) + "/" + "yolov5s.pt",  # model path or triton URL
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_format=0,  # save boxes coordinates in YOLO format or Pascal-VOC format (0 for YOLO and 1 for Pascal-VOC)
    save_csv=False,  # save results in CSV format
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
    max_det_fps=5,  # maximum detection rate (frames per second)
    roi_processor: Optional[ROIProcessor] = None,
    servo_policy: Optional[ServoPolicy] = None,
    throttle: Optional[DetectionThrottle] = None,
):
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    roi_processor = roi_processor or ROIProcessor(ROIConfig(ROI_X1, ROI_Y1, ROI_X2, ROI_Y2))
    servo_policy = servo_policy or ServoPolicy(move_servo)
    throttle = throttle or DetectionThrottle(max_det_fps)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))

    for path, im, im0s, vid_cap, s in dataset:
        raw_frame = im0s[0] if isinstance(im0s, list) else im0s  # handle webcam (list) vs image
        roi_data = roi_processor.extract_tensor(raw_frame, imgsz)
        if roi_data is None:
            LOGGER.warning("Invalid ROI bounds for current frame; skipping frame")
            continue

        if not throttle.allow():
            continue  # skip this frame, not yet time for next detection

        with dt[0]:
            im = torch.from_numpy(roi_data.tensor).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        # Inference
        with dt[1]:
            vis_path = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=vis_path).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=vis_path).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=vis_path)
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)


        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
            s += "{:g}x{:g} ".format(*im.shape[2:])  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            # Draw ROI rectangle on the full frame for visualization
            roi_processor.draw_overlay(im0)

            if len(det):
                roi_processor.scale_and_offset_boxes(det, im.shape[2:], roi_data.roi_h, roi_data.roi_w, roi_data.origin)

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # --- Servo control based on detected class ---
                    detected_class = names[int(c)].lower()
                    servo_policy.handle(detected_class)

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f"{names[c]}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"

                    if save_csv:
                        write_to_csv(save_path,p.name, label, confidence_str)

                    if save_txt:  # Write to file
                        if save_format == 0:
                            coords = (
                                (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                            )  # normalized xywh
                        else:
                            coords = (torch.tensor(xyxy).view(1, 4) / gn).view(-1).tolist()  # xyxy
                        line = (cls, *coords, conf) if save_conf else (cls, *coords)  # label format
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            if save_img:
                save_results(dataset, vid_path, vid_writer, vid_cap, i, im0, save_path)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1e3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0] if isinstance(weights, (list, tuple)) else weights)




def parse_opt():
    return parse_opt_og()


def main(opt):
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
