import torch

from models.common import DetectMultiBackend
from utils.dataloaders import LoadStreams, LoadImages
from utils.general import (Profile, check_img_size, cv2,
                           non_max_suppression, scale_boxes)
from utils.torch_utils import select_device

# @smart_inference_mode()
def run(
        weights='runs/train/exp16/weights/best.pt',  # model path or triton URL
        # source='1',  # file/dir/URL/glob/screen/0(webcam)
        source = 'exp-data/images/img_5.jpg',
        data='data/data.yaml',  # dataset.yaml path
        imgsz=(320, 320),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=5,  # maximum detections per image
        device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, pt = model.stride, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    # dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    # bs = len(dataset)
    #######
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    bs = 1
    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    dt =(Profile(), Profile(), Profile())

    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            # im0= im0s[i].copy()
            im0= im0s.copy()        
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round() 
                # yield {'imR' : im0, 'detR' : det}
                yield im0,det

def main():
    for imResult, predResult in run():        
        print("im ", imResult)
        print("det ", predResult)

# if __name__ == "__main__":
#     main()
