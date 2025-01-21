from ultralytics import YOLO
import cv2

class yolo_predict():
    def __init__(self):
        pass
    def predict_and_compare(self,frame_preocessed,frame):
        # Load the YOLO model
        model = YOLO("best.pt")
        modelpretrained = YOLO("yolov8s.pt")
        # Perform object detection
        results = model.predict(source=frame_preocessed, show=False)
        results_pretrained = modelpretrained.predict(source=frame, show=False)
        image_with_boxes = results[0].plot()
        image_with_boxes_pretrained = results_pretrained[0].plot()
        return image_with_boxes, image_with_boxes_pretrained


