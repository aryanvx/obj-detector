import cv2
from ultralytics import YOLO
import numpy as np

class ObjectDetector:

    def __init__(self, model_size="n"):
        
        print(f"loading YOLOv8{model_size} model…")
        self.model = YOLO(f"yolov8{model_size}.pt")
        self.colors = np.random.randint(0, 255, size=(80, 3), dtype=int)
        
    def detect_objects(self, frame, conf_threshold=0.5):
        
        results = self.model(frame, conf=conf_threshold, verbose=False)
        return results[0]
    
    def draw_detections(self, frame, results):

        for box in results.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            label = self.model.names[cls]
            color = tuple(map(int, self.colors[cls]))
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label_text = f"{label} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
            
            cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def run_webcam(self, conf_threshold=0.5, camera_id=0):

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():

            print("error: couldn't open webcam")
            return
        
        print("starting detection… press \"q\" to quit")
        print("press \"c\" to capture & save current frame")
        
        frame_count = 0
        
        while True:

            ret, frame = cap.read()
            if not ret:

                print("error: couldn't read frame")
                break

            frame = cv2.flip(frame, 1)
            
            results = self.detect_objects(frame, conf_threshold)
            
            frame = self.draw_detections(frame, results)
            
            frame_count += 1
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("YOLO Object Detection", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):

                break

            elif key == ord("c"):

                filename = f"capture_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"saved {filename}")
        
        cap.release()
        cv2.destroyAllWindows()


def main():
    import cv2
    
    print("detecting available cameras…")
    available_cameras = []
    for i in range(5):

        cap = cv2.VideoCapture(i)
        if cap.isOpened():

            ret, frame = cap.read()
            if ret:

                available_cameras.append(i)
                print(f"camera {i}: available")
            cap.release()
        else:

            print(f"camera {i}: not available")
    
    if not available_cameras:

        print("no cameras found!")
        return
    
    print(f"\nfound cameras: {available_cameras}")
    camera_choice = int(input(f"enter camera ID to use (default {available_cameras[0]}): ") or available_cameras[0])
    
    detector = ObjectDetector(model_size="n")
    detector.run_webcam(conf_threshold=0.5, camera_id=camera_choice)

if __name__ == "__main__":

    main()