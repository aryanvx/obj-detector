import cv2
from ultralytics import YOLO
import numpy as np
import time


class ObjectDetector:

    def __init__(self, model_size="n"):
        print(f"loading YOLOv8{model_size} model…")
        self.model = YOLO(f"yolov8{model_size}.pt")
        self.colors = np.random.randint(0, 255, size=(80, 3), dtype=int)
        
    def detect_objects(self, frame, conf_threshold=0.5):
        results = self.model(frame, conf=conf_threshold, verbose=False)
        return results[0]
    
    def draw_detections(self, frame, results):
        if results is None:
            return frame
            
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
            cv2.putText(frame, label_text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def run_webcam(self, conf_threshold=0.5, camera_id=0):
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("error: couldn't open webcam")
            return
        
        print("starting detection… press \"q\" to quit")
        print("press \"c\" to capture & save current frame")
        
        target_inference_fps = 15  # running YOLO at 15 FPS
        inference_dt = 1.0 / target_inference_fps
        inference_accumulator = 0.0
        
        prev_time = time.time()
        last_inference_time = time.time()
        
        results = None
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("error: couldn't read frame")
                break
            
            frame = cv2.flip(frame, 1)
            
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            
            inference_accumulator += dt
            
            if inference_accumulator >= inference_dt:
                results = self.detect_objects(frame, conf_threshold)
                inference_accumulator -= inference_dt
                last_inference_time = curr_time
            
            display_frame = self.draw_detections(frame.copy(), results)
            
            render_fps = 1 / (dt + 0.0001)
            inference_age = (curr_time - last_inference_time) * 1000  # ms
            
            cv2.putText(display_frame, f"render fps: {int(render_fps)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"inference: {int(1/inference_dt)} fps", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(display_frame, f"detection age: {inference_age:.0f}ms", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("YOLO object detection", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                filename = f"capture_{frame_count}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"saved {filename}")
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()


def main():
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