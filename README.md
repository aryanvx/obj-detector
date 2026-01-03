# Real-time object detection using YOLOv8

a Python-based real-time object detection system using YOLOv8 + OpenCV. Detects 80 different object classes through your webcam w/ bounding boxes & confidence scores.

## feats

- real-time object detection using YOLOv8
- supports 80 object classes (people, cars, phones, bottles, animals, etc.)
- adjustable confidence thresholds
- multiple model sizes for speed / accuracy tradeoff
- frame capture capability
- live FPS counter

## requirements

- Python 3.8+
- webcam
- macOS, Linux, or Windows

## installation

1. clone or download this repo

2. create & activate a venv:
```bash
python3 -m venv .venv
source .venv/bin/activate  # on Windows it's .venv\Scripts\activate
```

3. install dependencies:
```bash
pip install ultralytics opencv-python
```

## usage

### basic usage

run the detector w/ default settings:
```bash
python detector.py
```

the first run will automatically download the YOLOv8 model (~6MB).

### controls

**"q"** to quit the application
**"c"** to capture & save the current frame

### camera selection

if you have multiple cameras, find your camera ID:
```bash
python find_camera.py
```

then modify `detector.py` to use your preferred camera by changing the `camera_id` parameter.

## config

### model sizes

choose different model sizes for speed vs. accuracy tradeoff:

- `"n"` (nano) - fastest, least accurate
- `"s"` (small) - fast, good accuracy
- `"m"` (medium) - balanced
- `"l"` (large) - slower, very accurate
- `"x"` (xlarge) - slowest, most accurate

change in `detector.py`:
```python
detector = ObjectDetector(model_size='n')  # change "n" to your preferred size
```

### confidence threshold

adjust the detection sensitivity (0.0 - 1.0):

- lower values (0.3) = more detections, more false positives
- higher values (0.7) = fewer detections, more accurate

change in `detector.py`:
```python
detector.run_webcam(conf_threshold=0.5)  # adjust this value
```
## camera flip
i flipped the camera cause i don't like looking at my inverted self.

## detected obj classes

the model can detect 80 classes including:
- people
- vehicles (car, truck, bus, motorcycle, bicycle)
- animals (dog, cat, bird, horse, etc.)
- common objects (phone, laptop, bottle, cup, etc.)
- furniture (chair, couch, bed, table)
- and many more i haven't tried

## troubleshooting

### camera not working

1. check camera permissions in ```System Settings``` → ```Privacy & Security``` → ```Camera``` (i'm biased towards mac users sorry)
2. try different camera IDs (0, 1, 2, etc.)
3. run `find_camera.py` to see available cameras

### "couldn't read frame" error

- ensure camera permissions are granted
- try a different `camera_id` value
- close other apps using the camera

### slow performance

- use a smaller model size (`"n"` or `"s"`)
- increase confidence threshold to reduce detections
- close other resource-intensive applications

## license

MIT

## acks

- built w/ [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- uses [OpenCV](https://opencv.org/) for video processing
