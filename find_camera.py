import cv2

print("scanning for cameras…")
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            height, width = frame.shape[:2]
            print(f"camera {i}: working! resolution: {width}x{height}")
        else:
            print(f"camera {i}: opens but can't read frames")
        cap.release()
    else:
        print(f"camera {i}: not available")

print("\ndone! try the camera ID that shows \"working!\"")