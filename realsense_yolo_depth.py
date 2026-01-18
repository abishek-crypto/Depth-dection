import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import pyttsx3
import time

#  CONFIGURATION 
MIN_SAFE_DISTANCE = 2.0  # meters
SPEAK = True  

#  SETUP TTS 
if SPEAK:
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)

def speak(text):
    if SPEAK:
        engine.say(text)
        engine.runAndWait()
    else:
        print(text)

#  SETUP REALSENSE PIPELINE
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# Align depth to color frame
align_to = rs.stream.color
align = rs.align(align_to)

#  LOAD YOLOv8 MODEL 
model = YOLO("yolov8n.pt")  # You can replace with yolov8s.pt for better accuracy

#  MAIN LOOP 
print("Starting RealSense + YOLOv8 navigation... Press 'q' to quit.")
speak("Starting navigation system")

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
        color_image = np.asanyarray(color_frame.get_data())

        h, w, _ = color_image.shape

        # Run YOLO object detection
        results = model(color_image, stream=True)

        nearest = {"left": float("inf"), "center": float("inf"), "right": float("inf")}

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy().astype(int)
            conf = result.boxes.conf.cpu().numpy()

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                label = model.names[cls[i]]
                confidence = conf[i]

                # Compute depth median inside bounding box
                box_depth = depth_image[y1:y2, x1:x2]
                valid_depths = box_depth[np.isfinite(box_depth)]
                if len(valid_depths) == 0:
                    continue
                median_depth = np.median(valid_depths)

                # Draw detections
                color = (0, 255, 0) if median_depth > MIN_SAFE_DISTANCE else (0, 0, 255)
                cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(color_image, f"{label} {median_depth:.2f}m", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Determine which region object is in
                region_width = w // 3
                if x2 < region_width:
                    region = "left"
                elif x1 > 2 * region_width:
                    region = "right"
                else:
                    region = "center"

                if median_depth < nearest[region]:
                    nearest[region] = median_depth

        # Decision logic
        command = ""
        if nearest["center"] > MIN_SAFE_DISTANCE:
            command = "Move Forward"
        elif nearest["left"] > MIN_SAFE_DISTANCE:
            command = "Move Left"
        elif nearest["right"] > MIN_SAFE_DISTANCE:
            command = "Move Right"
        else:
            command = "Stop"

        # Speak command if changed
        static = getattr(speak, "last_cmd", None)
        if static != command:
            speak(command)
            speak.last_cmd = command

        # Display distances on bottom of screen
        cv2.rectangle(color_image, (0, h - 35), (w, h), (0, 0, 0), -1)
        cv2.putText(color_image,
                    f"Left: {nearest['left'] if nearest['left'] != float('inf') else '∞'} m   |   "
                    f"Center: {nearest['center'] if nearest['center'] != float('inf') else '∞'} m   |   "
                    f"Right: {nearest['right'] if nearest['right'] != float('inf') else '∞'} m",
                    (20, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 0), 2)

        cv2.putText(color_image, f"Command: {command}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("RealSense YOLOv8 Depth Navigation", color_image)

        # Quit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    speak("Navigation stopped")
