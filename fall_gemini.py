import pyrealsense2 as rs
import cv2
import mediapipe as mp
import google.generativeai as genai
import time
import base64
from PIL import Image
import io
import numpy as np

# Configure Gemini API
genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel('gemini-2.0-flash-exp')

# MediaPipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# RealSense setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
pipeline.start(config)

def encode_image_to_base64(image):
    """Convert OpenCV image to base64 for Gemini"""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def detect_fall_with_gemini(image):
    """Use Gemini to detect if person has fallen"""
    try:
        # Encode image
        base64_image = encode_image_to_base64(image)
        
        # Create prompt
        prompt = """
        Look at this image and determine if a person has fallen over or is lying on the floor.
        Only respond with exactly "FALLEN" if you see a person who has fallen or is lying down.
        Otherwise respond with "STANDING".
        Be very strict - only say FALLEN if the person is clearly on the ground/floor.
        """
        
        # Generate response
        response = model.generate_content([
            prompt,
            {"mime_type": "image/jpeg", "data": base64_image}
        ])
        
        return "FALLEN" in response.text.upper()
        
    except Exception as e:
        print(f"Gemini error: {e}")
        return False

def detect_person_with_mediapipe(image):
    """Use MediaPipe to detect if person is present"""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_image)
    return results.pose_landmarks is not None

print("Fall Detection System (Gemini + RealSense D455)")
print("Press 'q' to quit")

last_fall_detection = 0
detection_interval = 2.0  # Check every 2 seconds

try:
    while True:
        # Get frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            continue
            
        image = np.asanyarray(color_frame.get_data())
        
        # Check for person with MediaPipe
        person_detected = detect_person_with_mediapipe(image)
        
        if person_detected:
            # Check for fall with Gemini every 2 seconds
            current_time = time.time()
            if current_time - last_fall_detection > detection_interval:
                fall_detected = detect_fall_with_gemini(image)
                if fall_detected:
                    print("The person has fallen")
                last_fall_detection = current_time
        else:
            print("No person detected")
        
        # Display image
        cv2.imshow('Fall Detection', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Stopping...")
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    pose.close()
