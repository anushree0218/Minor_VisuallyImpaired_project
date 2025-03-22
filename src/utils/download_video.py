import cv2

def capture_video(camera_source=0):
    cap = cv2.VideoCapture(camera_source)
    return cap