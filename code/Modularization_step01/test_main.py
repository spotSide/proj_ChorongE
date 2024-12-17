# main.py

import cv2
import time
import openvino as ov
from test_depth import *
from pathlib import Path
import numpy as np



class WebcamProcessor:
    def __init__(self, camera_id=0, frame_width=1280, frame_height=720):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise ValueError("웹캠을 열 수 없습니다.")
        
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        self.current_frame = None

    def read_frame(self):
        """웹캠으로부터 프레임을 읽어옵니다."""
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("웹캠에서 영상을 읽을 수 없습니다.")
        self.current_frame = frame
        return frame

    def release(self):
        """웹캠 자원을 해제합니다."""
        self.cap.release()



def main():
    webcam_processor = WebcamProcessor(camera_id=0)
    unified_depth(webcam_processor)


if __name__ == "__main__":
    main()
