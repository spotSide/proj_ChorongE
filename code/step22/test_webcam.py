import cv2
import asyncio

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

    async def async_frame_provider(self, shared_data):
        """비동기적으로 웹캠 프레임을 읽어 공유 메모리에 저장합니다."""
        while shared_data['running']:
            try:
                frame = self.read_frame()
                shared_data['frame'] = frame.copy()
            except ValueError as e:
                print(e)
                shared_data['running'] = False
                break
            await asyncio.sleep(0)  # 이벤트 루프 양보

    def release(self):
        """웹캠 자원을 해제합니다."""
        self.cap.release()