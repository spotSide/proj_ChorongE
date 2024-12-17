import cv2
import asyncio
from ultralytics import YOLO
import os
import logging
from ultralytics import __version__


# 로깅 수준 설정
logging.getLogger("ultralytics").setLevel(logging.WARNING)

class YOLODetector:
    def __init__(self, model_path='best.pt'):
        # 현재 파일이 위치한 디렉토리에서 best.pt 파일을 찾습니다.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO 모델 파일을 찾을 수 없습니다: {model_path}")

        self.model = YOLO(model_path)

    def enhance_image(self, image):
        """명도 및 대비를 개선하는 함수 (CLAHE 적용)."""
        # LAB 색공간으로 변환
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # CLAHE 적용 (L 채널에만 적용)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        # 개선된 L 채널을 합쳐 다시 BGR로 변환
        limg = cv2.merge((cl, a, b))
        enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        return enhanced_image

    async def run_detection(self, shared_data):
        """비동기적으로 YOLO 모델을 사용해 객체 감지를 실행합니다."""
        print("Starting YOLO Detection...")
        while shared_data['running']:
            frame = shared_data.get('frame')
            if frame is None:
                await asyncio.sleep(0)  # 프레임이 준비될 때까지 대기
                continue

            # 이미지 품질 개선
            enhanced_frame = self.enhance_image(frame)

            # 이미지 전처리 (YOLO 입력 해상도: 640x640)
            img_resized = cv2.resize(enhanced_frame, (640, 640))

            # 모델 예측
            results = self.model(img_resized, verbose=False)

            # 감지 결과 시각화
            plots = results[0].plot()

            # 감지 결과를 1280x720 해상도로 조정
            output_frame = cv2.resize(plots, (1280, 720))

            # 결과 표시
            cv2.imshow("YOLO Detection", output_frame)

            # 'q' 키로 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                shared_data['running'] = False
                break

            await asyncio.sleep(0)  # 이벤트 루프 양보

        cv2.destroyAllWindows()

# print(f"Ultralytics YOLO version: {__version__}")
