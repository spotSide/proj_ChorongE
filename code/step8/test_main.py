from test_depth import *
from test_hand import *
from test_webcam import *
from test_detect import *
from tts import TextToSpeech, DepthWithTTS
import asyncio
import cv2

async def main():
    webcam_processor = WebcamProcessor(camera_id=0)  # 0 웹캠 4 리얼센스
    shared_data = {'frame': None, 'running': True}

    print("Starting async processes...")

    # TTS 및 DepthWithTTS 인스턴스 생성
    tts = TextToSpeech()
    depth_with_tts = DepthWithTTS(tts)

    # 프레임 공급 코루틴
    frame_task = asyncio.create_task(webcam_processor.async_frame_provider(shared_data))

    # Depth Estimation, Hand Detection, YOLO Detection 비동기 함수 시작
    depth_task = asyncio.create_task(depth_with_tts.run(shared_data))
    hand_task = asyncio.create_task(run_hand_detection(shared_data))
    yolo_detector = YOLODetector()
    yolo_task = asyncio.create_task(yolo_detector.run_detection(shared_data))

    try:
        await asyncio.gather(frame_task, depth_task, hand_task, yolo_task)
    except KeyboardInterrupt:
        print("Terminating...")
        shared_data['running'] = False
        await frame_task

    webcam_processor.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())
