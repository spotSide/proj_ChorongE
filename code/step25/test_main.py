from test_depth import *
from test_hand import *
from test_webcam import *
from test_detect import *
from tts import TextToSpeech, DepthWithTTS
import asyncio
import cv2

def initialize_components():
    """필요한 모든 구성 요소 초기화"""
    webcam_processor = WebcamProcessor(camera_id=0)  # 0: 일반 웹캠, 4: 리얼센스
    shared_data = {'frame': None, 'running': True}
    tts = TextToSpeech()
    depth_with_tts = DepthWithTTS(tts)
    yolo_detector = YOLODetector()

    return webcam_processor, shared_data, depth_with_tts, yolo_detector

async def main():
    # 구성 요소 초기화
    webcam_processor, shared_data, depth_with_tts, yolo_detector = initialize_components()

    print("Starting async processes...")

    # 프레임 공급 비동기 작업 생성
    frame_task = asyncio.create_task(webcam_processor.async_frame_provider(shared_data))

    # 개별 작업 비동기 실행
    depth_task = asyncio.create_task(depth_with_tts.run(shared_data))
    hand_task = asyncio.create_task(run_hand_detection(shared_data))
    yolo_task = asyncio.create_task(yolo_detector.run_detection(shared_data))

    try:
        # 모든 비동기 작업 실행 및 병렬 처리
        await asyncio.gather(frame_task, depth_task, hand_task, yolo_task)
    except KeyboardInterrupt:
        print("Terminating...")
        shared_data['running'] = False  # 모든 작업 중단 신호
        await frame_task  # 프레임 공급 작업 종료 대기

    # 자원 해제
    webcam_processor.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())
