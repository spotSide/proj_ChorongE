#main
#비동기로 전환

from test_depth import *
from test_hand import *
from test_webcam import *
import asyncio
import cv2


async def frame_provider(webcam_processor, shared_data):
    """비동기적으로 웹캠 프레임을 읽어 공유 메모리에 저장합니다."""
    while shared_data['running']:
        try:
            frame = webcam_processor.read_frame()
            shared_data['frame'] = frame.copy()
        except ValueError as e:
            print(e)
            shared_data['running'] = False
            break
        await asyncio.sleep(0)  # 이벤트 루프 양보


async def main():
    webcam_processor = WebcamProcessor(camera_id=0)
    shared_data = {'frame': None, 'running': True}

    print("Starting async processes...")

    # 프레임 공급 코루틴
    frame_task = asyncio.create_task(frame_provider(webcam_processor, shared_data))

    # Depth Estimation 및 Hand Detection 비동기 함수 시작
    depth_task = asyncio.create_task(unified_depth(shared_data))
    hand_task = asyncio.create_task(run_hand_detection(shared_data))

    try:
        # 모든 작업 완료 대기
        await asyncio.gather(frame_task, depth_task, hand_task)
    except KeyboardInterrupt:
        print("Terminating...")
        shared_data['running'] = False
        await frame_task

    # 자원 해제
    webcam_processor.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())