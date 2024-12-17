import pyttsx3
import threading
import cv2
import asyncio
from datetime import datetime  # 현재 시간 출력용
from test_depth import setup_depth_model, process_depth_sections, display_depth_sections

class TextToSpeech:
    def __init__(self, rate=150, volume=0.9, voice_index=0):
        """TTS 엔진 초기화 및 설정"""
        self.engine = pyttsx3.init()

        # 속도 설정
        self.engine.setProperty('rate', rate)

        # 볼륨 설정
        self.engine.setProperty('volume', volume)

        # 음성 설정
        voices = self.engine.getProperty('voices')
        if voice_index < len(voices):
            self.engine.setProperty('voice', voices[voice_index].id)
        else:
            print("Voice index out of range. Using default voice.")

    def speak(self, text):
        """주어진 텍스트를 비동기적으로 음성으로 출력"""
        thread = threading.Thread(target=self._speak_thread, args=(text,))
        thread.start()

    def _speak_thread(self, text):
        """스레드에서 TTS 음성 출력"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] TTS Output: {text}")
        self.engine.say(text)
        self.engine.runAndWait()


class DepthWithTTS:
    def __init__(self, tts):
        """Depth 모델과 TTS를 결합한 클래스"""
        self.depth_processor = setup_depth_model()
        self.tts = tts
        self.last_tts_time = 0  # 마지막 TTS 출력 시간 초기화

    async def run(self, shared_data):
        """비동기적으로 뎁스 모델을 실행하고 결과를 TTS로 출력"""
        while shared_data['running']:
            frame = shared_data.get('frame')
            if frame is None:
                await asyncio.sleep(0)  # 이벤트 루프 양보
                continue

            try:
                # OpenVINO 뎁스 모델 처리
                depth_result = self.depth_processor.process_frame(frame)
                depth_map = (depth_result.squeeze(0) - depth_result.min()) / (depth_result.max() - depth_result.min())
                depth_frame = self.depth_processor.visualize_result(depth_result)

                # 깊이 섹션 분석
                decision = process_depth_sections(depth_map, num_rows=5, num_cols=5, threshold=0.8)

                # TTS로 결과 출력 (7초 딜레이 적용)
                current_time = asyncio.get_event_loop().time()
                if current_time - self.last_tts_time >= 7:  # 7초 경과 확인
                    self.tts.speak(decision)
                    self.last_tts_time = current_time

                    # 현재 시간과 TTS 출력 로그
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{now}] TTS Output: {decision}")

                # 섹션이 표시된 뎁스 이미지
                depth_frame_with_sections = display_depth_sections(
                    depth_frame.copy(), depth_map, num_rows=5, num_cols=5, output_width=1280, output_height=720
                )

                # 텍스트 출력
                cv2.putText(
                    depth_frame_with_sections,
                    decision,
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )

                # 화면 출력
                cv2.imshow("Depth Estimation", depth_frame_with_sections)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    shared_data['running'] = False
                    break

            except Exception as e:
                print(f"Error in unified_depth_with_tts: {e}")
                shared_data['running'] = False
                break

            await asyncio.sleep(0)  # 이벤트 루프 양보

        cv2.destroyAllWindows()
