import pyttsx3
import threading
import cv2
import asyncio
import time
from datetime import datetime  # 현재 시간 출력용
from test_depth import setup_depth_model, process_depth_sections, display_depth_sections
import sys
from io import StringIO
from queue import Queue

class TextToSpeech:
    def __init__(self, rate=150, volume=0.9, voice_index=0):
        """TTS 엔진 초기화 및 설정"""
        self.engine = pyttsx3.init()
        self.queue = Queue()  # TTS 메시지 관리 큐

        # 속도 및 볼륨 설정
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)

        # 음성 설정
        voices = self.engine.getProperty('voices')
        if voice_index < len(voices):
            self.engine.setProperty('voice', voices[voice_index].id)
        else:
            print("Voice index out of range. Using default voice.")

        # 상태 변수
        self.last_tts_time = 0  # 마지막 TTS 실행 시간
        self.last_avoid_time = 0  # 마지막 Avoid 메시지 큐 추가 시간
        self.is_tts_busy = False  # 현재 TTS 실행 중인지 여부

        # TTS 큐 처리 스레드 시작
        self.tts_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.tts_thread.start()

    def speak(self, text, priority=False):
        """주어진 텍스트를 TTS 큐에 추가"""
        if text is None:  # None 상태는 처리하지 않음
            return

        if priority:
            # 최우선 메시지는 큐를 비우고 바로 추가
            with self.queue.mutex:
                self.queue.queue.clear()  # 기존 메시지 제거
            self.queue.put(text)  # 최우선 메시지 추가
        else:
            # Avoid 메시지는 5초에 한 번만 추가
            if "Avoid" in text:
                current_time = time.time()
                if current_time - self.last_avoid_time < 5:
                    return  # 5초 이내에는 메시지 추가 안 함
                self.last_avoid_time = current_time

            # 일반 메시지는 큐에 중복되지 않게 추가
            if text not in self.queue.queue:
                self.queue.put(text)

    def _process_queue(self):
        """큐에서 메시지를 꺼내 순차적으로 음성 출력"""
        while True:
            text = self.queue.get()  # 메시지 가져오기
            self.is_tts_busy = True
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{now}] TTS Output: {text}")  # 터미널 출력
            self.engine.say(text)
            self.engine.runAndWait()
            self.is_tts_busy = False
            time.sleep(0.5)  # 메시지 간 간격 추가

import sys
import asyncio
from datetime import datetime

class FlagMonitor:
    def __init__(self, tts):
        self.catch_flag = False  # Catch 플래그 상태
        self.detect_flag = False  # Detect 플래그 상태
        self.previous_combined_state = False  # 이전 결합 상태
        self.tts = tts  # TTS 인스턴스
        self.last_detected_class = None  # 마지막 감지된 클래스 이름
        self.is_priority_tts_active = False  # 최우선 TTS 활성화 상태
        self.original_stdout = sys.stdout  # 원래 stdout 저장
        sys.stdout = self  # stdout 리다이렉트

    def write(self, text):
        """터미널 출력 감지 및 플래그 상태 업데이트"""
        # Catch 플래그 상태 업데이트
        if "catch flag - 5s" in text:
            self.catch_flag = True
        elif "catch end" in text:
            self.catch_flag = False

        # Detect 플래그 상태 업데이트
        if "class detect flag - 5s" in text:
            self.detect_flag = True
        elif "class flag end" in text:
            self.detect_flag = False

        # 클래스 이름 감지 (Detected: cider (0.90))
        if "Detected:" in text:
            parts = text.split("Detected:")
            if len(parts) > 1:
                class_name = parts[1].strip().split(" ")[0]  # 클래스 이름 추출
                self.last_detected_class = class_name

        # 원래 stdout에 출력
        self.original_stdout.write(text)

    def flush(self):
        """flush 호출을 위한 메서드 (stdout 요구사항)"""
        self.original_stdout.flush()

    async def monitor_flags(self):
        """플래그 상태를 지속적으로 모니터링 (둘 다 True일 때 TTS 출력)"""
        while True:
            # 현재 상태 결합
            current_combined_state = (self.catch_flag and self.detect_flag)

            # 둘 다 True일 때만 처리
            if current_combined_state and not self.previous_combined_state:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{now}] Both Catch and Detect Flags are True!")

                # TTS로 '[class name] catch' 출력 (최우선순위)
                if self.last_detected_class and not self.tts.is_tts_busy:
                    tts_message = f"{self.last_detected_class} catch"
                    self.tts.speak(tts_message)

            # 상태가 False로 유지되거나 다시 False로 변경된 경우
            self.previous_combined_state = current_combined_state

            await asyncio.sleep(0.1)  # 0.1초마다 상태 확인

class DepthWithTTS:
    def __init__(self, tts):
        """Depth 모델과 TTS를 결합한 클래스"""
        self.depth_processor = setup_depth_model()
        self.tts = tts

    async def run(self, shared_data):
        """비동기적으로 뎁스 모델을 실행하고 결과를 TTS로 출력"""
        while shared_data['running']:
            frame = shared_data['frame']
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

                # TTS로 결과 출력
                if decision:
                    self.tts.speak(decision)

                # 섹션이 표시된 뎁스 이미지
                depth_frame_with_sections = display_depth_sections(
                    depth_frame.copy(), depth_map, num_rows=5, num_cols=5, output_width=1280, output_height=720
                )

                # 텍스트 출력
                if decision:
                    cv2.putText(
                        depth_frame_with_sections,
                        decision,
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 255), 2, cv2.LINE_AA
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
