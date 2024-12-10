import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)
mp_drawing = mp.solutions.drawing_utils

# 웹캠 열기
cap = cv2.VideoCapture(0)

# 자동 화이트 밸런스 설정 (선택 사항)
cap.set(cv2.CAP_PROP_AUTO_WB, 1)
# cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4500)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("웹캠을 찾을 수 없습니다.")
        continue

    # 색상 보정 (YCrCb 공간에서 조정)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(image)
    cr = cv2.add(cr, 13)
    cb = cv2.subtract(cb, 20)
    image = cv2.merge([y, cr, cb])
    image = cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)

    # 이미지에서 손 감지 (RGB 변환 필요)
    image.flags.writeable = False
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    image.flags.writeable = True

    # 이미지에 감지된 손 그리기
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 손 그리기 (랜드마크 좌표 변환 불필요)
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
            )

    # 결과 이미지 보여주기
    cv2.imshow("MediaPipe Hands", image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()