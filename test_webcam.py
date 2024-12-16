import cv2
from ultralytics import YOLO

model = YOLO('./best.pt')

# 웹캠 열기 (기본 카메라: 0, 뎁스 카메라: 4)
cap = cv2.VideoCapture(4)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # 웹캠에서 프레임을 읽지 못한 경우 루프 종료

    # 이미지 전처리
    img_resized = cv2.resize(frame, (640, 640))  # 모델 입력 크기에 맞게 조정
    
    # 모델에 프레임 전달
    results = model(img_resized)

    # 결과 플로팅
    plots = results[0].plot()

    # 결과 표시
    cv2.imshow("Detection", plots)

    # 'q' 키를 눌러 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 해제 및 모든 윈도우 종료
cap.release()
cv2.destroyAllWindows()
