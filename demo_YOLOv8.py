from ultralytics import YOLO
import cv2

model = YOLO(
    r"D:\University\Nam 4\HK2\Nhap mon tri tue nhan tao\Model\first_300.pt")

cap = cv2.VideoCapture(
    r"D:\University\Nam 4\HK2\Nhap mon tri tue nhan tao\Model\BiểnĐông\IMG_6057.MOV")

# Định nghĩa codec và tạo đối tượng VideoWriter để ghi video
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec để mã hóa video
out = cv2.VideoWriter(
    r'D:\University\Nam 4\HK2\Nhap mon tri tue nhan tao\Model\output.avi', fourcc, 20.0, (640, 640))

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break
    new_width = 640
    new_height = 640

    # Thay đổi kích thước khung hình
    resized_frame = cv2.resize(frame, (new_width, new_height))

    results = model.predict(resized_frame, conf=0.5)

    # Vẽ bouding box

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        confident_score = float(box.conf.cpu().numpy())
        label_id = int(box.cls.cpu().numpy())
        label = model.names[label_id]

        cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(resized_frame, f'{label} {confident_score:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    out.write(resized_frame)
    cv2.imshow("Video", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
out.release()
cv2.destroyAllWindows()
