from ultralytics import YOLO
import cv2
import winsound   

model = YOLO(r"C:\Users\Ankit kumar\Downloads\model\runs\detect\train15\weights\best.pt")

cap = cv2.VideoCapture(0)

DANGER_CLASSES = {"sleeping", "unsafe_equipment", "smoking", "phone_usage", "no_helmet"}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    detected_danger = False

    for box in results.boxes:
        cls = int(box.cls)
        label = model.names[cls]
        conf = float(box.conf)

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if label in DANGER_CLASSES and conf > 0.50:
            detected_danger = True

    if detected_danger:
        winsound.Beep(1000, 500)     

    cv2.imshow("Safety Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
