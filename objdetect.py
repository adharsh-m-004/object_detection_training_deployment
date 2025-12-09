from ultralytics import YOLO
import cv2
import time

model = YOLO('##')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

font = cv2.FONT_HERSHEY_SIMPLEX
prev_frame_time = 0

print("--- Starting Camera Detection ---")
print("Press 'q' to exit the live feed.")
while cap.isOpened():
    success, frame = cap.read()
    
    if success:
        results = model.predict(source=frame, stream=True, verbose=False, conf=0.5)
        for r in results:
            annotated_frame = r.plot()
        new_frame_time = time.time()
        if new_frame_time != prev_frame_time:
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps_text = f"FPS: {fps:.2f}"
            cv2.putText(annotated_frame, fps_text, (10, 30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("YOLOv8 Live Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Warning: Could not read frame from camera.")
        break

cap.release()
cv2.destroyAllWindows()

print("--- Detection Finished ---")
