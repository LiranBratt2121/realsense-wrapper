from ultralytics import YOLO
import supervision as sv
import cv2
from time import time

model = YOLO("weights/best.pt", verbose=False)

cap = cv2.VideoCapture(r'videos\videoplayback (1).mp4')

box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    start_time = time()

    results = model(frame, verbose=False)[0]

    detections = sv.Detections.from_yolov8(results)
    
    labels = [f"{model.names[class_id]} {confidence:0.2f}" for _, confidence, class_id, _ in detections]
        
    annotated_frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
    
    fps = 1 / (time() - start_time)
    
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    out.write(annotated_frame)

    cv2.imshow("frame", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()

cv2.destroyAllWindows()
