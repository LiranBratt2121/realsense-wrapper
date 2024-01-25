from ultralytics import YOLO
import os
import cv2
import supervision as sv

model = YOLO(r"weights\best.pt", verbose=True)

path = r'C:\Users\bratt\Documents\note-dataset\test\images'

photos = os.listdir(path)

box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)

while photos:
    photo_url = photos.pop()

    image = cv2.imread(os.path.join(path, photo_url))

    results = model(image)[0]

    detections = sv.Detections.from_yolov8(results)
    
    labels = [f"{model.names[class_id]} {confidence:0.2f}" for _, confidence , class_id, _  in detections]
        
    image = box_annotator.annotate(scene=image, detections=detections, labels=labels)

    cv2.imshow("Image", image)

    key = cv2.waitKey(0)

    if key == ord('q'):
        break

cv2.destroyAllWindows()
