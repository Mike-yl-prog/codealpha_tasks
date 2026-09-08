from ultralytics import YOLO


class PlayerDetector:
    def __init__(self, model_path="yolo11n.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame):
         results = self.model(frame, verbose=false)

         detections = []


         for results in results:
              for box in results.boxes:
                   class_id = int(box.cls[0])
                   confidence = float(box.conf[0])

                   x1,y1,x2, y2 = map (
                    int,
                    box.xyxy[0]
                   )

                   detections.append({
                       "class_id": class_id,
                       "confidence": confidence,
                       "bbox": (x1, y1, x2, y2)
                    })
         return detections
                    