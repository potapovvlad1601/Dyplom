from ultralytics import YOLO


class CowClassifier:
    def __init__(self, model_path, device = "cpu"):
        self.model = YOLO(model_path)
        self.device = device


    def classify(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None, 0.0

        results = self.model.predict(crop, device=self.device, verbose=False)

        for r in results:
            if r.probs is None:
                continue

            top1 = int(r.probs.top1)
            conf = float(r.probs.top1conf)

            return self.model.names[top1], conf

        return None, 0.0