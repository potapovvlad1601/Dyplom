from ultralytics import YOLO


class CowTracker:
    def __init__(self, model_path, tracker_config, device = "cpu"):
        self.model = YOLO(model_path)
        self.tracker_config = tracker_config
        self.class_names = self.model.names
        self.device = device

    def track(self, frame):
        return self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_config,
            conf=0.5,
            verbose=False,
            device = self.device
        )

    def get_cows(self, results):
        cows = []

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.class_names[cls_id]

                if "cow" not in cls_name.lower():
                    continue

                if box.id is None:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id[0])

                cows.append({
                    "id": track_id,
                    "bbox": (x1, y1, x2, y2)
                })

        return cows