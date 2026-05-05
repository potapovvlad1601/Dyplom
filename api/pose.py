from ultralytics import YOLO


class PoseEstimator:
    def __init__(self, model_path, device = "cpu"):
        self.model = YOLO(model_path)
        self.device = device

    def get_keypoints(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return {}

        results = self.model.predict(crop, device = self.device, verbose=False)
        keypoints = {}

        for r in results:
            if r.keypoints is None:
                continue

            kpts = r.keypoints.xy.cpu().numpy()

            for person in kpts:
                for kp_id, (x, y) in enumerate(person):
                    if x > 0 and y > 0:
                        keypoints[kp_id] = (int(x + x1), int(y + y1))

        return keypoints