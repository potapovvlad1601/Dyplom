from ultralytics import YOLO


class PoseEstimator:
    def __init__(self, model_path, device="cpu"):
        self.model = YOLO(model_path)
        self.device = device

    def get_keypoints(self, frame, bbox):
        x1, y1, x2, y2 = bbox

        h, w = frame.shape[:2]

        bw = x2 - x1
        bh = y2 - y1

        pad_x = int(bw * 0.1)
        pad_y = int(bh * 0.1)

        px1 = max(0, x1 - pad_x)
        py1 = max(0, y1 - pad_y)
        px2 = min(w, x2 + pad_x)
        py2 = min(h, y2 + pad_y)

        crop = frame[py1:py2, px1:px2]

        if crop.size == 0:
            return {}

        results = self.model.predict(crop, device=self.device, verbose=False)

        keypoints = {}

        for r in results:
            if r.keypoints is None:
                continue

            kpts = r.keypoints.xy.cpu().numpy()

            for person in kpts:
                for kp_id, (x, y) in enumerate(person):
                    if x > 0 and y > 0:
                        keypoints[kp_id] = (
                            int(x + px1),
                            int(y + py1)
                        )

        return keypoints
