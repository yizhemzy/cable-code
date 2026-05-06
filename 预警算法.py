import warnings
warnings.filterwarnings('ignore')
import cv2
import numpy as np
from ultralytics import YOLO

class CableDefectWarning:
    def __init__(self, fixed_cable_area=80000, scale_factor=15.0, weight_crack=2.5, weight_peel=0.7):
        self.fixed_cable_area = fixed_cable_area
        self.scale_factor = scale_factor
        self.weight_crack = weight_crack
        self.weight_peel = weight_peel

    def get_cable_area(self, image_bgr):
        try:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return 0
            max_contour = max(contours, key=cv2.contourArea)
            return int(cv2.contourArea(max_contour))
        except Exception:
            return 0

    def get_defect_type_and_weight(self, mask_np):
        if mask_np.dtype != np.uint8:
            mask_np = (mask_np > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return "none", 1.0
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < 20:
            return "none", 1.0
        x, y, w, h = cv2.boundingRect(cnt)
        max_side = max(w, h)
        min_side = min(w, h)
        ratio = max_side / (min_side + 1e-6)
        perimeter = cv2.arcLength(cnt, True)
        compactness = (perimeter * perimeter) / (4 * np.pi * area + 1e-6)
        if ratio >= 3.0 or (area < 2000 and compactness > 2.5):
            return "crack", self.weight_crack
        else:
            return "peel", self.weight_peel

    def compute_defect_area(self, mask_np):
        return int(np.sum(mask_np > 0.5))

    def get_warning_level(self, risk):
        if risk < 0.1:
            return 0, "安全无隐患"
        elif risk < 0.3:
            return 1, "轻微隐患"
        elif risk < 0.6:
            return 2, "中度故障"
        else:
            return 3, "严重险情"

    def judge(self, image_bgr, mask_np, conf):
        cable_area = self.get_cable_area(image_bgr)
        if cable_area == 0:
            cable_area = self.fixed_cable_area

        defect_type, weight = self.get_defect_type_and_weight(mask_np)
        defect_area = self.compute_defect_area(mask_np)
        relative_area = defect_area / (cable_area + 1e-6)
        risk = conf * weight * relative_area * self.scale_factor

        level, info = self.get_warning_level(risk)
        return {
            "defect_type": defect_type,
            "defect_area_pixel": defect_area,
            "cable_area_pixel": cable_area,
            "relative_area": round(relative_area, 6),
            "confidence": round(conf, 4),
            "risk_score": round(risk, 4),
            "warning_level": level,
            "warning_info": info,
            "is_alarm": level >= 2
        }

if __name__ == '__main__':
    model = YOLO('D:/代码/ultralytics/runs/train/stage33/weights/best.pt')
    warning_engine = CableDefectWarning(fixed_cable_area=80000, scale_factor=15.0)

    results = model.predict(
        source='D:/代码/自建管道内电缆数据集/valid/images',
        imgsz=640,
        project='runs/segment',
        name='exp',
        save=True,
        save_txt=False,
    )

    for i, result in enumerate(results):
        print(f"\n===== 图片 {i + 1}: {result.path} =====")
        if result.masks is None:
            print("未检测到任何缺陷")
            continue
        orig_img = result.orig_img
        masks = result.masks.data.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        for j, (mask, conf) in enumerate(zip(masks, confs)):
            warn = warning_engine.judge(orig_img, mask, conf)
            print(f"  缺陷 {j + 1}:")
            print(f"    类型: {warn['defect_type']}")
            print(f"    置信度: {warn['confidence']}")
            print(f"    缺陷面积: {warn['defect_area_pixel']} px")
            print(f"    电缆面积: {warn['cable_area_pixel']} px")
            print(f"    相对面积: {warn['relative_area']}")
            print(f"    风险值: {warn['risk_score']}")
            print(f"    预警等级: {warn['warning_level']} - {warn['warning_info']}")
            if warn['is_alarm']:
                print(f"    ⚠️ 触发警告！")
            print()
    print("推理完成！")