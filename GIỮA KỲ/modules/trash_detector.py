
import os
import sys
import tempfile
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Thêm thư mục gốc TEST vào đường dẫn tìm module
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.decorators import measure_time

try:
    from modules.image_processor import ImageProcessor
except ImportError:
    try:
        from image_processor import ImageProcessor
    except ImportError:
        ImageProcessor = None

WORKSPACE = "thanh-phuong-tf1ig"
WORKFLOW_ID = "general-segmentation-api-2"

# Tên tiếng Anh gửi cho mô hình (mô hình hiểu tiếng Anh tốt hơn) -> (nhóm, tên tiếng Việt).
# Muốn nhận diện thêm vật gì thì thêm một dòng ở đây.
VAT_THE = {
    "plastic bottle": ("Nhựa", "Chai nhựa"),
    "crushed plastic bottle": ("Nhựa", "Chai nhựa bẹp"),
    "glass bottle": ("Thủy tinh", "Chai thủy tinh"),
    "aluminum can": ("Kim loại", "Lon nhôm"),
    "tin can": ("Kim loại", "Lon thiếc"),
    "metal can": ("Kim loại", "Lon kim loại"),
    "cardboard": ("Giấy", "Bìa cứng"),
    "paper": ("Giấy", "Giấy"),
    "stack of paper": ("Giấy", "Chồng giấy")
}

MAU_NHOM = {
    "Nhựa": (33, 150, 243),
    "Thủy tinh": (0, 200, 150),
    "Kim loại": (255, 160, 0),
    "Giấy": (200, 90, 220),
    "Khác": (150, 150, 150),
}


def _gop_nhom(ten_anh):
    """Dự phòng: nếu mô hình trả tên lạ, đoán nhóm theo từ khóa."""
    t = ten_anh.lower()
    if "glass" in t:
        return "Thủy tinh"
    if "can" in t or "metal" in t:
        return "Kim loại"
    if "paper" in t or "cardboard" in t:
        return "Giấy"
    if "plastic" in t:
        return "Nhựa"
    return "Khác"


def _tim_vat(x):
    """Tìm mọi mục có 'class' và 'confidence' trong kết quả workflow."""
    ra = []
    if isinstance(x, dict):
        if "class" in x and "confidence" in x:
            ra.append(x)
        for v in x.values():
            ra += _tim_vat(v)
    elif isinstance(x, list):
        for v in x:
            ra += _tim_vat(v)
    return ra


def _sang_rgb(image):
    
    if isinstance(image, (str, os.PathLike)):
        image = Image.open(Path(image))
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            raise ValueError("Ảnh NumPy phải có dạng (cao, rộng, 3) hoặc (cao, rộng, 4).")
        arr = arr[..., :3][..., ::-1]  # BGR -> RGB
        return np.ascontiguousarray(arr.astype(np.uint8, copy=False))
    raise TypeError("image phải là PIL.Image, numpy.ndarray hoặc đường dẫn ảnh.")


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _bo_dau(s):
    """Bỏ dấu tiếng Việt (dùng khi máy không có font hỗ trợ dấu)."""
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _tai_font(size):
    """Tìm font có hỗ trợ tiếng Việt. Trả về (font, co_ho_tro_dau)."""
    for ten in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf", "NotoSans-Regular.ttf"):
        try:
            return ImageFont.truetype(ten, size), True
        except OSError:
            continue
    return ImageFont.load_default(), False


class TrashDetector:
    def __init__(
        self,
        model_path=None,  # giữ để tương thích code cũ, không dùng
        confidence_threshold=0.3,
        image_size=640,
        iou_threshold=0.6,
        dung_tien_xu_ly=True,
        api_key=None,
        api_url=None,
    ):
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold phải nằm trong khoảng 0 đến 1.")
        if image_size <= 0:
            raise ValueError("image_size phải lớn hơn 0.")

        from inference_sdk import InferenceConfiguration, InferenceHTTPClient

        self.api_key = api_key or os.environ.get("ROBOFLOW_API_KEY")
        if not self.api_key:
            raise RuntimeError("Chưa đặt biến môi trường ROBOFLOW_API_KEY.")
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.iou_threshold = iou_threshold
        self.processor = (
            ImageProcessor(target_size=image_size)
            if dung_tien_xu_ly and ImageProcessor is not None
            else None
        )
        self.client = InferenceHTTPClient(
            api_url=api_url or os.environ.get("ROBOFLOW_API_URL", "https://serverless.roboflow.com"),
            api_key=self.api_key,
        ).configure(InferenceConfiguration(api_key_transport="header"))

    # ------------------------------------------------------------------
    def _hinh_hoc(self, cao, rong):
        """Phép biến đổi từ ảnh gốc sang ảnh gửi đi: (hệ số thu phóng, lề trái, lề trên).
        Công thức giống ImageProcessor.process (letterbox ở giữa)."""
        if self.processor is None:
            return 1.0, 0, 0
        t = self.processor.target_size
        scale = min(t / cao, t / rong)
        rh = max(1, round(cao * scale))
        rw = max(1, round(rong * scale))
        return scale, (t - rw) // 2, (t - rh) // 2
    @measure_time
    def detect(self, image, confidence_threshold=None):
        nguong = self.confidence_threshold if confidence_threshold is None else confidence_threshold
        rgb = _sang_rgb(image)
        cao, rong = rgb.shape[:2]

        anh_gui = self.processor.process(rgb) if self.processor else rgb
        scale, le_trai, le_tren = self._hinh_hoc(cao, rong)

        # Lưu ảnh tạm rồi gửi đường dẫn (cách đã chạy ổn)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            duong_dan = tmp.name
        try:
            Image.fromarray(anh_gui).save(duong_dan, format="JPEG", quality=95)
            kq = self.client.run_workflow(
                workspace_name=WORKSPACE,
                workflow_id=WORKFLOW_ID,
                images={"image": duong_dan},
                parameters={"classes": ", ".join(VAT_THE.keys())},
                use_cache=True,
            )
        finally:
            os.remove(duong_dan)

        ra, da_co = [], set()
        for p in _tim_vat(kq):
            if p["confidence"] < nguong:
                continue
            x, y, w, h = p.get("x"), p.get("y"), p.get("width"), p.get("height")
            if None in (x, y, w, h):
                continue
            khoa = (p["class"], x, y, w, h)
            if khoa in da_co:  # bỏ kết quả bị lặp y hệt
                continue
            da_co.add(khoa)

            # x, y là TÂM khung trên ảnh đã gửi -> đổi về góc và quy về ảnh gốc
            x1 = (x - w / 2 - le_trai) / scale
            y1 = (y - h / 2 - le_tren) / scale
            x2 = (x + w / 2 - le_trai) / scale
            y2 = (y + h / 2 - le_tren) / scale
            bbox = [
                int(max(0, min(rong, round(x1)))),
                int(max(0, min(cao, round(y1)))),
                int(max(0, min(rong, round(x2)))),
                int(max(0, min(cao, round(y2)))),
            ]
            nhom, ten_vn = VAT_THE.get(p["class"], (_gop_nhom(p["class"]), p["class"]))
            ra.append(
                {
                    "class_name": nhom,
                    "vat_the": ten_vn,
                    "confidence": round(float(p["confidence"]), 4),
                    "bbox": bbox,
                }
            )
        return self._loc_trung(ra)

    def _loc_trung(self, vat):
        """Bỏ khung chồng lên nhau trên cùng một vật (giữ khung điểm cao hơn)."""
        giu = []
        for v in sorted(vat, key=lambda d: d["confidence"], reverse=True):
            if all(_iou(v["bbox"], g["bbox"]) < self.iou_threshold for g in giu):
                giu.append(v)
        return giu

    # ------------------------------------------------------------------
    @staticmethod
    def ve_ket_qua(image, detections):
        """Vẽ khung và nhãn tiếng Việt lên bản sao của ảnh. Trả về ảnh PIL (RGB)."""
        anh = Image.fromarray(_sang_rgb(image))
        d = ImageDraw.Draw(anh)
        w_anh = anh.width
        font, co_dau = _tai_font(max(14, w_anh // 40))
        day = max(2, w_anh // 300)

        for det in detections:
            if not det.get("bbox"):
                continue
            x1, y1, x2, y2 = det["bbox"]
            mau = MAU_NHOM.get(det["class_name"], MAU_NHOM["Khác"])
            d.rectangle([x1, y1, x2, y2], outline=mau, width=day)

            nhan = f'{det["vat_the"]} {det["confidence"]:.0%}'
            if not co_dau:
                nhan = _bo_dau(nhan)
            hop = d.textbbox((x1, y1), nhan, font=font)
            cao_nhan = hop[3] - hop[1] + 6
            rong_nhan = (hop[2] - hop[0]) + 8
            y_nhan = y1 - cao_nhan if y1 - cao_nhan >= 0 else y1
            x_nhan = max(0, min(x1, w_anh - rong_nhan))  # không để nhãn tràn khỏi mép ảnh
            d.rectangle([x_nhan, y_nhan, x_nhan + rong_nhan, y_nhan + cao_nhan], fill=mau)
            d.text((x_nhan + 4, y_nhan + 2), nhan, fill="white", font=font)
        return anh
