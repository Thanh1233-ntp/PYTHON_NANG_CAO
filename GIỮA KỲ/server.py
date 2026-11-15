"""
Backend nhận diện rác: tải ảnh lên -> trả kết quả JSON.
Không cần train, không cần tải dataset. Gọi workflow dò theo tên chữ trên Roboflow.

Chạy:
    pip install -r requirements.txt
    (Windows)  setx ROBOFLOW_API_KEY "KEY_MOI"   rồi mở lại cửa sổ lệnh
    (Mac/Linux) export ROBOFLOW_API_KEY="KEY_MOI"
    uvicorn server:app --port 8000

Thử ngay trên trình duyệt: http://localhost:8000/docs  (bấm POST /nhan-dien -> Try it out -> chọn ảnh)
"""
import os
import tempfile
from collections import Counter

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from inference_sdk import InferenceConfiguration, InferenceHTTPClient

# ----- Cấu hình -----
# Mặc định gọi qua Roboflow. Nếu bạn tự chạy inference server trên máy,
# đặt ROBOFLOW_API_URL=http://localhost:9001
API_URL = os.environ.get("ROBOFLOW_API_URL", "https://serverless.roboflow.com")
WORKSPACE = "thanh-phuong-tf1ig"
WORKFLOW_ID = "general-segmentation-api-2"

# Tên vật cần dò (tiếng Anh cho mô hình hiểu tốt). Thêm/bớt ở đây để chỉnh độ chính xác.
TEN_VAT = "plastic bottle, glass bottle, aluminum can, cardboard, paper"

NGUONG_MAC_DINH = 0.3

# ----- Kết nối Roboflow -----
API_KEY = os.environ.get("ROBOFLOW_API_KEY")
client = None
if API_KEY:
    client = InferenceHTTPClient(api_url=API_URL, api_key=API_KEY).configure(
        InferenceConfiguration(api_key_transport="header")
    )

app = FastAPI(title="Nhận diện rác")

# Cho phép giao diện (chạy ở địa chỉ khác) gọi vào. Khi đưa lên thật, thay "*" bằng địa chỉ web của bạn.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def gop_nhom(ten: str) -> str:
    """Gộp tên vật về 4 nhóm. Kiểm tra 'glass' trước để chai thủy tinh không bị tính là nhựa."""
    t = ten.lower()
    if "glass" in t:
        return "Thủy tinh"
    if "can" in t or "metal" in t:
        return "Kim loại"
    if "paper" in t or "cardboard" in t:
        return "Giấy"
    if "plastic" in t:
        return "Nhựa"
    return "Khác"


def tim_vat(x):
    """Tìm mọi mục có 'class' và 'confidence' trong kết quả workflow (không phụ thuộc cấu trúc)."""
    ra = []
    if isinstance(x, dict):
        if "class" in x and "confidence" in x:
            ra.append(x)
        for v in x.values():
            ra += tim_vat(v)
    elif isinstance(x, list):
        for v in x:
            ra += tim_vat(v)
    return ra


@app.get("/")
def kiem_tra():
    return {"trang_thai": "ok", "da_co_api_key": client is not None}


@app.post("/nhan-dien")
def nhan_dien(
    anh: UploadFile = File(...),
    nguong: float = NGUONG_MAC_DINH,
    debug: bool = False,
):
    if client is None:
        raise HTTPException(500, "Chưa đặt biến môi trường ROBOFLOW_API_KEY.")
    if not (anh.content_type or "").startswith("image/"):
        raise HTTPException(400, "File tải lên phải là ảnh.")

    duoi = os.path.splitext(anh.filename or "")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=duoi) as tmp:
        tmp.write(anh.file.read())
        duong_dan = tmp.name

    try:
        kq = client.run_workflow(
            workspace_name=WORKSPACE,
            workflow_id=WORKFLOW_ID,
            images={"image": duong_dan},
            parameters={"classes": TEN_VAT},
            use_cache=True,
        )
    except Exception as e:
        raise HTTPException(502, f"Lỗi khi gọi dịch vụ nhận diện: {e}")
    finally:
        os.remove(duong_dan)

    vat = []
    for p in tim_vat(kq):
        if p["confidence"] >= nguong:
            vat.append(
                {
                    "nhom": gop_nhom(p["class"]),
                    "ten": p["class"],
                    "do_tin_cay": round(p["confidence"], 2),
                    "x": p.get("x"),
                    "y": p.get("y"),
                    "rong": p.get("width"),
                    "cao": p.get("height"),
                }
            )

    ket_qua = {
        "so_vat": len(vat),
        "dem_theo_nhom": dict(Counter(v["nhom"] for v in vat)),
        "vat": vat,
    }
    if debug:
        ket_qua["du_lieu_tho"] = kq  # xem cấu trúc thật nếu tọa độ ra None
    return ket_qua
