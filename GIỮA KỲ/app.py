import io
import zipfile
import os
import time

import pandas as pd
import streamlit as st
from PIL import Image
from modules.trash_detector import TrashDetector
from modules.trash_evaluator import TrashEvaluator

st.set_page_config(page_title="EcoVision AI | Nhận diện rác thải", page_icon="♻️", layout="wide")

st.markdown("""
<style>
/* 1. Nền tổng thể trang web (Trắng sáng thanh lịch) */
.stApp, header[data-testid="stHeader"] {
    background-color: #f7faf8 !important;
    color: #1a382b !important;
}

/* 2. Thanh bên Sidebar (Màu xanh nhạt dịu mắt) */
[data-testid="stSidebar"] {
    background-color: #ebf5ee !important;
}

/* 3. Khối Hero trên cùng (Banner xanh lá tươi, chữ trắng nổi bật) */
/* Khối Hero phong cách sáng, nhạt và thanh lịch */
.hero {
    padding: 1.8rem 2rem;
    border: 1px solid #cce8dc;
    border-radius: 20px;
    background: linear-gradient(135deg, #eaf6f0, #d9efe3);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
    margin-bottom: 1.5rem;
}
.eyebrow {
    color: #2b7a55 !important;
    text-transform: uppercase;
    letter-spacing: .15em;
    font-size: 0.8rem;
    font-weight: 700;
}
.hero h1 {
    margin: .35rem 0;
    color: #0f3d2e !important;
    font-size: 2.5rem;
    font-weight: 800;
}
.hero p {
    color: #285441 !important;
    margin: 0;
    font-size: 1.05rem;
}

/* 4. Thẻ Metric & Khung danh mục hỗ trợ (Nền trắng, viền xanh nhẹ, chữ đậm dễ nhìn) */
.metric {
    background: #ffffff;
    border: 1px solid #d1e7dd;
    border-radius: 14px;
    padding: 1.2rem;
    min-height: 100px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04);
}
.metric-label {
    color: #4b7a63;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
}
.metric-value {
    color: #0f3d2e !important;
    font-size: 1.6rem;
    font-weight: 750;
    margin-top: 0.3rem;
}

/* 5. Khung tải tệp (File Uploader) nền trắng, viền rõ ràng */
[data-testid="stFileUploadDropzone"] {
    background-color: #ffffff !important;
    border: 1.5px dashed #68b684 !important;
    border-radius: 14px !important;
}
[data-testid="stFileUploadDropzone"] * {
    color: #264e36 !important;
}
[data-testid="stFileUploadDropzone"] button {
    background-color: #ebf5ee !important;
    color: #1b5e3f !important;
    border: 1px solid #95d5b2 !important;
    border-radius: 8px;
    font-weight: 600;
}

/* 6. Nút bấm chính (Xanh lá đậm, chữ trắng) */
div.stButton > button[kind="primary"] {
    background: #1b5e3f !important;
    color: #ffffff !important;
    border: none;
    border-radius: 10px;
    padding: 0.5rem 1.2rem;
    font-weight: 700;
    transition: all 0.2s ease;
}
div.stButton > button[kind="primary"]:hover {
    background: #154931 !important;
    color: #ffffff !important;
}

/* 7. Màu tiêu đề h1, h2, h3 và chữ thông thường */
h1, h2, h3, h4, p, span, label {
    color: #153826;
}
</style>
""", unsafe_allow_html=True)

# --- API key: lấy từ biến môi trường, hoặc từ .streamlit/secrets.toml ---
if not os.environ.get("ROBOFLOW_API_KEY"):
    try:
        os.environ["ROBOFLOW_API_KEY"] = st.secrets["ROBOFLOW_API_KEY"]
    except Exception:
        pass
KEY_OK = bool(os.environ.get("ROBOFLOW_API_KEY"))


@st.cache_resource
def load_detector(conf, imgsz):
    return TrashDetector(confidence_threshold=conf, image_size=imgsz)


with st.sidebar:
    st.markdown("## ♻️ EcoVision AI")
    st.caption("PHÒNG LAB NHẬN DIỆN RÁC THẢI")
    st.divider()
    confidence = st.slider("Ngưỡng độ tin cậy", 0.05, 0.95, 0.30, 0.05)
    image_size = st.select_slider("Kích thước ảnh", options=[320, 416, 512, 640, 800, 960, 1024], value=640)
    st.caption("Ngưỡng thấp hơn có thể phát hiện nhiều vật thể hơn nhưng dễ xuất hiện kết quả sai.")
    if KEY_OK:
        st.success("Đã kết nối dịch vụ nhận diện")
    else:
        st.error("Chưa có khóa ROBOFLOW_API_KEY. Hãy đặt biến môi trường hoặc file .streamlit/secrets.toml")

st.markdown("""
<div class="hero">
<div class="eyebrow">Thị giác máy tính vì môi trường</div>
<h1>Nhìn rác thải theo cách mới.</h1>
<p>Tải ảnh lên, nhận diện các loại rác thải và khám phá kết quả dự đoán của mô hình.</p>
</div>
""", unsafe_allow_html=True)

upload_col, info_col = st.columns([1.3, 1], gap="large")
with upload_col:
    st.subheader("01 · Tải ảnh lên")
    uploaded_files = st.file_uploader(
        "Chọn một hoặc nhiều ảnh",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
    )
with info_col:
    st.subheader("02 · Các lớp được hỗ trợ")
    st.markdown('<div class="metric"><div class="metric-label">DANH MỤC CỦA MÔ HÌNH</div><div class="metric-value" style="font-size:1.2rem">Thủy tinh · Kim loại</div><div class="metric-value" style="font-size:1.2rem">Giấy · Nhựa</div></div>', unsafe_allow_html=True)

if uploaded_files:
    st.markdown(f"### Đã chọn {len(uploaded_files)} ảnh")
    with st.expander("Xem trước các ảnh đã chọn"):
        preview_cols = st.columns(min(3, len(uploaded_files)))
        for idx, uploaded in enumerate(uploaded_files):
            with preview_cols[idx % len(preview_cols)]:
                st.image(uploaded, caption=uploaded.name, use_container_width=True)

    if st.button("✨ Phân tích tất cả ảnh", type="primary", use_container_width=True):
        if not KEY_OK:
            st.error("Chưa có khóa ROBOFLOW_API_KEY nên chưa thể nhận diện.")
            st.stop()

        try:
            detector = load_detector(confidence, image_size)
            evaluator = TrashEvaluator()
        except Exception as e:
            st.error("Không thể khởi tạo hệ thống nhận diện.")
            st.exception(e)
            st.stop()

        progress = st.progress(0, text="Đang chuẩn bị phân tích...")
        results = []
        errors = []
        all_rows = []
        start_all = time.perf_counter()

        for idx, uploaded in enumerate(uploaded_files):
            progress.progress(idx / len(uploaded_files), text=f"Đang phân tích ảnh {idx + 1}/{len(uploaded_files)}: {uploaded.name}")
            try:
                uploaded.seek(0)
                image = Image.open(uploaded).convert("RGB")
                start = time.perf_counter()
                detections = detector.detect(image)
                annotated = detector.ve_ket_qua(image, detections)
                evaluation = evaluator.evaluate(detections)
                elapsed = time.perf_counter() - start
                results.append({
                    "name": uploaded.name, "image": image, "annotated": annotated,
                    "detections": detections, "evaluation": evaluation,
                    "elapsed": elapsed,
                })
                for det in detections:
                    all_rows.append({"Ảnh": uploaded.name, **det})
            except Exception as e:
                errors.append((uploaded.name, str(e)))

        progress.progress(1.0, text="Đã hoàn tất phân tích.")
        elapsed_all = time.perf_counter() - start_all

        if results:
            st.markdown("### Tổng quan kết quả")
            total_objects = sum(len(r["detections"]) for r in results)
            total_classes = len({d["class_name"] for r in results for d in r["detections"]})
            metric_cols = st.columns(4)
            for col, label, value in zip(metric_cols,
                ["ẢNH ĐÃ XỬ LÝ", "TỔNG VẬT THỂ", "SỐ NHÓM RÁC", "TỔNG THỜI GIAN"],
                [str(len(results)), str(total_objects), str(total_classes), f"{elapsed_all:.2f}s"]):
                with col:
                    st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)

            st.markdown("### Kết quả từng ảnh")
            for idx, result in enumerate(results, 1):
                detections = result["detections"]
                evaluation = result["evaluation"]
                with st.expander(f'Ảnh {idx}: {result["name"]} — {len(detections)} vật thể', expanded=(idx == 1)):
                    c1, c2 = st.columns(2, gap="large")
                    with c1:
                        st.markdown("#### Ảnh gốc")
                        st.image(result["image"], use_container_width=True)
                    with c2:
                        st.markdown("#### Kết quả AI nhận diện")
                        st.image(result["annotated"], use_container_width=True)
                    st.caption(f'Thời gian xử lý: {result["elapsed"]:.2f} giây')

                    if detections:
                        df = pd.DataFrame(detections)
                        st.markdown("#### Thống kê theo lớp")
                        chart_col, table_col = st.columns([0.8, 1.2])
                        with chart_col:
                            st.bar_chart(df["class_name"].value_counts())
                        with table_col:
                            shown = df.rename(columns={
                                "class_name": "Nhóm", "vat_the": "Vật thể",
                                "confidence": "Độ tin cậy", "bbox": "Khung bao",
                            })
                            shown["Độ tin cậy"] = shown["Độ tin cậy"].map(lambda x: f"{x:.1%}")
                            shown["Khung bao"] = shown["Khung bao"].map(lambda b: ", ".join(map(str, b)) if b else "—")
                            st.dataframe(shown, use_container_width=True, hide_index=True)

                        st.markdown("#### Đánh giá kết quả")
                        ev_cols = st.columns(3)
                        for col, label, value in zip(ev_cols,
                            ["NHÓM CHÍNH", "MỨC TIN CẬY CHUNG", "VẬT CẦN KIỂM TRA LẠI"],
                            [str(evaluation["nhom_chinh"]),
                             f'{evaluation["muc_tin_cay"]} ({evaluation["do_tin_cay_tb"]:.0%})',
                             str(evaluation["so_vat_can_kiem_tra"])]):
                            with col:
                                st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value" style="font-size:1.2rem">{value}</div></div>', unsafe_allow_html=True)

                        st.markdown("#### Nhận xét")
                        for line in evaluation["nhan_xet"]:
                            st.markdown(f"- {line}")
                        if evaluation["theo_nhom"]:
                            group_df = pd.DataFrame([
                                {"Nhóm": n, "Số lượng": v["so_luong"], "Tỉ lệ": f'{v["ty_le"]:.0%}',
                                 "Tin cậy TB": f'{v["do_tin_cay_tb"]:.1%}',
                                 "Tin cậy thấp nhất": f'{v["do_tin_cay_thap_nhat"]:.1%}'}
                                for n, v in evaluation["theo_nhom"].items()
                            ])
                            st.dataframe(group_df, use_container_width=True, hide_index=True)
                        if evaluation["huong_dan"]:
                            with st.expander("♻️ Gợi ý xử lý theo nhóm (mang tính tham khảo)"):
                                for nhom, guide in evaluation["huong_dan"].items():
                                    st.markdown(f"**{nhom}:** {guide}")
                    else:
                        st.info("Không phát hiện vật thể nào trong ảnh này.")

            st.markdown("### Tải kết quả")
            if all_rows:
                csv_df = pd.DataFrame(all_rows)
                csv_df["bbox"] = csv_df["bbox"].map(lambda b: " ".join(map(str, b)) if b else "")
                csv_df = csv_df.rename(columns={
                    "class_name": "nhom", "vat_the": "vat_the",
                    "confidence": "do_tin_cay", "bbox": "khung_bao_x1_y1_x2_y2",
                })
                st.download_button("⬇️ Tải thống kê tất cả ảnh (CSV)",
                    csv_df.to_csv(index=False).encode("utf-8-sig"),
                    "ket_qua_nhieu_anh.csv", "text/csv", use_container_width=True)

            archive_buffer = io.BytesIO()
            with zipfile.ZipFile(archive_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for idx, result in enumerate(results, 1):
                    image_buffer = io.BytesIO()
                    result["annotated"].save(image_buffer, format="PNG")
                    archive.writestr(f'{idx:03d}_ket_qua.png', image_buffer.getvalue())
            st.download_button("⬇️ Tải tất cả ảnh đã đánh dấu (ZIP)",
                archive_buffer.getvalue(), "anh_nhan_dien.zip", "application/zip",
                use_container_width=True)

        if errors:
            st.warning(f"Có {len(errors)} ảnh không xử lý được.")
            for name, error in errors:
                st.error(f"{name}: {error}")
else:
    st.info("Hãy tải một hoặc nhiều ảnh lên ở phía trên để bắt đầu.")
    st.markdown("""
    <div class="metric"><div class="metric-label">BẮT ĐẦU NHANH</div>
    <p>1. Chọn một hoặc nhiều ảnh JPG, PNG hoặc WEBP.</p>
    <p>2. Điều chỉnh độ tin cậy ở thanh bên.</p>
    <p>3. Nhấn <b>Phân tích tất cả ảnh</b> để xem kết quả và xuất dữ liệu.</p></div>
    """, unsafe_allow_html=True)

st.divider()
st.caption("EcoVision AI · Nhận diện rác thải · Dự đoán có thể chưa chính xác.")
