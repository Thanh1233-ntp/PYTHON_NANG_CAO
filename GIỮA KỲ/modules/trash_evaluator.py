"""
TrashEvaluator: đánh giá kết quả của TrashDetector.

Đầu vào: danh sách detections, mỗi phần tử có ít nhất
    {"class_name": "Nhựa", "confidence": 0.84}   (các khóa khác như bbox, vat_the được bỏ qua)

Đầu ra của evaluate(): dict gồm
    tong_so, theo_nhom, nhom_chinh, do_tin_cay_tb, muc_tin_cay,
    so_vat_can_kiem_tra, nhan_xet, huong_dan

Dùng được độc lập (không cần Streamlit hay Roboflow).
"""
from collections import defaultdict
from typing import Any, Dict, List, Optional


class TrashEvaluator:
    # Mức độ tin cậy
    NGUONG_CAO = 0.70
    NGUONG_TRUNG_BINH = 0.50

    # Gợi ý chung, mang tính tham khảo (quy định phân loại rác khác nhau theo từng địa phương)
    HUONG_DAN = {
        "Nhựa": (
            "Kiểm tra ký hiệu vật liệu, làm sạch nếu cần và phân loại "
            "theo quy định thu gom tại địa phương."
        ),
        "Thủy tinh": (
            "Thu gom riêng. Nếu bị vỡ, cần đóng gói an toàn để tránh "
            "gây thương tích."
        ),
        "Kim loại": (
            "Phân loại riêng, kiểm tra vật sắc nhọn và làm sạch nếu cần "
            "trước khi đưa đi thu gom."
        ),
        "Giấy": (
            "Giữ giấy khô, phân loại riêng. Giấy dính nhiều dầu mỡ "
            "hoặc bị nhiễm bẩn có thể không phù hợp để tái chế."
        ),
        "Khác": (
            "Chưa có hướng dẫn phân loại cụ thể. Cần kiểm tra vật thể "
            "trước khi quyết định cách xử lý."
        )
    }

    def __init__(
        self,
        nguong_cao: float = NGUONG_CAO,
        nguong_trung_binh: float = NGUONG_TRUNG_BINH,
    ) -> None:
        if not 0.0 <= nguong_trung_binh <= nguong_cao <= 1.0:
            raise ValueError("Cần 0 <= nguong_trung_binh <= nguong_cao <= 1.")
        self.nguong_cao = nguong_cao
        self.nguong_trung_binh = nguong_trung_binh

    # ------------------------------------------------------------------
    def muc_tin_cay(self, diem: float) -> str:
        if diem >= self.nguong_cao:
            return "Cao"
        if diem >= self.nguong_trung_binh:
            return "Trung bình"
        return "Thấp"

    def evaluate(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        tong = len(detections)
        if tong == 0:
            return {
                "tong_so": 0,
                "theo_nhom": {},
                "nhom_chinh": None,
                "do_tin_cay_tb": 0.0,
                "muc_tin_cay": "Không có dữ liệu",
                "so_vat_can_kiem_tra": 0,
                "nhan_xet": [
                    "Không phát hiện vật thể nào. Hãy thử ảnh rõ hơn hoặc giảm ngưỡng độ tin cậy."
                ],
                "huong_dan": {},
            }

        diem_theo_nhom = defaultdict(list)
        for d in detections:
            diem_theo_nhom[d["class_name"]].append(float(d["confidence"]))

        theo_nhom = {}
        for nhom, ds in diem_theo_nhom.items():
            theo_nhom[nhom] = {
                "so_luong": len(ds),
                "ty_le": round(len(ds) / tong, 4),
                "do_tin_cay_tb": round(sum(ds) / len(ds), 4),
                "do_tin_cay_thap_nhat": round(min(ds), 4),
            }

        # Nhóm chính: nhiều vật nhất; bằng nhau thì lấy nhóm có độ tin cậy trung bình cao hơn
        nhom_chinh = max(
            theo_nhom,
            key=lambda n: (theo_nhom[n]["so_luong"], theo_nhom[n]["do_tin_cay_tb"]),
        )

        tat_ca = [float(d["confidence"]) for d in detections]
        tb = sum(tat_ca) / tong
        can_kiem_tra = sum(1 for x in tat_ca if x < self.nguong_trung_binh)

        return {
            "tong_so": tong,
            "theo_nhom": theo_nhom,
            "nhom_chinh": nhom_chinh,
            "do_tin_cay_tb": round(tb, 4),
            "muc_tin_cay": self.muc_tin_cay(tb),
            "so_vat_can_kiem_tra": can_kiem_tra,
            "nhan_xet": self._nhan_xet(tong, theo_nhom, nhom_chinh, tb, can_kiem_tra),
            "huong_dan": {n: self.HUONG_DAN[n] for n in theo_nhom if n in self.HUONG_DAN},
        }

    def _nhan_xet(self, tong, theo_nhom, nhom_chinh, tb, can_kiem_tra) -> List[str]:
        ds = [f"Phát hiện {tong} vật thể thuộc {len(theo_nhom)} nhóm."]
        if len(theo_nhom) == 1:
            ds.append(f"Tất cả đều thuộc nhóm {nhom_chinh}.")
        else:
            chi_tiet = ", ".join(
                f"{n}: {v['so_luong']}"
                for n, v in sorted(theo_nhom.items(), key=lambda kv: -kv[1]["so_luong"])
            )
            ds.append(
                f"Nhóm xuất hiện nhiều nhất là {nhom_chinh} "
                f"({theo_nhom[nhom_chinh]['so_luong']}/{tong}). Chi tiết: {chi_tiet}."
            )
        muc = self.muc_tin_cay(tb)
        ds.append(f"Độ tin cậy trung bình {tb:.0%} (mức {muc.lower()}).")
        if can_kiem_tra:
            ds.append(
                f"Có {can_kiem_tra} vật có độ tin cậy dưới {self.nguong_trung_binh:.0%}, "
                "nên kiểm tra lại bằng mắt."
            )
        if "Khác" in theo_nhom:
            ds.append("Có vật chưa xếp được vào 4 nhóm chính (nhóm Khác).")
        return ds

    # ------------------------------------------------------------------
    @staticmethod
    def so_voi_nhan_chuan(
        detections: List[Dict[str, Any]], nhan_dung: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """So kết quả với nhãn đúng do người dùng cung cấp (dùng khi đo độ chính xác).
        Trả về None nếu không có nhãn chuẩn."""
        if not nhan_dung:
            return None
        cac_nhom = [d["class_name"] for d in detections]
        return {
            "nhan_dung": nhan_dung,
            "co_phat_hien_dung": nhan_dung in cac_nhom,
            "chi_phat_hien_dung": bool(cac_nhom) and all(n == nhan_dung for n in cac_nhom),
        }
