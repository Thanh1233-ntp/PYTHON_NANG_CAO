- Mình có phân loại mức độ rác không? Ví dụ không có rác, ít rác, nhiều rác? Bà muốn có nhiêu ảnh? Những ảnh ngoài 3 loại kia ra thì có tải về không? Ảnh đơn điệu chỉ chứa một thứ hay ảnh chứa nhiều thứ khác nhau

- Chỉ phân loại 3 loại rác là platic paper metal thôi đúng không? Có cần thêm không, ví dụ như glass cardboard? đề xuất thêm một mục là trash nếu ảnh đó không liên quan gì đến mấy loại trên.

- Nếu 1 ảnh có từ 2 loại rác trở lên thì ảnh đó để vào mục nào? Hay cả 2 mục đều có ảnh đó? Hoặc nó sẽ nằm trong mục trash

- Trong mục data có cần tạo thêm mấy mục con như 'plastic', 'paper', 'metal'... không?

- hello word

- Hi, pls confirm help me

## Trả lời
**1. Về mức độ rác và ảnh thử nghiệm**

- Trước mắt, mình muốn chuẩn bị khoảng **30–50 ảnh** để thử nghiệm.
- Ảnh cần đa dạng: ảnh có ít rác, nhiều rác, ảnh chỉ có một loại rác và ảnh có nhiều loại rác cùng lúc.
- Có thể thêm một số ảnh không chứa 4 loại rác mục tiêu để kiểm tra mô hình có nhận diện nhầm hay không.
- Có thể thống kê mức độ rác dựa trên số vật thể được phát hiện: không phát hiện, ít hoặc nhiều. 

**2. Về các loại rác cần nhận diện**

Mình thống nhất dự án sẽ nhận diện 4 loại:
- Plastic: Nhựa
- Paper: Giấy
- Metal: Kim loại
- Glass: Thủy tinh

Trước mắt chưa thêm Cardboard hoặc Trash để tránh làm phạm vi dự án quá lớn.

Lưu ý: cần kiểm tra mô hình YOLO hiện tại có hỗ trợ đủ 4 loại không. 

**3. Nếu một ảnh có nhiều loại rác thì xếp vào đâu?**

Ảnh đó chỉ cần lưu **một lần**, không cần chia vào nhiều thư mục.

Ví dụ, một ảnh có chai nhựa, giấy và lon kim loại thì mô hình trả về cả 3 loại trong cùng kết quả. Không xếp ảnh đó vào mục Trash.

Nếu ảnh không chứa 4 loại rác mục tiêu, có thể ghi nhận là “Không phát hiện loại rác mục tiêu”. Chưa cần tạo lớp Trash riêng.

**4. Cách tổ chức thư mục ảnh**

Không cần chia thư mục theo từng loại rác vì một ảnh có thể chứa nhiều loại cùng lúc.

Để tất cả ảnh thử nghiệm chung trong thư mục:

`data/test_images/`

Ví dụ:

- `image_001.jpg`
- `image_002.jpg`
- `image_003.jpg`

Dễ quản lý ảnh và kiểm tra khả năng nhận diện nhiều loại rác trong cùng một ảnh.
