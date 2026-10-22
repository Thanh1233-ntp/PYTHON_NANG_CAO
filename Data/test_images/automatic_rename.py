# File Python này sẽ tự động đổi tên các file ảnh trong thư mục hiện tại theo định dạng "image_0001.jpg", "image_0002.png", v.v.

import os

# Thư mục chứa file ảnh (để dấu chấm nếu ở cùng thư mục với script)
folder_path = "."

# Lấy danh sách file và sắp xếp theo thời gian hoặc tên cũ
files = sorted(os.listdir(folder_path))

# Lọc chỉ lấy các file ảnh (ví dụ: .jpg, .png)
valid_extensions = (".jpg", ".jpeg", ".png", ".webp")
image_files = [f for f in files if f.lower().endswith(valid_extensions)]

for index, filename in enumerate(image_files, start=1):
    # Lấy phần mở rộng của file cũ (.jpg, .png...)
    ext = os.path.splitext(filename)[1]
    
    # Tạo tên mới với định dạng 4 chữ số (0001, 0002...)
    new_name = f"image_{index:04d}{ext}"
    # Nếu có thêm ảnh thì chỉ cần đổi thành index:05d sau đó chạy, rồi đổi lại thành 04d chạy là được.
    
    # Đường dẫn cũ và mới
    old_path = os.path.join(folder_path, filename)
    new_path = os.path.join(folder_path, new_name)
    
    # Đổi tên
    os.rename(old_path, new_path)

print(f"Đã đổi tên thành công {len(image_files)} file!")