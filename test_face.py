import cv2
import face_recognition
import numpy as np

print("--- BẮT ĐẦU TEST ---")

# 1. Kiểm tra phiên bản
print(f"Numpy version: {np.__version__}")
try:
    import dlib
    print(f"Dlib version: {dlib.__version__}")
except:
    print("Dlib chưa cài hoặc lỗi import")

# 2. Thử tạo một ảnh giả (Dummy Image) - Màu đen 100x100
# Nếu cái này mà lỗi -> Thư viện hỏng 100%
print("\n[TEST 1] Thử với ảnh giả lập (Black Image)...")
dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
try:
    encodings = face_recognition.face_encodings(dummy_image)
    print("✅ TEST 1 OK: Thư viện hoạt động bình thường với ảnh giả.")
except Exception as e:
    print(f"❌ TEST 1 FAILED: {e}")
    print("=> KẾT LUẬN: Thư viện dlib/numpy bị xung đột. Cần cài lại.")

# 3. Thử load file thật
print("\n[TEST 2] Thử load file assets/master.jpg...")
img_path = "assets/master.jpg"
img = cv2.imread(img_path)

if img is None:
    print("❌ LỖI: Không tìm thấy file ảnh! Bạn xem lại tên file/thư mục.")
else:
    print(f"Đã đọc được ảnh. Shape: {img.shape}, Dtype: {img.dtype}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        encodings = face_recognition.face_encodings(rgb)
        print("✅ TEST 2 OK: Ảnh chủ nhân hợp lệ!")
    except Exception as e:
        print(f"❌ TEST 2 FAILED: {e}")