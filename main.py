#import sys
#sys.path.append(r'C:\Users\morteza\PycharmProjects\Shirt-try-on\.venv\Lib\site-packages')
import cv2
import mediapipe as mp
import numpy as np

# تنظیمات Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# بارگذاری تصاویر لباس با کانال آلفا
dress1 = cv2.imread('dress1.png', cv2.IMREAD_UNCHANGED)
dress2 = cv2.imread('dress2.png', cv2.IMREAD_UNCHANGED)

# بررسی موفقیت بارگذاری تصاویر
if dress1 is None or dress2 is None:
    raise FileNotFoundError("تصاویر لباس پیدا نشدند. لطفاً مسیر فایل‌ها را بررسی کنید.")

# اطمینان از داشتن کانال آلفا
def ensure_alpha(img: np.ndarray) -> np.ndarray:
    if img.shape[2] == 3:
        alpha_channel = np.full(img.shape[:2], 255, dtype=img.dtype)
        return cv2.merge((*cv2.split(img), alpha_channel))
    return img

dress1 = ensure_alpha(dress1)
dress2 = ensure_alpha(dress2)

# افزودن لباس با آلفا بلندینگ
def add_dress_to_frame(frame: np.ndarray, dress: np.ndarray, x: int, y: int) -> None:
    h, w = dress.shape[:2]
    if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
        return  # خارج از محدوده تصویر

    overlay = dress[:, :, :3]
    alpha = dress[:, :, 3] / 255.0
    roi = frame[y:y+h, x:x+w]

    blended = (alpha[..., None] * overlay + (1 - alpha[..., None]) * roi).astype(np.uint8)
    frame[y:y+h, x:x+w] = blended

# دسترسی به وب‌کم
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # برای ویندوز؛ در لینوکس یا مک می‌توان CAP_DSHOW را حذف کرد

with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        results = pose.process(image_rgb)

        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            try:
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

                # موقعیت و اندازه لباس
                center_x = int((left_shoulder.x + right_shoulder.x) / 2 * frame.shape[1])
                shoulders_y = int((left_shoulder.y + right_shoulder.y) / 2 * frame.shape[0])
                hips_y = int((left_hip.y + right_hip.y) / 2 * frame.shape[0])

                dress_width = int(abs(right_shoulder.x - left_shoulder.x) * frame.shape[1] * 1.5)
                dress_height = int((hips_y - shoulders_y) * 2)

                if dress_width > 0 and dress_height > 0:
                    resized_dress1 = cv2.resize(dress1, (dress_width, dress_height), interpolation=cv2.INTER_AREA)
                    resized_dress2 = cv2.resize(dress2, (dress_width, dress_height), interpolation=cv2.INTER_AREA)

                    dress1_x = center_x - dress_width // 2
                    dress1_y = shoulders_y - dress_height // 3

                    dress2_x = dress1_x
                    dress2_y = dress1_y + dress_height + 10

                    add_dress_to_frame(image_bgr, resized_dress1, dress1_x, dress1_y)
                    add_dress_to_frame(image_bgr, resized_dress2, dress2_x, dress2_y)
            except IndexError:
                pass  # در صورتی که برخی از نقاط در فریم تشخیص داده نشوند

        cv2.imshow('Virtual Dressing Room', image_bgr)
        if cv2.waitKey(1) & 0xFF == 27:  # کلید ESC
            break

cap.release()
cv2.destroyAllWindows()
