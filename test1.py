import os
import cv2
import csv
import torch
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
from facenet_pytorch import InceptionResnetV1, MTCNN
import time

# โหลดข้อมูลพนักงาน
employee_data = pd.read_csv('/mnt/c/projectaunaun/realtimefacerecognition/Employeedataset_df.csv')
employee_dict = {row['name']: row for _, row in employee_data.iterrows()}

# โหลดโมเดล SVM
clf = joblib.load('/mnt/c/projectaunaun/realtimefacerecognition/models/svm_face_model.pkl')

# กำหนด device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(keep_all=False, device=device)

def get_face_embedding(img):
    img = Image.fromarray(img).convert('RGB').resize((160, 160))
    img_tensor = torch.tensor(np.array(img) / 255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    img_tensor = (img_tensor - 0.5) / 0.5
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        embedding = resnet(img_tensor)
    return embedding.squeeze().cpu().numpy()

def recognize_face_in_frame(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(img_rgb)
    detected_names = []

    person_count = 0
    if boxes is not None:
        person_count = len(boxes)
        for box in boxes:
            x1, y1, x2, y2 = [int(coord) for coord in box]
            face = img_rgb[y1:y2, x1:x2]
            if face.size == 0:
                continue

            embedding = get_face_embedding(face)
            probs = clf.predict_proba([embedding])[0]
            max_prob = np.max(probs)
            name = clf.classes_[np.argmax(probs)]

            if max_prob < 0.70:
                name = "Unknown"

            detected_names.append(name)

            emp_id = position = "N/A"
            if name in employee_dict:
                emp = employee_dict[name]
                emp_id = emp['employee_id']
                position = emp['position']

            timestamp = datetime.now().strftime("%H:%M:%S")
            label = f"{name} ({max_prob:.2f})"
            info = f"ID: {emp_id} | {position} | {timestamp}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, info, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, f"Number of people in the frame: {person_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return frame, detected_names[0] if detected_names else "Unknown"


# ตั้งค่า video capture
cap1 = cv2.VideoCapture("rtsp://admin:Forth%401234@192.168.1.219:554/cam/realmonitor?channel=1&subtype=1&rtsp_transport=tcp")

cv2.namedWindow('Camera 1', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camera 1', 960, 540)  # ขยายหน้าต่างใหญ่ขึ้น

# ตัวแปรบันทึก attendance
save_state = None
saved_names_today = set()
countdown_start = None
save_done_time = None
show_saved_text_until = None

file_path = '/mnt/c/projectaunaun/realtimefacerecognition/attendance_log.csv'
file_exists = os.path.isfile(file_path)

def save_attendance(name):
    global save_state, countdown_start, save_done_time, show_saved_text_until, file_exists

    if name == "Unknown":
        return None, []

    now = datetime.now()
    current_date = now.date()
    records_today = [entry for entry in saved_names_today if entry[0] == name and entry[1] == current_date]

    if save_state != name:
        save_state = name
        countdown_start = time.time()
        save_done_time = None
        show_saved_text_until = None

    elapsed = time.time() - countdown_start
    return elapsed, records_today

# Main loop
while True:
    ret1, frame1 = cap1.read()
    
    if not ret1:
        continue  # รอกล้องต่อไป

    frame1, name1 = recognize_face_in_frame(frame1)
    elapsed, records_today = save_attendance(name1) or (None, [])

    if elapsed is not None:
        if elapsed < 3:
            remaining = 3 - int(elapsed)
            text = f"Counting: {remaining}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
            text_x = int((frame1.shape[1] - text_size[0]) / 2)
            text_y = int(frame1.shape[0] * 0.12)
            cv2.putText(frame1, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        elif elapsed >= 3 and elapsed < 5:
            text = "Save Successfully"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
            text_x = int((frame1.shape[1] - text_size[0]) / 2)
            text_y = int(frame1.shape[0] * 0.15)
            cv2.putText(frame1, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        elif elapsed >= 5 and save_done_time is None:
            emp_id = position = "N/A"
            if name1 in employee_dict:
                emp = employee_dict[name1]
                emp_id = emp['employee_id']
                position = emp['position']

            with open(file_path, "a", newline='', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Date", "Time", "Name", "Employee_ID", "Position", "Status", "Camera"])
                    file_exists = True

                status = "Check-in" if not records_today else "Check-out"
                writer.writerow([datetime.now().date(), datetime.now().strftime("%H:%M:%S"), name1, emp_id, position, status, 1])

            saved_names_today.add((name1, datetime.now().date(), status.lower()))
            save_done_time = time.time()
            show_saved_text_until = save_done_time + 3

    cv2.imshow('Camera 1', frame1)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

cap1.release()
cv2.destroyAllWindows()
