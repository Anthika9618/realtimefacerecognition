import os
import csv
import cv2
import glob
import torch
import joblib
import numpy as np
import pandas as pd
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import ttk, messagebox
from sklearn.svm import SVC
from facenet_pytorch import InceptionResnetV1, MTCNN
import threading
import queue

# ========================================== Data Employee =================================================
employee_data = pd.read_csv('/mnt/c/projectaunaun/realtimefacerecognition/Employeedataset_df.csv')
employee_dict = {row['name']: row for _, row in employee_data.iterrows()}

# ======================================== Load SVM Model ================================================
clf = joblib.load('/mnt/c/projectaunaun/realtimefacerecognition/models/svm_face_model.pkl')

# ======================================== Load Gender Model ==============================================
gender_net = cv2.dnn.readNetFromCaffe(
    '/mnt/c/projectaunaun/realtimefacerecognition/models/gender_deploy.prototxt',
    '/mnt/c/projectaunaun/realtimefacerecognition/models/gender_net.caffemodel'
)
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
gender_list = ('Male', 'Female')

# ============================================= Device =====================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(keep_all=False, device=device)

# ========================================== Face Functions ================================================
def get_face_embedding(img):
    img = Image.fromarray(img).convert('RGB').resize((160, 160))
    img_tensor = torch.tensor(np.array(img) / 255., dtype=torch.float32).permute(2,0,1).unsqueeze(0)
    img_tensor = (img_tensor - 0.5)/0.5
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        embedding = resnet(img_tensor)
    return embedding.squeeze().cpu().numpy()

def recognize_face_in_frame(frame):
    scale_factor = 0.4
    small_frame = cv2.resize(frame, (0,0), fx=scale_factor, fy=scale_factor)
    img_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    boxes, probs = mtcnn.detect(img_rgb)
    detected_names = []
    if boxes is not None and probs is not None:
        for box, prob in zip(boxes, probs):
            if prob < 0.98:
                continue
            x1, y1, x2, y2 = [int(coord/scale_factor) for coord in box]
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue
            embedding = get_face_embedding(face)
            probs_svm = clf.predict_proba([embedding])[0]
            max_prob = np.max(probs_svm)
            name = clf.classes_[np.argmax(probs_svm)]
            if max_prob < 0.80:
                name = "Unknown"
            detected_names.append(name)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"{name} ({max_prob:.2f})", (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    return frame, detected_names[0] if detected_names else "Unknown"

def predict_gender(face_bgr):
    blob = cv2.dnn.blobFromImage(face_bgr, scalefactor=1.0, size=(227,227),
                                 mean=MODEL_MEAN_VALUES, swapRB=False)
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = gender_list[gender_preds[0].argmax()]
    return gender

# ========================================== Admin Login ====================================================
def login_window():
    login_success = False
    root = tk.Tk()
    root.title("Admin Login")
    root.geometry("500x350")
    root.configure(bg="#f5f7fa")
    style = ttk.Style()
    style.configure("TLabel", font=("Segoe UI", 16))
    style.configure("TEntry", font=("Segoe UI", 16))
    style.configure("TButton", font=("Segoe UI", 16, "bold"), padding=6)
    frame = tk.Frame(root, bg="white", bd=2, relief="ridge", padx=20, pady=20)
    frame.place(relx=0.5, rely=0.5, anchor="center")

    logo_image = Image.open("/mnt/c/projectaunaun/realtimefacerecognition/img/logo.png").resize((50,50), Image.Resampling.LANCZOS)
    logo_photo = ImageTk.PhotoImage(logo_image)
    root.logo_photo = logo_photo

    title_frame = tk.Frame(frame, bg="white")
    title_frame.grid(row=0, column=0, columnspan=2, pady=(0,20))
    logo_label = tk.Label(title_frame, image=root.logo_photo, bg="white")
    logo_label.grid(row=0, column=0, padx=(0,10))
    tk.Label(title_frame, text="FORTH RECOGNITION SYSTEM", font=("Segoe UI",16,"bold"), bg="white", fg="#2c3e50").grid(row=0,column=1)

    tk.Label(frame, text="Username:", bg="white", font=("Segoe UI",14)).grid(row=1,column=0, sticky="e", padx=5, pady=5)
    entry_username = tk.Entry(frame, font=("Segoe UI",14), width=25)
    entry_username.grid(row=1,column=1, padx=5,pady=5)

    tk.Label(frame, text="Password:", bg="white", font=("Segoe UI",14)).grid(row=2,column=0, sticky="e", padx=5, pady=5)
    entry_password = tk.Entry(frame, font=("Segoe UI",14), show="*", width=25)
    entry_password.grid(row=2,column=1, padx=5,pady=5)

    def attempt_login():
        nonlocal login_success
        username = entry_username.get().strip()
        password = entry_password.get().strip()
        if username=="admin" and password=="admin0263":
            login_success=True
            root.destroy()
        else:
            messagebox.showerror("Login Failed","Incorrect username or password.")
            entry_password.delete(0,tk.END)

    tk.Button(frame, text="Login", command=attempt_login,
              font=("Segoe UI",14,"bold"), bg="#4CAF50", fg="white", width=15).grid(row=3,column=0,columnspan=2,pady=20)

    root.mainloop()
    return login_success

# ========================================== Retrain SVM Model =============================================
def retrain_svm_model():
    dataset_path = "./dataset"
    embeddings = []
    labels = []
    for person_folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, person_folder)
        if not os.path.isdir(folder_path):
            continue
        for img_path in glob.glob(os.path.join(folder_path,"*jpg")):
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            boxes,_ = mtcnn.detect(img_rgb)
            if boxes is not None:
                for box in boxes:
                    x1,y1,x2,y2 = [int(coord) for coord in box]
                    face = img_rgb[y1:y2, x1:x2]
                    if face.size==0:
                        continue
                    embedding = get_face_embedding(face)
                    embeddings.append(embedding)
                    labels.append(person_folder.split("_",1)[1])
    if embeddings:
        clf_new = SVC(probability=True)
        clf_new.fit(embeddings, labels)
        joblib.dump(clf_new,"/mnt/c/projectaunaun/realtimefacerecognition/models/svm_face_model.pkl")
        print("[INFO] SVM retrained and saved.")
    else:
        print("[WARNING] No embeddings found. Model not retrain.")

# ============================================= FaceRegisterApp ============================================
class FaceRegisterApp:
    def __init__(self, window, video_source=0):
        self.window = window
        self.window.title("Face Recognition & Register")
        self.video_source = video_source

        # Open camera
        self.vid = cv2.VideoCapture(self.video_source)
        self.frame_queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()

        # GUI
        self.frame = tk.Frame(window)
        self.frame.pack()
        self.canvas = tk.Canvas(self.frame, width=640, height=360)
        self.canvas.grid(row=0, column=0)
        self.btn_register = tk.Button(self.frame, text="Register Unknown", command=self.register_unknown)
        self.btn_register.grid(row=1, column=0, pady=10)

        # Variables
        self.current_frame = None
        self.current_name = "Unknown"
        self.delay = 20
        self.frame_count = 0

        # Start threads
        self.capture_thread = threading.Thread(target=self.update_camera, daemon=True)
        self.capture_thread.start()
        self.recognition_thread = threading.Thread(target=self.recognize_face_loop, daemon=True)
        self.recognition_thread.start()

        self.window.after(self.delay, self.update_gui)
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.recognized_name = "Unknown"
        self.recognized_frame = None

    def update_camera(self):
        while not self.stop_event.is_set():
            ret, frame = self.vid.read()
            if ret and not self.frame_queue.full():
                self.frame_queue.put(frame)

    def recognize_face_loop(self):
        while not self.stop_event.is_set():
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                self.frame_count += 1
                if self.frame_count % 2 == 0:
                    frame, name = recognize_face_in_frame(frame)
                    self.recognized_frame = frame
                    self.recognized_name = name

    def update_gui(self):
        if self.recognized_frame is not None:
            frame_resized = cv2.resize(self.recognized_frame, (640, 360))
            cv2image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.imgtk = imgtk
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.window.after(self.delay, self.update_gui)

    def register_unknown(self):
        # ใส่ฟังก์ชัน Register ตามเดิม
        messagebox.showinfo("Info", "Register Unknown clicked. Implement as needed.")

    def on_closing(self):
        self.stop_event.set()
        if self.vid.isOpened():
            self.vid.release()
        self.window.destroy()

# ============================================ Main =========================================================
if __name__ == "__main__":
    if not login_window():
        print("Login failed. Exiting program.")
        exit()
    root = tk.Tk()
    app = FaceRegisterApp(root, video_source="rtsp://admin:Forth@1234@192.168.1.218:554/cam/realmonitor?channel=1&subtype=1&rtsp_transport=tcp")
    root.mainloop()
