import tkinter as tk
from tkinter import messagebox, scrolledtext
import cv2
import os
import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Initialize Tkinter
root = tk.Tk()
root.title("Automated Attendance System")
root.geometry("600x400")
root.configure(bg="#E8F0F2")

# Global variables
student_name_var = tk.StringVar()
student_id_var = tk.StringVar()
attendance_file = 'Attendance.csv'
trained_model_file = 'face_trained.yml'
labels_file = 'labels.pkl'
faces_directory = 'faces'
attendance_df = None
email_file = 'students_email.csv'

# Ensure directories and files exist
if not os.path.exists(faces_directory):
    os.makedirs(faces_directory)
if not os.path.exists(attendance_file):
    attendance_df = pd.DataFrame(columns=["ID", "Name", "Timestamp", "Status"])
    attendance_df.to_csv(attendance_file, index=False)
else:
    attendance_df = pd.read_csv(attendance_file)

# Load email data if exists
if not os.path.exists(email_file):
    pd.DataFrame(columns=["ID", "Email"]).to_csv(email_file, index=False)

def show_main_menu():
    for widget in root.winfo_children():
        widget.destroy()
    tk.Label(root, text="Automated Attendance System", font=("Helvetica", 18, "bold"), bg="#E8F0F2").pack(pady=20)
    tk.Button(root, text="Register Student", font=("Helvetica", 14), command=register_student).pack(pady=10)
    tk.Button(root, text="Train Model", font=("Helvetica", 14), command=train_model).pack(pady=10)
    tk.Button(root, text="Mark Attendance", font=("Helvetica", 14), command=mark_attendance).pack(pady=10)
    tk.Button(root, text="View Attendance", font=("Helvetica", 14), command=view_attendance).pack(pady=10)
    tk.Button(root, text="Send Email", font=("Helvetica", 14), command=send_email).pack(pady=10)

def register_student():
    for widget in root.winfo_children():
        widget.destroy()
    tk.Label(root, text="Register Student", font=("Helvetica", 18, "bold"), bg="#E8F0F2").pack(pady=20)
    tk.Label(root, text="Enter Student Name:", font=("Helvetica", 14), bg="#E8F0F2").pack(pady=10)
    tk.Entry(root, textvariable=student_name_var, font=("Helvetica", 14)).pack(pady=5)
    tk.Label(root, text="Enter Student ID:", font=("Helvetica", 14), bg="#E8F0F2").pack(pady=10)
    tk.Entry(root, textvariable=student_id_var, font=("Helvetica", 14)).pack(pady=5)
    tk.Label(root, text="Enter Student Email:", font=("Helvetica", 14), bg="#E8F0F2").pack(pady=10)
    student_email_var = tk.StringVar()
    tk.Entry(root, textvariable=student_email_var, font=("Helvetica", 14)).pack(pady=5)
    tk.Button(root, text="Capture Image", font=("Helvetica", 14), command=lambda: capture_student_image(student_email_var)).pack(pady=10)
    tk.Button(root, text="Back", font=("Helvetica", 14), command=show_main_menu).pack(pady=10)

def capture_student_image(student_email_var):
    student_name = student_name_var.get().strip()
    student_id = student_id_var.get().strip()
    student_email = student_email_var.get().strip()
    if not student_name or not student_id or not student_email:
        messagebox.showerror("Error", "Please enter all fields")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Capture Student Image")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        cv2.imshow("Capture Student Image", frame)
        
        k = cv2.waitKey(1)
        if k % 256 == 32:  # Press SPACE to capture image
            img_name = os.path.join(faces_directory, f"{student_id}.jpg")
            cv2.imwrite(img_name, gray[y:y+h, x:x+w])  # Save the detected face region
            messagebox.showinfo("Success", f"{img_name} written!")

            # Save student email information
            students_df = pd.read_csv(email_file)
            students_df = pd.concat([students_df, pd.DataFrame([{"ID": student_id, "Email": student_email}])], ignore_index=True)
            students_df.to_csv(email_file, index=False)

            break

    cap.release()
    cv2.destroyAllWindows()
    tk.Button(root, text="Back", font=("Helvetica", 14), command=show_main_menu).pack(pady=10)



def train_model():
    try:
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        faces, labels = [], []

        label_map = {}
        image_size = (200, 200)  # Set a consistent size for all images

        for filename in os.listdir(faces_directory):
            if filename.endswith(".jpg"):
                image_path = os.path.join(faces_directory, filename)
                gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                resized_image = cv2.resize(gray_image, image_size)  # Resize the image
                faces.append(resized_image)
                labels.append(int(filename.split('.')[0]))  # Use student ID as label
                label_map[int(filename.split('.')[0])] = filename.split('.')[0]

        faces = np.array(faces)
        labels = np.array(labels)
        face_recognizer.train(faces, labels)
        face_recognizer.write(trained_model_file)
        with open(labels_file, 'wb') as f:
            pickle.dump(label_map, f)

        messagebox.showinfo("Success", "Model trained successfully!")
    except Exception as e:
        messagebox.showerror("Error", str(e))
    show_main_menu()

def mark_attendance():
    global attendance_df  # Ensure attendance_df is accessible in this function
    try:
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.read(trained_model_file)

        with open(labels_file, 'rb') as f:
            label_map = pickle.load(f)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        cap = cv2.VideoCapture(0)
        cv2.namedWindow("Attendance System")

        marked_names = set()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                label, confidence = face_recognizer.predict(face)
                name = label_map[label]

                if name not in marked_names:
                    now = datetime.now()
                    date_str = now.strftime("%Y-%m-%d")
                    existing_entry = attendance_df[(attendance_df["Name"] == name) & (attendance_df["Timestamp"].str.startswith(date_str))]

                    if existing_entry.empty:
                        # Mark entry time
                        time_stamp = now.strftime("%Y-%m-%d %H:%M:%S")
                        new_entry = pd.DataFrame({"ID": [label], "Name": [name], "Timestamp": [time_stamp], "Status": ["Entry"]})
                        attendance_df = pd.concat([attendance_df, new_entry], ignore_index=True)
                        print(f"{name} entry marked at {time_stamp}")
                    else:
                        last_entry = pd.to_datetime(existing_entry["Timestamp"]).max()
                        if now - last_entry >= timedelta(minutes=2):
                            if "Exit" not in existing_entry["Status"].values:
                                # Mark exit time
                                time_stamp = now.strftime("%Y-%m-%d %H:%M:%S")
                                new_entry = pd.DataFrame({"ID": [label], "Name": [name], "Timestamp": [time_stamp], "Status": ["Exit"]})
                                attendance_df = pd.concat([attendance_df, new_entry], ignore_index=True)
                                print(f"{name} exit marked at {time_stamp}")
                            else:
                                print(f"{name} exit already marked for today.")
                        else:
                            print(f"{name} entry already marked for today.")

                    marked_names.add(name)

                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            cv2.imshow("Attendance System", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        attendance_df.to_csv(attendance_file, index=False)
        messagebox.showinfo("Success", "Attendance saved successfully!")
    except Exception as e:
        messagebox.showerror("Error", str(e))
    show_main_menu()

def send_email():
    try:
        students_df = pd.read_csv(email_file)
        attendance_df = pd.read_csv(attendance_file)

        for _, student in students_df.iterrows():
            student_id = student['ID']
            email = student['Email']

            # Check if both Entry and Exit are recorded for the student
            student_attendance = attendance_df[attendance_df["ID"] == student_id]
            
            entry_recorded = not student_attendance[student_attendance["Status"] == "Entry"].empty
            exit_recorded = not student_attendance[student_attendance["Status"] == "Exit"].empty

            if entry_recorded and exit_recorded:
                status = "Your attendance is successfully recorded for the day."
            else:
                status = "your entry has been marked."

            subject = "Your Attendance Status"
            body = f"Dear Student,\n\nHere is your attendance status:\n\n{status}\n\nRegards,\nAutomated Attendance System"

            # Setup the email server
            sender_email = "saivamshi939@gmail.com"  # Replace with your email
            sender_password = "fkaclisiwjawdsnf"  # Replace with your email password

            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = email
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))

            # Send email
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, email, msg.as_string())

            print(f"Email sent to {email}")
            
        messagebox.showinfo("Success", "Emails sent successfully!")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def view_attendance():
    global attendance_df
    for widget in root.winfo_children():
        widget.destroy()
    tk.Label(root, text="View Attendance", font=("Helvetica", 18, "bold"), bg="#E8F0F2").pack(pady=20)
    
    text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=10, font=("Helvetica", 12))
    text_area.pack(pady=10)
    attendance_data = pd.read_csv(attendance_file).to_string(index=False)
    text_area.insert(tk.END, attendance_data)
    
    tk.Button(root, text="Back", font=("Helvetica", 14), command=show_main_menu).pack(pady=10)

# Show main menu
show_main_menu()

# Run Tkinter event loop
root.mainloop()