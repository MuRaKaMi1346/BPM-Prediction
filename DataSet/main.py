from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import librosa
import numpy as np
import os
import glob

# Path ของโฟลเดอร์ที่เก็บไฟล์เสียง
folder_path = "C:\\RSU\\CPE270_Project(AI)\\DataSet\\DataSample\\audio_track"
output_path = "C:\\RSU\\CPE270_Project(AI)\\DataSet\\DataSample\\BPM_Data.xlsx"

# สร้าง DataFrame ว่างสำหรับเก็บข้อมูลทั้งหมด
all_data = []

# หาไฟล์เสียงทั้งหมดในโฟลเดอร์ (.mp3 และ .wav)
audio_files = glob.glob(os.path.join(folder_path, "*.mp3")) + glob.glob(os.path.join(folder_path, "*.wav"))

i = 1
for audio_path in audio_files:
    try:
        print(f"[{i}]กำลังโหลดไฟล์: {os.path.basename(audio_path)}")  # แสดงชื่อไฟล์ที่กำลังโหลด
        i+=1
        y, sr = librosa.load(audio_path)
        duration = librosa.get_duration(y=y, sr=sr)

        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        num_onsets = len(onset_times)

        intervals = np.diff(onset_times)
        onsets_per_sec = num_onsets / duration if duration > 0 else 0
        avg_interval = np.mean(intervals) if len(intervals) > 0 else 0

        y_percussive = librosa.effects.hpss(y)
        percussive_mean = np.mean(np.abs(y_percussive))

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = int(round(tempo.item()))

        all_data.append({
            "filename": os.path.basename(audio_path),
            "duration_sec": duration,
            "onsets_per_sec": onsets_per_sec,
            "avg_interval": avg_interval,
            "percussive_mean": percussive_mean,
            "BPM": tempo
        })

    except Exception as e:
        print(f"เกิดข้อผิดพลาดกับไฟล์ {audio_path}: {e}")

# สร้าง DataFrame จากข้อมูลทั้งหมด
new_data = pd.DataFrame(all_data)

# โหลดข้อมูลเก่า (ถ้ามี) แล้วรวมข้อมูล
if os.path.exists(output_path):
    old_data = pd.read_excel(output_path, engine="openpyxl")
    combined_data = pd.concat([old_data, new_data], ignore_index=True)
else:
    combined_data = new_data

# บันทึกกลับไปที่ไฟล์ Excel
combined_data.to_excel(output_path, index=False, engine="openpyxl")

print("สำเร็จ")
######################################################################################
                           #AI ANALYSIS#
######################################################################################
data_last = pd.read_excel(output_path, engine="openpyxl")
if len(data_last) < 10:
    print("ข้อมูลยังมีไม่พอสำหรับการเทรน AI ")
    exit()

data_last = data_last.dropna()
X = data_last[["duration_sec", "onsets_per_sec", "avg_interval", "percussive_mean"]]
y = data_last["BPM"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# เทรนโมเดลแบบ Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ทำนาย BPM
y_pred = model.predict(X_test)

# ประเมินผล
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"🎯 Mean Absolute Error (MAE): {mae:.2f}")
print(f"📈 R-squared (R²): {r2:.2f}")

print("ใส่ข้อมูลเพลงเพื่อทำนาย BPM:")
try:
    A = float(input("ระยะเวลาเพลง (วินาที): "))
    B = float(input("จำนวนจังหวะต่อวินาที: "))
    C = float(input("ช่วงเวลาระหว่างจังหวะเฉลี่ย (วินาที): "))
    D = float(input("ความแรงของเสียง Percussive : "))
except ValueError:
    print("❌ กรอกข้อมูลเป็นตัวเลขเท่านั้น")
    exit()

new_song = [[A, B, C, D]]  # [duration_sec, onsets_per_sec, avg_interval, percussive_mean]
predicted_bpm = model.predict(new_song)
print(f"🎵 AI ประเมินว่าเพลงนี้น่าจะมี BPM ≈ {int(predicted_bpm[0])}")
