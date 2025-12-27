import cv2
import glob
import os
import numpy as np

# โฟลเดอร์ที่เก็บวิดีโอ real และ synthesis
REAL_FOLDER = "/Users/kunruethai/Documents/CP2025/Celeb-real1"
SYN_FOLDER  = "/Users/kunruethai/Documents/CP2025/Celeb-synthesis1"

# โฟลเดอร์สำหรับเก็บรูปที่ต่อกันแล้ว
OUTPUT_FOLDER = "/Users/kunruethai/Documents/CP2025/paired_captures"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# นามสกุลไฟล์วิดีโอที่ต้องการอ่าน
video_extensions = ("*.mp4", "*.mov", "*.avi", "*.mkv")

def list_videos(folder):
    files = []
    for ext in video_extensions:
        files.extend(glob.glob(os.path.join(folder, ext)))
    files.sort()   # เรียงชื่อไฟล์ก่อนจับคู่
    return files

# ดึงรายชื่อวิดีโอ
real_videos = list_videos(REAL_FOLDER)
syn_videos  = list_videos(SYN_FOLDER)

print(f"real videos : {len(real_videos)} ไฟล์")
print(f"syn  videos : {len(syn_videos)} ไฟล์")

# ใช้จำนวนต่ำสุดระหว่าง real, syn และ 41 ไฟล์แรก (ถ้าอยากเปลี่ยนเป็น 20 ก็แก้ 41 -> 20)
pair_count = min(100, len(real_videos), len(syn_videos))
print(f"จะประมวลผล {pair_count} คู่แรก")

TARGET_W, TARGET_H = 810, 466  # ขนาดของรูปแต่ละด้าน

for i in range(pair_count):
    real_path = real_videos[i]
    syn_path  = syn_videos[i]

    print(f"\nคู่ที่ {i+1}:")
    print(f"  real: {real_path}")
    print(f"  syn : {syn_path}")

    # ----- อ่านเฟรมจาก real -----
    cap_real = cv2.VideoCapture(real_path)
    cap_real.set(cv2.CAP_PROP_POS_MSEC, 1000)  # ไปที่วินาทีที่ 1
    ok_r, frame_real = cap_real.read()
    cap_real.release()

    if not ok_r or frame_real is None:
        print("  ❌ อ่านเฟรมจาก real ไม่ได้ ข้ามคู่นี้")
        continue

    # ----- อ่านเฟรมจาก syn -----
    cap_syn = cv2.VideoCapture(syn_path)
    cap_syn.set(cv2.CAP_PROP_POS_MSEC, 1000)   # ไปที่วินาทีที่ 1
    ok_s, frame_syn = cap_syn.read()
    cap_syn.release()

    if not ok_s or frame_syn is None:
        print("  ❌ อ่านเฟรมจาก syn ไม่ได้ ข้ามคู่นี้")
        continue

    # ----- resize ทั้งสองภาพเป็น 810x466 -----
    frame_real_resized = cv2.resize(frame_real, (TARGET_W, TARGET_H))
    frame_syn_resized  = cv2.resize(frame_syn,  (TARGET_W, TARGET_H))

    # ----- ต่อกันแนวนอน: [ real | syn ] -----
    # real อยู่ซ้าย, syn อยู่ขวา
    combined = np.hstack((frame_real_resized, frame_syn_resized))  # 1620x466

    # ----- ตั้งชื่อไฟล์เป็น df11, df12, df13, ... -----
    index_num = 1 + i            # i = 0 -> df11, i = 1 -> df12, ...
    out_name  = f"df{index_num}.jpg"
    out_path  = os.path.join(OUTPUT_FOLDER, out_name)

    cv2.imwrite(out_path, combined)
    print(f"  ✅ บันทึกรูป: {out_path}")

print("\nเสร็จหมดแล้ว ✅")
