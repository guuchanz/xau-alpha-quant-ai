Setup
Python 3.11.9
---
# 📈 XAU Alpha Quant AI Pipeline

ระบบบอทเทรดและวิเคราะห์ราคาทองคำ (XAU/USD) ด้วย Machine Learning ที่ออกแบบมาสำหรับใช้งานบนระบบ Production (Windows Server) เน้นความเสถียร รองรับสถาปัตยกรรมแบบ Separation of Concerns และมี Data Pipeline ที่แม่นยำ

---

## 🏗️ ตอนที่ 1: Project Architecture + Environment + Configuration

เป้าหมายของเฟสนี้คือการ **"กันระบบพังตอนตี 3"** เมื่อนำไปรันจริงบน Production เซิร์ฟเวอร์ โดยแยกโครงสร้างพารามิเตอร์และการจัดการระบบออกจากโค้ด Logic 100%

### 📁 โครงสร้างโปรเจกต์ (Folder Architecture)
```text
alpha_quant_ai/
│
├── configs/
│   └── config.yaml          # ศูนย์บัญชาการพารามิเตอร์ทั้งหมด
│
├── data/
│   ├── raw/                 # เก็บไฟล์ CSV ดิบจากแหล่งข้อมูล (ห้ามแก้ไข)
│   └── processed/           # เก็บไฟล์ Parquet หลังจากทำ Feature Engineering
│
├── logs/                    # เก็บไฟล์ Log รายวัน (ระบบ Auto-rotate ป้องกันดิสก์เต็ม)
│
├── models/                  # เก็บไฟล์ .pkl หรือ .joblib ของ LightGBM / XGBoost
│
├── src/
│   ├── __init__.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config_loader.py # ตัวแปลงไฟล์ YAML เป็น Python Dictionary
│   │   └── logger.py        # ตัวจัดการ Log ด้วย Loguru 
│   │
│   ├── data_pipeline/       # (สำหรับตอนที่ 2 และ 3)
│   ├── ml_engine/           # (สำหรับตอนที่ 4 และ 5)
│   ├── backtester/          # (สำหรับตอนที่ 6)
│   └── live_trader/         # (สำหรับตอนที่ 7 และ 8)
│
├── .gitignore
├── requirements.txt         # ไฟล์ล็อกเวอร์ชันของ Library (ต้านทาน Dependency Rot)
└── main.py                  # Entry point จุดเริ่มต้นการทำงานของระบบ
```

### 🛠️ ส่วนประกอบสำคัญในระบบพื้นฐาน
1. **`requirements.txt` ที่ล็อกเวอร์ชันชัดเจน**: ป้องกันปัญหา Dependency Rot หรือระบบพังจากการอัปเดตเวอร์ชันของ Library ในอนาคต
2. **`config.yaml` + Config Loader**: แยกพารามิเตอร์และตัวแปรในการตั้งค่าออกจาก Logic ทั้งหมดอย่างเด็ดขาด ง่ายต่อการ Tuning โมเดล
3. **Production Logger**: ตัวจัดการระบบ Log อัตโนมัติ ป้องกันปัญหา Log ไฟล์ใหญ่เกินไปจนฮาร์ดดิสก์เต็มและทำให้เซิร์ฟเวอร์ค้าง

---

## 🔄 ตอนที่ 2: Data Loader & Cross-Asset Synchronization

การสร้าง Data Pipeline โดยจัดการกับข้อจำกัดจริงในโลกของการทำ Quant (Quant Reality):

### ⚠️ ข้อจำกัดและกฎเหล็กที่ต้องจัดการ
* **ข้อจำกัดของ Yahoo Finance (YFinance):** ไม่อนุญาตให้ดาวน์โหลดข้อมูล Timeframe ย่อยระดับนาที (เช่น 15m) ย้อนหลังเกิน 60 วัน (หากต้องการข้อมูลย้อนหลัง 1-3 ปีในอนาคต ต้องเชื่อมต่อ API ของ OANDA, Polygon หรือ MT5) ในเฟส PoC นี้ ระบบจะบังคับให้ดึงข้อมูลย้อนหลังสูงสุดได้ 60 วันโดยอัตโนมัติ
* **การทำ Multi-Timeframe ไม่ใช่แค่การ Merge ข้อมูล:** กฎเหล็กคือต้องคำนวณ Indicator ของ Timeframe ใหญ่ (D1, H4) ให้เสร็จสิ้นก่อน แล้วจึงนำค่าที่ได้มา Merge ร่วมกับแท่งราคา 15m หากนำราคาดิบมา Merge ก่อน ค่าอินดิเคเตอร์ที่คำนวณได้จะไม่ตรงกับกราฟจริง

### 🎯 ขอบเขตการทำงานในเฟสนี้
ระบบจะทำการโหลดและเชื่อมโยงข้อมูลข้ามสินทรัพย์ (**Cross-Asset Synchronization**) โดยนำข้อมูลราคาทองคำ (Gold), เงิน (Silver) และดัชนีดอลลาร์ (US Dollar Index) มารวมเข้าด้วยกันในมิติของเวลาที่ถูกต้อง ส่วนระบบ Multi-Timeframe จะถูกยกไปรวมในขั้นตอนถัดไป

---

## 🧪 ตอนที่ 3: Feature Engineering Pipeline

เฟสสำหรับการแปลงข้อมูลดิบ (OHLCV) ให้กลายเป็น **Data Matrix ขนาด 40-80 คอลัมน์** เพื่อพร้อมสำหรับส่งต่อให้ Machine Learning นำไปประมวลผล

### 🚀 ขั้นตอนการทำงานและจุดเด่น
1. **การคำนวณ Technical Indicator:** ใช้ไลบรารี `ta` (Technical Analysis) เพื่อลดความซ้ำซ้อนและป้องกันความผิดพลาดจากการเขียนสูตรคำนวณเอง
2. **การสร้าง Target Label ที่ถูกต้อง:** ออกแบบฟังก์ชันคำนวณ Target ด้วยเงื่อนไข:
   $$\text{Target Label} = N\text{-bars ahead} \ge 0.5 \times \text{ATR}$$
3. **ป้องกัน Look-ahead Bias:** ออกแบบโครงสร้างคลาสเฉพาะ (`Feature Pipeline Class`) เพื่อควบคุมทิศทางการไหลของข้อมูล ป้องกันไม่ให้โมเดลเห็นข้อมูลในอนาคตระหว่างการคำนวณโดยเด็ดขาด
