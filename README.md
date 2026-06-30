ตอนที่ 1: Project Architecture + Environment + Configuration
เป้าหมายของตอนที่ 1 ไม่ใช่การเขียนโค้ดเทรด แต่คือการ "กันระบบพังตอนตี 3" เมื่อคุณต้องเอาไปรันบน Windows Server สิ่งที่เราจะสร้างในวันนี้มี 4 โครงสร้างหลัก:
Folder Architecture แบบ Separation of Concerns

alpha_quant_ai/
│
├── configs/
│   └── config.yaml            # ศูนย์บัญชาการพารามิเตอร์ทั้งหมด
│
├── data/
│   ├── raw/                   # เก็บไฟล์ csv ดิบ (ห้ามแก้)
│   └── processed/             # เก็บ parquet หลังจากทำ Feature แล้ว
│
├── logs/                      # เก็บไฟล์ Log รายวัน (Auto-rotate)
├── models/                    # เก็บไฟล์ .pkl หรือ .joblib ของ LightGBM/XGBoost
│
├── src/
│   ├── __init__.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config_loader.py   # ตัวแปลง YAML เป็น Python Dict
│   │   └── logger.py          # ตัวจัดการ Loguru
│   │
│   ├── data_pipeline/         # (สำหรับตอนที่ 2 และ 3)
│   ├── ml_engine/             # (สำหรับตอนที่ 4 และ 5)
│   ├── backtester/            # (สำหรับตอนที่ 6)
│   └── live_trader/           # (สำหรับตอนที่ 7 และ 8)
│
├── .gitignore
├── requirements.txt
└── main.py                    # Entry point ของระบบ

requirements.txt ที่ล็อกเวอร์ชันชัดเจน (ต้านทาน Dependency Rot)
config.yaml + Config Loader (แยก Parameter ออกจาก Logic 100%)
Production Logger (ป้องกันปัญหา Log ไฟล์ใหญ่จน Harddisk เต็มแล้วเซิร์ฟเวอร์ค้าง)

ตอนที่ 2: เมื่อโครงสร้างพื้นฐานพร้อมแล้ว เราจะเข้าสู่หัวใจของการทำ Data Pipeline ใน ตอนที่ 2: Data Loader & Cross-Asset Synchronization
ในเฟสนี้เราจะต้องเจอกับ "ความจริงของโลก Quant" 2 ข้อ ที่เราต้องจัดการก่อนเขียนโค้ด:
ข้อจำกัดของ Yahoo Finance: YFinance ไม่อนุญาตให้โหลดข้อมูล Timeframe ย่อยระดับนาที (เช่น 15m) ย้อนหลังเกิน 60 วัน (หากจะเอาย้อนหลัง 1-3 ปี ต้องไปต่อ API ของ OANDA, Polygon หรือ MT5) ดังนั้นในตอนที่ 2 นี้ ระบบจะบังคับให้ดึงได้สูงสุด 60 วันโดยอัตโนมัติ เพื่อให้เราพัฒนาระบบจนจบ PoC ก่อน
การทำ Multi-Timeframe ไม่ใช่แค่การ Merge: กฎเหล็กคือ ต้องคำนวณ Indicator ของ Timeframe ใหญ่ (D1, H4) ให้เสร็จก่อน แล้วค่อยนำมา Merge กับแท่ง 15M หากเรา Merge ราคาดิบๆ ลงมาก่อน ค่าที่คำนวณจะได้ไม่ตรงกับกราฟจริง ดังนั้นในเฟสนี้เราจะทำ Cross-Asset Synchronization (นำทอง, ซิลเวอร์, ดอลลาร์ มารวมกัน) ก่อน ส่วน Multi-Timeframe จะถูกนำไปรวมอยู่ในตอนที่ 3 (Feature Engineering)

ตอนที่ 3: 
ในเฟสนี้เราจะแปลงข้อมูล OHLCV ธรรมดาให้กลายเป็น Data Matrix ขนาด 40-80 คอลัมน์ โดยใช้ไลบรารี ta (Technical Analysis) เพื่อลดความซ้ำซ้อนของการเขียนสูตรเอง และไฮไลต์สำคัญคือ การสร้าง Target Label ($N$-bars ahead $\ge 0.5 \times \text{ATR}$) ที่ถูกต้องโดยไม่เกิด Look-ahead bias ครับขั้นตอนการสร้าง Feature Engineering Pipelineเราจะสร้างคลาสใหม่เพื่อรับผิดชอบการสร้าง Features และ Target โดยเฉพาะ
