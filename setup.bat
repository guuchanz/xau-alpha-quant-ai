@echo off
REM เปลี่ยนไปยังโฟลเดอร์โปรเจกต์
cd /d %~dp0

REM ติดตั้ง dependencies จาก requirements.txt
pip install -r requirements.txt

REM รันโปรแกรมหลัก main.py
python main.py

REM รอให้ผู้ใช้กดปิดหน้าต่าง
pause
