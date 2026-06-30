import sys
from pathlib import Path
from loguru import logger

def setup_logger(mode: str = "development"):
    logger.remove() # ลบ Default handler ของ Loguru ทิ้งก่อน

    # 1. แสดงผลบน Console (มีสีสัน สำหรับตอนนั่งแก้โค้ด)
    log_level = "DEBUG" if mode == "development" else "INFO"
    logger.add(
        sys.stdout, 
        level=log_level, 
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
    )

    # 2. บันทึกลง Harddisk (สำหรับ Production)
    log_path = Path("logs") / "trading_system.log"
    logger.add(
        log_path,
        level="DEBUG",
        rotation="100 MB",     # ขึ้นไฟล์ใหม่เมื่อเกิน 100MB
        retention="30 days",   # เก็บย้อนหลังแค่ 30 วัน
        compression="zip",     # บีบอัดไฟล์เก่าเป็น .zip
        encoding="utf-8"
    )
    
    return logger