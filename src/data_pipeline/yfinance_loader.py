import yfinance as yf
import pandas as pd
from loguru import logger
from datetime import datetime, timedelta
from .base_loader import BaseDataLoader

class YFinanceLoader(BaseDataLoader):
    def fetch_data(self, symbol: str, interval: str, lookback_days: int) -> pd.DataFrame:
        # กฎข้อบังคับของ YFinance สำหรับ Intraday
        if interval in ["1m", "5m", "15m", "30m"] and lookback_days > 60:
            logger.warning(f"YFinance จำกัดข้อมูล {interval} สูงสุด 60 วัน (ปรับอัตโนมัติจาก {lookback_days} -> 60)")
            lookback_days = 60
            
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        logger.info(f"Downloading {symbol} [{interval}] from {start_date.date()} to {end_date.date()}")
        
        try:
            # ใช้ progress=False เพื่อไม่ให้ Terminal รก
            df = yf.download(symbol, start=start_date, end=end_date, interval=interval, progress=False)
            
            if df.empty:
                logger.error(f"ไม่พบข้อมูลสำหรับ {symbol} ที่ Timeframe {interval}")
                return pd.DataFrame()
            
            # แก้ปัญหา YFinance เวอร์ชันใหม่ที่คืนค่ามาเป็น MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            df.index.name = "timestamp"
            
            # แปลง Timezone ให้เป็น UTC เสมอ
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            else:
                df.index = df.index.tz_convert('UTC')
                
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        except Exception as e:
            logger.error(f"Error downloading {symbol}: {str(e)}")
            return pd.DataFrame()