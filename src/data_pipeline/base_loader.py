from abc import ABC, abstractmethod
import pandas as pd

class BaseDataLoader(ABC):
    @abstractmethod
    def fetch_data(self, symbol: str, interval: str, lookback_days: int) -> pd.DataFrame:
        """
        ดึงข้อมูลราคาประวัติศาสตร์
        :param symbol: ชื่อคู่เงิน/สินทรัพย์
        :param interval: กรอบเวลา เช่น '15m', '1h', '1d'
        :param lookback_days: จำนวนวันย้อนหลัง
        """
        pass