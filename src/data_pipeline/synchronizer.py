import pandas as pd
from loguru import logger
from typing import Dict

class DataSynchronizer:
    @staticmethod
    def merge_cross_assets(target_df: pd.DataFrame, aux_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        รวมข้อมูลต่างสินทรัพย์เข้าด้วยกัน โดยยึด Timestamp ของ Target เป็นแกนหลัก
        """
        logger.info("เริ่มกระบวนการ Cross-Asset Synchronization...")
        
        # 1. เปลี่ยนชื่อ Column ของ Target เพื่อไม่ให้ซ้ำซ้อน
        merged_df = target_df.copy()
        merged_df.columns = [f"target_{col.lower()}" for col in merged_df.columns]
        
        # 2. นำ Auxiliary เข้ามา Left Join ทะลุ Index
        for symbol, aux_df in aux_dfs.items():
            aux_cleaned = aux_df.copy()
            # ใช้ชื่อ Symbol ที่ถูกตั้งใน Config มาทำ Prefix (ตัวอย่าง: DX-Y.NYB_close)
            safe_symbol = symbol.replace("=", "").replace(".", "_").replace("-", "")
            aux_cleaned.columns = [f"{safe_symbol}_{col.lower()}" for col in aux_cleaned.columns]
            
            merged_df = merged_df.join(aux_cleaned, how="left")
            
        # 3. จัดการ Missing Values (Data Cleansing)
        missing_before = merged_df.isna().sum().sum()
        
        # กฎเหล็ก Quant: ห้ามใช้ bfill() (Backward Fill) เพราะเป็นการเอาอนาคตมาเติมอดีต (Data Leakage)
        # ต้องใช้ ffill() (Forward Fill) เพื่อดึงราคาปิดของแท่งก่อนหน้ามาใช้ในกรณีที่สินทรัพย์นั้นตลาดปิด
        merged_df = merged_df.ffill()
        
        # หากยังมี NaN อยู่ที่แท่งแรกสุด (เพราะไม่มีข้อมูลให้ ffill) ให้ Drop ทิ้ง
        merged_df = merged_df.dropna()
        
        missing_after = merged_df.isna().sum().sum()
        logger.debug(f"Missing Values Cleared: ก่อน {missing_before} -> หลัง {missing_after}")
        logger.success(f"Final Synchronized Shape: {merged_df.shape}")
        
        return merged_df