from src.utils.config_loader import Config
from src.utils.logger import setup_logger
from src.data_pipeline.yfinance_loader import YFinanceLoader
from src.data_pipeline.synchronizer import DataSynchronizer
from src.data_pipeline.feature_engineer import FeatureEngineer
import pandas as pd
import os

def main():
    cfg = Config.load("configs/config.yaml")
    logger = setup_logger(mode=cfg["project"]["mode"])
    
    logger.info("==================================================")
    logger.info(f"🚀 Starting {cfg['project']['name']} [Phase 1-3]")
    logger.info("==================================================")

    # ---------------- Phase 2: Data Loader ----------------
    loader = YFinanceLoader()
    base_tf = cfg["data"]["timeframes"][-1] 
    
    target_df = loader.fetch_data(cfg["data"]["symbols"]["target"], base_tf, cfg["data"]["lookback_days"])
    aux_dfs = {sym: loader.fetch_data(sym, base_tf, cfg["data"]["lookback_days"]) for sym in cfg["data"]["symbols"]["auxiliary"]}
    
    if target_df.empty:
        logger.error("หยุดการทำงาน: ไม่พบข้อมูล Target")
        return

    synced_data = DataSynchronizer.merge_cross_assets(target_df, aux_dfs)

    # ---------------- Phase 3: Feature Engineering ----------------
    # ดึงค่าพารามิเตอร์จาก Config
    fe = FeatureEngineer(
        atr_period=cfg["risk"]["atr_period"],
        target_atr_mult=cfg["model"]["target_atr_multiplier"],
        forward_bars=3  # 3 แท่งหน้า (สามารถย้ายไปใส่ config ได้)
    )
    
    # 3.1 สร้าง Features 
    data_with_features = fe.generate_features(synced_data)
    
    # 3.2 สร้าง Target Label และ Drop NaN
    final_data = fe.generate_target(data_with_features)
    
    # บันทึก Dataset ที่พร้อมเข้าโมเดล ML
    save_path = "data/processed/ml_dataset.parquet"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    final_data.to_parquet(save_path)
    
    logger.success(f"Phase 3 เสร็จสมบูรณ์! Dataset Shape: {final_data.shape}")
    logger.success(f"จำนวน Feature ทั้งหมด: {len(final_data.columns) - 2} Features")
    
    # ดูความสมดุลของ Target (Class Imbalance Check)
    buy_signals = final_data['target_long'].sum()
    sell_signals = final_data['target_short'].sum()
    logger.info(f"โอกาสเกิด Long: {buy_signals} ครั้ง, Short: {sell_signals} ครั้ง จากทั้งหมด {len(final_data)} แท่ง")

if __name__ == "__main__":
    main()