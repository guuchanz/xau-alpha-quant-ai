from src.utils.config_loader import Config
from src.utils.logger import setup_logger
from src.data_pipeline.yfinance_loader import YFinanceLoader
from src.data_pipeline.synchronizer import DataSynchronizer
from src.data_pipeline.feature_engineer import FeatureEngineer
from src.ml_engine.lgb_trainer import LightGBMTrainer
from src.backtester.validator import WalkForwardValidator
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
    # buy_signals = final_data['target_long'].sum()
    # sell_signals = final_data['target_short'].sum()
    # logger.info(f"โอกาสเกิด Long: {buy_signals} ครั้ง, Short: {sell_signals} ครั้ง จากทั้งหมด {len(final_data)} แท่ง")

    # สมมติว่าตอนนี้เรามี final_data จาก Phase 3 แล้ว
    # โหลดไฟล์ที่ทำไว้จาก Phase 3 กลับมาเพื่อความรวดเร็ว
    dataset_path = "data/processed/ml_dataset.parquet"
    if not os.path.exists(dataset_path):
        logger.error("ไม่พบ Dataset! กรุณารัน Phase 1-3 ให้ผ่านก่อน")
        return
        
    final_data = pd.read_parquet(dataset_path)

    # ---------------- Phase 4: Machine Learning Engine ----------------
    logger.info("เตรียมข้อมูล X (Features) และ y (Target)...")
    
    # คัดเฉพาะคอลัมน์ที่เป็น Feature (ตัด OHLCV ดิบๆ และ Target ทิ้ง)
    # เราไม่ให้โมเดลเห็นราคาดิบ (close) เพราะมันเกิด Non-stationary problem
    drop_cols = ['target_open', 'target_high', 'target_low', 'target_close', 'target_volume', 'target_long', 'target_short']
    
    # ถ้ามีข้อมูลของ Auxiliary ติดมาด้วย ก็ดรอปราคาดิบทิ้งเช่นกัน
    drop_cols.extend([col for col in final_data.columns if 'close' in col.lower() or 'open' in col.lower()])
    
    X = final_data.drop(columns=[col for col in drop_cols if col in final_data.columns])
    # เราจะสร้างโมเดลสำหรับสัญญาณ BUY (Long) ก่อน
    y_long = final_data['target_long']
    
    logger.info(f"จำนวน Features ทั้งหมดที่จะเข้าโมเดล: {X.shape[1]}")

    # 4.1 เริ่มกระบวนการ Train (ใช้ Optuna หาพารามิเตอร์ 20 รอบ เพื่อประหยัดเวลาทดสอบ)
    lgb_long_trainer = LightGBMTrainer(n_trials=10) # ลดเหลือ 10 รอบเพื่อให้รันเร็วขึ้น
    lgb_long_trainer.optimize_and_train(X, y_long)
    best_params = lgb_long_trainer.best_params
    # 4.2 ตรวจสอบความสำคัญของ Features (Feature Importance)
    # importance = pd.DataFrame({
    #     'Feature': X.columns,
    #     'Importance': model_long.feature_importance(importance_type='gain')
    # }).sort_values(by='Importance', ascending=False)
    
    # logger.info("\n--- 🌟 Top 5 Feature Importance 🌟 ---")
    # print(importance.head(5).to_string(index=False))
    
    # 4.3 เซฟโมเดลเก็บไว้
    # import joblib
    # os.makedirs('models', exist_ok=True)
    # joblib.dump(model_long, 'models/lgb_long_model.pkl')
    # logger.success("บันทึกโมเดลสำเร็จที่: models/lgb_long_model.pkl")

# ---------------- Phase 5: Walk-Forward Validation ----------------
    validator = WalkForwardValidator(n_splits=5)
    
    logger.info("นำ Best Params มาทดสอบเสมือนจริงด้วย Walk-Forward...")
    oof_preds, avg_precision = validator.evaluate(X, y_long, best_params)
    
    # เพิ่มผลการทำนาย (Probability) กลับเข้าไปใน Dataset จริง เพื่อเตรียมส่งต่อให้ Phase 6
    final_data['long_signal_prob'] = oof_preds
    
    # บันทึก Dataset ที่มี Probability เพื่อนำไปแบ็คเทสต์ต่อ
    save_path = "data/processed/backtest_dataset.parquet"
    final_data.to_parquet(save_path)
    
    logger.success(f"บันทึกข้อมูลเตรียมทำ Backtest เรียบร้อยที่: {save_path}")
    logger.info("==================================================")
    logger.info(f"🎯 สรุปความพร้อมของ AI: มีความแม่นยำในการเข้าทำกำไรที่ {avg_precision:.2%}")
    logger.info("==================================================")

if __name__ == "__main__":
    main()
