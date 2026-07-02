from src.backtester.vector_backtest import VectorBacktester
from src.utils.config_loader import Config
from src.utils.logger import setup_logger
from src.data_pipeline.yfinance_loader import YFinanceLoader
from src.data_pipeline.synchronizer import DataSynchronizer
from src.data_pipeline.feature_engineer import FeatureEngineer
from src.ml_engine.lgb_trainer import LightGBMTrainer
from src.backtester.validator import WalkForwardValidator
from src.backtester.vector_backtest import VectorBacktester
from src.backtester.optimizer import StrategyOptimizer
import vectorbt as vbt
import pandas as pd
import os

def main():
    cfg = Config.load("configs/config.yaml")
    logger = setup_logger(mode=cfg["project"]["mode"])
    
    logger.info("==================================================")
    logger.info(f"🚀 Starting {cfg['project']['name']} [Phase 1: Project Architecture + Environment + Configuration]")
    logger.info("==================================================")

    # ---------------- Phase 2: Data Loader ----------------
    logger.info("==================================================")
    logger.info(f"🚀 Starting {cfg['project']['name']} [Phase 2: Data Loader]")
    logger.info("==================================================")
    loader = YFinanceLoader()
    base_tf = cfg["data"]["timeframes"][-1] 
    
    target_df = loader.fetch_data(cfg["data"]["symbols"]["target"], base_tf, cfg["data"]["lookback_days"])
    aux_dfs = {sym: loader.fetch_data(sym, base_tf, cfg["data"]["lookback_days"]) for sym in cfg["data"]["symbols"]["auxiliary"]}
    
    if target_df.empty:
        logger.error("หยุดการทำงาน: ไม่พบข้อมูล Target")
        return

    synced_data = DataSynchronizer.merge_cross_assets(target_df, aux_dfs)

    # ---------------- Phase 3: Feature Engineering ----------------
    logger.info("==================================================")
    logger.info(f"🚀 Starting {cfg['project']['name']} [Phase 3: Feature Engineering]")
    logger.info("==================================================")
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

    # สมมติว่าตอนนี้เรามี final_data จาก Phase 3 แล้ว
    # โหลดไฟล์ที่ทำไว้จาก Phase 3 กลับมาเพื่อความรวดเร็ว
    dataset_path = "data/processed/ml_dataset.parquet"
    if not os.path.exists(dataset_path):
        logger.error("ไม่พบ Dataset! กรุณารัน Phase 1-3 ให้ผ่านก่อน")
        return
        
    final_data = pd.read_parquet(dataset_path)

    # ---------------- Phase 4: Machine Learning Engine ----------------
    logger.info("==================================================")
    logger.info(f"🚀 Starting {cfg['project']['name']} [Phase 4: Machine Learning Engine]")
    logger.info("==================================================")
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
    lgb_long_trainer = LightGBMTrainer(n_trials=30) # ลดเหลือ 10 รอบเพื่อให้รันเร็วขึ้น
    lgb_long_trainer.optimize_and_train(X, y_long)
    
    best_params = lgb_long_trainer.best_params

    # 4.2 ตรวจสอบความสำคัญของ Features (Feature Importance)
    model_long = lgb_long_trainer.optimize_and_train(X, y_long)
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model_long.feature_importance(importance_type='gain')
    }).sort_values(by='Importance', ascending=False)
    
    logger.info("\n--- 🌟 Top 5 Feature Importance 🌟 ---")
    print(importance.head(5).to_string(index=False))
    
    # 4.3 เซฟโมเดลเก็บไว้
    import joblib
    os.makedirs('models', exist_ok=True)
    joblib.dump(model_long, 'models/lgb_long_model.pkl')
    logger.success("บันทึกโมเดลสำเร็จที่: models/lgb_long_model.pkl")

# ---------------- Phase 5: Walk-Forward Validation ----------------
    logger.info("==================================================")
    logger.info(f"🚀 Starting {cfg['project']['name']} [Phase 5: Walk-Forward Validation]")
    logger.info("==================================================")

    validator = WalkForwardValidator(n_splits=50)
    
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

    logger.info("==================================================")
    logger.info(f"🚀 Starting {cfg['project']['name']} [Phase 6: Backtesting]")
    logger.info("==================================================")

    # โหลดข้อมูลที่มีค่า Probability จาก Phase 5
    dataset_path = "data/processed/backtest_dataset.parquet"
    if not os.path.exists(dataset_path):
        logger.error("ไม่พบ Dataset! กรุณารัน Phase 1-5 ให้ผ่านก่อน")
        return
        
    final_data = pd.read_parquet(dataset_path)

    # ตรวจสอบว่ามีคอลัมน์เป้าหมายและ probability ครบไหม
    if 'target_close' not in final_data.columns:
        logger.error("ข้อมูลไม่ครบถ้วน ขาด target_close")
        return

    if 'long_signal_prob' not in final_data.columns:
        final_data['long_signal_prob'] = 0.5
    if 'short_signal_prob' not in final_data.columns:
        final_data['short_signal_prob'] = 0.5

    # ---------------- Phase 6: Vectorized Backtesting ----------------
    # ตั้งค่า Entry/Exit (ปรับตามความมั่นใจของโมเดล)
    # เนื่องจากความแม่นยำเราอยู่ที่ ~58% การเข้าที่ 0.55 ถือว่าปลอดภัยระดับนึง
    backtester = VectorBacktester(
        init_cash=10000.0, 
        fees=0.0002, # เผื่อค่า Spread และ Slippage ไว้ที่ 0.02%
        entry_threshold=0.66, 
        exit_threshold=0.48 
    )
    
    portfolio = backtester.run_bidirectional_backtest(final_data)

    logger.info("==================================================")
    logger.info(f"🚀 Starting {cfg['project']['name']} [Phase 7: Optimization]")
    logger.info("==================================================")

    dataset_path = "data/processed/backtest_dataset.parquet"
    if not os.path.exists(dataset_path):
        logger.error("ไม่พบ Dataset! กรุณารัน Phase 1-5 ให้ผ่านก่อน")
        return
        
    final_data = pd.read_parquet(dataset_path)

    # ---------------- Phase 7: Strategy Optimization ----------------
    optimizer = StrategyOptimizer(
        init_cash=10000.0, 
        fees=0.0002 
    )
    
    # รันทดสอบ 27 รูปแบบ
    results_df, best_portfolio = optimizer.run_sweep(final_data)
    
    logger.info("\n🏆 --- Top 5 Parameters ที่ทำกำไรสูงสุด --- 🏆")
    print(results_df.head(5).to_string(index=False))
    
    # แสดงสถิติเชิงลึกของอันดับ 1
    logger.info("\n📊 --- สถิติของกลยุทธ์ที่ดีที่สุด --- 📊")
    stats = best_portfolio.stats()
    for metric in ['Start Value', 'End Value', 'Max Drawdown [%]', 'Total Trades']:
        if metric in stats:
            print(f"{metric+':':<20} {stats[metric]}")
    
    # บันทึกกราฟอันดับ 1 ไว้ดู
    os.makedirs('reports', exist_ok=True)
    plot_path = "reports/best_optimized_backtest.html"
    best_portfolio.plot().write_html(plot_path)
    logger.success(f"บันทึกกราฟกลยุทธ์ที่ดีที่สุดเรียบร้อยที่: {plot_path}")

    logger.info("==================================================")
    logger.info("🚀 Phase 8: Unlocking Short Side (Two-Way Trading) - Continuous Pipeline")
    logger.info("==================================================")

    # โหลด Dataset หลักที่เตรียมไว้ตั้งแต่ Phase 3
    dataset_path = "data/processed/ml_dataset.parquet"
    if not os.path.exists(dataset_path):
        logger.error("ไม่พบ Dataset! กรุณารัน Phase 1-3 ให้ผ่านก่อน")
        return
        
    final_data = pd.read_parquet(dataset_path)

    # คัดแยกเฉพาะ Features (X) ออกมาจากข้อมูลหลัก
    drop_cols = ['target_open', 'target_high', 'target_low', 'target_close', 'target_volume', 'target_long', 'target_short']
    drop_cols.extend([col for col in final_data.columns if 'close' in col.lower() or 'open' in col.lower()])
    X = final_data.drop(columns=[col for col in drop_cols if col in final_data.columns])

    # ---------------- 🤖 1. เทรนโมเดลฝั่ง LONG ----------------
    logger.info("--- [🤖 ขั้นตอนที่ 1: เทรนสมองกลฝั่ง LONG] ---")
    y_long = final_data['target_long']
    trainer_long = LightGBMTrainer(n_trials=30)
    trainer_long.optimize_and_train(X, y_long)
    
    validator_long = WalkForwardValidator(n_splits=50)
    oof_long, _ = validator_long.evaluate(X, y_long, trainer_long.best_params)

    # ---------------- 🤖 2. เทรนโมเดลฝั่ง SHORT ----------------
    logger.info("--- [🤖 ขั้นตอนที่ 2: เทรนสมองกลฝั่ง SHORT] ---")
    y_short = final_data['target_short'] 
    trainer_short = LightGBMTrainer(n_trials=30)
    trainer_short.optimize_and_train(X, y_short)
    
    validator_short = WalkForwardValidator(n_splits=50)
    oof_short, _ = validator_short.evaluate(X, y_short, trainer_short.best_params)

    # ---------------- 🛠️ 3. ปรับโครงสร้างข้อมูล (Data Padding & Alignment) ----------------
    logger.info("--- [🛠️ ขั้นตอนที่ 3: จัดการระยะแถวข้อมูลให้เท่ากันเพื่อป้องกัน KeyError] ---")
    total_rows = len(final_data)
    
    # กรณีทำ TimeSeriesSplit แล้วแถวผลลัพธ์สั้นกว่าข้อมูลดิบ ให้เติมค่า 0.5 (ไม่มั่นใจ) ในแถวแรกๆ
    if len(oof_long) < total_rows:
        diff_long = total_rows - len(oof_long)
        oof_long = np.concatenate([np.full(diff_long, 0.5), oof_long])
        
    if len(oof_short) < total_rows:
        diff_short = total_rows - len(oof_short)
        oof_short = np.concatenate([np.full(diff_short, 0.5), oof_short])

    # ผูกค่าลง DataFrame หลักเพื่อเตรียมใช้ส่งสัญญานเทรด
    final_data['long_signal_prob'] = oof_long
    final_data['short_signal_prob'] = oof_short

    if 'long_signal_prob' not in final_data.columns:
        final_data['long_signal_prob'] = 0.5
    if 'short_signal_prob' not in final_data.columns:
        final_data['short_signal_prob'] = 0.5

    # ---------------- 📈 ขั้นตอนที่ 4: ปรับจูนสู่จุดสมดุลเพื่อพลิกเป็นบวก ----------------
    logger.info("--- [📈 ขั้นตอนที่ 4: เริ่มต้นทำแบ็คเทสต์แบบ Positive Expectancy Engine] ---")
    
    # 1. ยกระดับเกณฑ์เป็น 0.70 เพื่อคัดเอาเฉพาะจังหวะ "เน้นๆ" ลดจำนวนเทรดขยะจาก 215 ครั้ง
    long_entries = final_data['long_signal_prob'] >= 0.80
    long_exits = final_data['long_signal_prob'] < 0.50
    
    short_entries = final_data['short_signal_prob'] >= 0.80
    short_exits = final_data['short_signal_prob'] < 0.50
    
    # 2. แก้ไขสัดส่วนความเสี่ยงให้ได้เปรียบเชิงคณิตศาสตร์ (Win Rate 57% + R:R 1:1.75)
    portfolio = vbt.Portfolio.from_signals(
        close=final_data['target_close'],
        entries=long_entries,
        exits=long_exits,
        short_entries=short_entries, 
        short_exits=short_exits,     
        fees=0.0002,
        init_cash=10000.0,
        freq='15T',
        direction='both',
        sl_stop=0.02,   # <--- เปลี่ยนเป็น 0.02 (จำกัดแผลแค่ 2%)
        tp_stop=0.035   # <--- เปลี่ยนเป็น 0.035 (รันกำไรคำโต 3.5%)
    )
    
    stats = portfolio.stats()
    
    logger.info("\n==================================================")
    logger.info("🛡️ ผลประกอบการเวอร์ชัน Positive Expectancy Engine 🛡️")
    logger.info("==================================================")
    
    metrics_to_show = ['Start Value', 'End Value', 'Total Return [%]', 'Max Drawdown [%]', 'Total Trades', 'Win Rate [%]']
    for metric in metrics_to_show:
        if metric in stats:
            print(f"{metric+':':<20} {stats[metric]}")

if __name__ == "__main__":
    main()