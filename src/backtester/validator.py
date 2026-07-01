import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

class WalkForwardValidator:
    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits

    def evaluate(self, X: pd.DataFrame, y: pd.Series, best_params: dict):
        logger.info(f"เริ่มทำ Walk-Forward Validation จำนวน {self.n_splits} Folds...")
        
        # ใช้ TimeSeriesSplit เพื่อให้ข้อมูล Train อยู่ก่อนหน้า Test เสมอ
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        fold_metrics = []
        oof_predictions = np.zeros(len(X)) # Out-of-Fold Predictions
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
            
            # ป้องกัน Data Leakage ระหว่างรอยต่อ Train/Test ด้วยการเว้นระยะ (Purge) 
            # (ข้ามแท่งไป 5 แท่ง เพื่อให้ Indicator ที่คำนวณคาบเกี่ยวกันขาดออกจากกัน)
            purge_bars = 5
            if len(X_train) > purge_bars:
                X_train = X_train.iloc[:-purge_bars]
                y_train = y_train.iloc[:-purge_bars]
            
            # Train โมเดลเฉพาะใน Fold นี้
            train_data = lgb.Dataset(X_train, label=y_train)
            
            # เราจะไม่ใช้ Early Stopping ด้วย Test Set ในรอบนี้ เพราะนี่คือการจำลองเทรดจริง
            # (ต้องตั้งสมมติฐานว่าเราไม่เห็น Test Set เลย)
            model = lgb.train(
                best_params,
                train_data,
                num_boost_round=100, # ใช้จำนวนต้นไม้ที่คงที่
            )
            
            # นำไปทำนายใน Test Set (อนาคตที่ไม่เคยเห็น)
            preds_prob = model.predict(X_test)
            oof_predictions[test_idx] = preds_prob
            
            # แปลงความน่าจะเป็นให้เป็นสัญญาณซื้อขาย (Threshold = 0.5)
            preds_binary = (preds_prob >= 0.5).astype(int)
            
            # คำนวณ Metrics สำหรับ Fold นี้
            # ในสาย Quant เราสนใจ Precision (ความแม่นยำเมื่อสัญญาณบอกให้ซื้อ) มากกว่า Accuracy
            precision = precision_score(y_test, preds_binary, zero_division=0)
            
            logger.debug(f"Fold {fold+1}/{self.n_splits} | Train: {len(X_train)} bars, Test: {len(X_test)} bars | Precision: {precision:.2%}")
            fold_metrics.append(precision)

        # สรุปผลรวม
        avg_precision = np.mean(fold_metrics)
        logger.success(f"✅ Walk-Forward Validation เสร็จสมบูรณ์ | Average Precision: {avg_precision:.2%}")
        
        return oof_predictions, avg_precision