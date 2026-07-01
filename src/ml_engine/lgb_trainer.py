import lightgbm as lgb
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss
from loguru import logger

class LightGBMTrainer:
    def __init__(self, n_trials: int = 30):
        self.n_trials = n_trials
        self.best_params = None
        self.model = None

    def _objective(self, trial, X, y):
        # 1. กำหนด Search Space (บังคับให้ตื้น เพื่อลด Overfitting)
        param = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 8, 32), # ห้ามเกิน 32
            'max_depth': trial.suggest_int('max_depth', 3, 5),    # กฎเหล็ก: ต้นไม้ต้องตื้น
            'min_child_samples': trial.suggest_int('min_child_samples', 50, 200),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.9),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
            'bagging_freq': 1,
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0, log=True), # L1 Regularization
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True) # L2 Regularization
        }

        # 2. ใช้ TimeSeriesSplit ป้องกันการแอบดูอนาคตตอนจูนโมเดล
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            # ใช้ early_stopping เพื่อหยุดเมื่อโมเดลเริ่ม Overfit
            callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False)]
            
            gbm = lgb.train(
                param,
                train_data,
                valid_sets=[val_data],
                num_boost_round=500,
                callbacks=callbacks
            )

            preds = gbm.predict(X_val)
            score = log_loss(y_val, preds)
            scores.append(score)

        return np.mean(scores)

    def optimize_and_train(self, X: pd.DataFrame, y: pd.Series):
        logger.info("เริ่มกระบวนการ Optuna Hyperparameter Tuning (LightGBM)...")
        optuna.logging.set_verbosity(optuna.logging.WARNING) # ซ่อน Log รกๆ
        
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: self._objective(trial, X, y), n_trials=self.n_trials)
        
        self.best_params = study.best_params
        self.best_params['objective'] = 'binary'
        self.best_params['verbosity'] = -1
        
        logger.success(f"ค้นพบ Best Params: {self.best_params}")
        
        # เทรนโมเดลจริงด้วยพารามิเตอร์ที่ดีที่สุด บนข้อมูลทั้งหมดที่ให้มา
        logger.info("กำลัง Train โมเดลรอบสุดท้ายด้วย Best Params...")
        train_data = lgb.Dataset(X, label=y)
        self.model = lgb.train(self.best_params, train_data, num_boost_round=150)
        
        return self.model

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("โมเดลยังไม่ถูก Train!")
        return self.model.predict(X)