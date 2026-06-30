import pandas as pd
import numpy as np
import ta
from loguru import logger

class FeatureEngineer:
    def __init__(self, atr_period: int = 14, target_atr_mult: float = 0.5, forward_bars: int = 3):
        self.atr_period = atr_period
        self.target_atr_mult = target_atr_mult
        self.forward_bars = forward_bars

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("เริ่มสร้าง Technical & Cross-Asset Features...")
        df = df.copy()
        
        # --- 1. Volatility (ต้องสร้าง ATR ก่อนเพื่อไปใช้กับ Target) ---
        df['atr'] = ta.volatility.AverageTrueRange(
            high=df['target_high'], low=df['target_low'], close=df['target_close'], window=self.atr_period
        ).average_true_range()
        
        bb = ta.volatility.BollingerBands(close=df['target_close'], window=20, window_dev=2)
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_width'] = bb.bollinger_pband() # Band width percentage
        
        # --- 2. Trend & Moving Averages ---
        for p in [20, 50, 100, 200]:
            df[f'ema_{p}'] = ta.trend.EMAIndicator(close=df['target_close'], window=p).ema_indicator()
        
        # ระยะห่างระหว่างเส้น (Distance)
        df['ema_20_50_dist'] = (df['ema_20'] - df['ema_50']) / df['ema_50']
        df['ema_50_200_dist'] = (df['ema_50'] - df['ema_200']) / df['ema_200']
        
        macd = ta.trend.MACD(close=df['target_close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()
        
        df['adx'] = ta.trend.ADXIndicator(
            high=df['target_high'], low=df['target_low'], close=df['target_close'], window=14
        ).adx()
        
        # --- 3. Momentum & Oscillators ---
        for p in [7, 14]:
            df[f'rsi_{p}'] = ta.momentum.RSIIndicator(close=df['target_close'], window=p).rsi()
            
        df['roc_10'] = ta.momentum.ROCIndicator(close=df['target_close'], window=10).roc()
        
        stoch = ta.momentum.StochasticOscillator(
            high=df['target_high'], low=df['target_low'], close=df['target_close'], window=14, smooth_window=3
        )
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # --- 4. Cross-Asset Ratios (ความสัมพันธ์ข้ามสินทรัพย์) ---
        if 'dxy_nyb_close' in df.columns:
            # ดอลลาร์แข็ง ทองมักจะร่วง (ใช้ Correlation ย้อนหลัง 20 แท่ง)
            df['corr_gold_dxy_20'] = df['target_close'].rolling(20).corr(df['dxy_nyb_close'])
            df['dxy_momentum'] = df['dxy_nyb_close'].pct_change(3)
        
        if 'si_f_close' in df.columns:
            # อัตราส่วน ทองคำ/เงิน (Gold/Silver Ratio)
            df['gold_silver_ratio'] = df['target_close'] / df['si_f_close']
        
        # --- 5. Time-based Features ---
        # AI จะได้เรียนรู้พฤติกรรมช่วง London Session vs NY Session
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        
        # --- 6. Price Action (Lag Returns) ---
        for i in [1, 2, 3, 5]:
            df[f'return_lag_{i}'] = df['target_close'].pct_change(periods=i)
            
        return df

    def generate_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        สร้าง Target แบบไม่แอบดูอนาคต (No Look-Ahead Bias)
        เงื่อนไข: ภายใน N แท่งถัดไป ราคาต้องวิ่งเกิน (ATR * 0.5) 
        """
        logger.info(f"สร้าง Target: วิ่งเกิน {self.target_atr_mult} ATR ภายใน {self.forward_bars} แท่งถัดไป")
        
        # หา Max High และ Min Low ในอนาคต N แท่ง
        # ใช้ .rolling(N).max().shift(-N) คือการดึงข้อมูล N แท่งในอนาคตมาไว้ที่แถวปัจจุบัน
        future_highs = df['target_high'].rolling(self.forward_bars).max().shift(-self.forward_bars)
        future_lows = df['target_low'].rolling(self.forward_bars).min().shift(-self.forward_bars)
        
        # ระยะห่างจากราคาปิดปัจจุบัน ไปยังจุดสูงสุด/ต่ำสุดในอนาคต
        max_up_move = future_highs - df['target_close']
        max_down_move = df['target_close'] - future_lows
        
        atr_threshold = df['atr'] * self.target_atr_mult
        
        # บันทึก Target: ถ้าถึงเกณฑ์ = 1, ถ้าไม่ถึง = 0
        df['target_long'] = (max_up_move >= atr_threshold).astype(int)
        df['target_short'] = (max_down_move >= atr_threshold).astype(int)
        
        # คลีนข้อมูล: ตัด N แถวสุดท้ายที่ดึงอนาคตไม่ได้ทิ้ง และตัดแถวแรกๆ ที่ Indicator ยังคำนวณไม่เสร็จ (NaN)
        initial_len = len(df)
        df.dropna(inplace=True)
        final_len = len(df)
        
        logger.debug(f"Drop NaN (ช่วงก่อตัว Indicator & ท้ายตาราง): {initial_len} -> {final_len} rows")
        
        return df