import pandas as pd
import vectorbt as vbt
from loguru import logger
import os

class VectorBacktester:
    def __init__(self, init_cash: float = 10000.0, fees: float = 0.0001, entry_threshold: float = 0.65, exit_threshold: float = 0.5):
        self.init_cash = init_cash
        self.fees = fees # ค่าธรรมเนียม Broker (เช่น 0.01%)
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold

    def run_backtest(self, df: pd.DataFrame, price_col: str = 'target_close', prob_col: str = 'long_signal_prob'):
        logger.info(f"เริ่มจำลองการเทรดด้วยทุนเริ่มต้น ${self.init_cash:,.2f}...")
        
        # 1. สร้างสัญญาณ Entry และ Exit จาก Probability
        entries = df[prob_col] >= self.entry_threshold
        exits = df[prob_col] < self.exit_threshold
        
        # 2. ป้อนข้อมูลเข้าสู่ Portfolio จำลอง
        # ใช้ direction='longonly' เพราะโมเดลเราเทรนมาเพื่อฝั่งขาขึ้นอย่างเดียวในตอนนี้
        portfolio = vbt.Portfolio.from_signals(
            close=df[price_col],
            entries=entries,
            exits=exits,
            fees=self.fees,
            init_cash=self.init_cash,
            freq='15T', # Timeframe 15 นาที
            direction='longonly' 
        )
        
        # 3. สรุปผลสถิติที่ Quant สนใจ
        stats = portfolio.stats()
        
        logger.info("\n==================================================")
        logger.info("📊 สรุปผลประกอบการ (Backtest Statistics) 📊")
        logger.info("==================================================")
        
        # ดึงเฉพาะค่าที่สำคัญมาแสดงผล
        important_metrics = [
            'Start Value', 'End Value', 'Total Return [%]', 
            'Max Drawdown [%]', 'Win Rate [%]', 'Total Trades', 
            'Profit Factor', 'Sharpe Ratio'
        ]
        
        for metric in important_metrics:
            if metric in stats:
                print(f"{metric+':':<20} {stats[metric]}")
                
        # 4. บันทึกกราฟแสดงผล
        os.makedirs('reports', exist_ok=True)
        plot_path = "reports/backtest_result.html"
        portfolio.plot().write_html(plot_path)
        logger.success(f"บันทึกกราฟ Interactive แบ็คเทสต์เรียบร้อยที่: {plot_path}")
        
        return portfolio