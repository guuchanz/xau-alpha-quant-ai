import pandas as pd
import vectorbt as vbt
from loguru import logger
import os

class VectorBacktester:
    def __init__(self, init_cash: float = 10000.0, fees: float = 0.0002, 
                 entry_threshold: float = 0.66, exit_threshold: float = 0.50,
                 sl_pct: float = 0.03, tp_pct: float = 0.01):
        """
        เพิ่มระบบควบคุมความเสี่ยง: sl_pct (Stop Loss) และ tp_pct (Take Profit)
        """
        self.init_cash = init_cash
        self.fees = fees
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.sl_pct = sl_pct  # เช่น 0.03 = 3%
        self.tp_pct = tp_pct  # เช่น 0.01 = 1%

    def run_bidirectional_backtest(self, df: pd.DataFrame):
        logger.info(f"🎬 เริ่มจำลองการเทรดสองฝั่ง + ระบบ Risk Management (SL: {self.sl_pct*100}%, TP: {self.tp_pct*100}%)...")
        
        if 'long_signal_prob' not in df.columns or 'short_signal_prob' not in df.columns:
            raise KeyError("ข้อมูลไม่สมบูรณ์! ไม่พบคอลัมน์ 'long_signal_prob' หรือ 'short_signal_prob'")
            
        # สัญญาณฝั่ง Long 
        entries = df['long_signal_prob'] >= self.entry_threshold
        exits = df['long_signal_prob'] < self.exit_threshold
        
        # สัญญาณฝั่ง Short 
        short_entries = df['short_signal_prob'] >= self.entry_threshold
        short_exits = df['short_signal_prob'] < self.exit_threshold
        
        # ประมวลผลโดยเพิ่มเงื่อนไข sl_stop และ tp_stop เข้าไปควบคุมพอร์ต
        portfolio = vbt.Portfolio.from_signals(
            close=df['target_close'],
            entries=entries,
            exits=exits,
            short_entries=short_entries, 
            short_exits=short_exits,     
            fees=self.fees,
            init_cash=self.init_cash,
            freq='15T',
            direction='both',
            sl_stop=self.sl_pct,  # <--- เปิดใช้งานระบบหยุดขาดทุนอัตโนมัติ
            tp_stop=self.tp_pct   # <--- เปิดใช้งานระบบล็อกกำไรอัตโนมัติ
        )
        
        stats = portfolio.stats()
        
        logger.info("\n==================================================")
        logger.info("🛡️ ผลประกอบการหลังติดตั้งระบบควบคุมความเสี่ยง (SL/TP) 🛡️")
        logger.info("==================================================")
        
        metrics_to_show = ['Start Value', 'End Value', 'Total Return [%]', 'Max Drawdown [%]', 'Total Trades', 'Win Rate [%]']
        for metric in metrics_to_show:
            if metric in stats:
                print(f"{metric+':':<20} {stats[metric]}")
            
        os.makedirs('reports', exist_ok=True)
        report_path = "reports/bidirectional_backtest_rm.html"
        portfolio.plot().write_html(report_path)
        logger.success(f"บันทึกกราฟกลยุทธ์ Risk Management เรียบร้อยที่: report_path")
        
        return portfolio