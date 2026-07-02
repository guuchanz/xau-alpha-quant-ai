import pandas as pd
import vectorbt as vbt
import itertools
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

class StrategyOptimizer:
    def __init__(self, init_cash: float = 10000.0, fees: float = 0.0002):
        self.init_cash = init_cash
        self.fees = fees

    def run_sweep(self, df: pd.DataFrame, price_col: str = 'target_close', prob_col: str = 'long_signal_prob'):
        logger.info("เริ่มกระบวนการ Grid Search Optimization...")
        
        # 1. กำหนดช่วงพารามิเตอร์ที่จะนำมาผสมกัน
        # AI ต้องมั่นใจกี่ % ถึงจะยอมเข้าซื้อ?
        entry_thresholds = [0.66, 0.70, 0.75] 
        
        # ยอมขาดทุนได้กี่ % ต่อไม้? (0.5%, 1%, 1.5%)
        sl_stops = [0.01, 0.02, 0.03] 
        
        # เอากำไรที่กี่ % ? (1%, 2%, 3%)
        tp_stops = [0.03, 0.05, 0.07] 
        
        # นำมาสับเปลี่ยนคู่กัน (3 x 3 x 3 = 27 จักรวาลคู่ขนาน)
        combinations = list(itertools.product(entry_thresholds, sl_stops, tp_stops))
        logger.info(f"กำลังจำลองการเทรดทั้งหมด {len(combinations)} รูปแบบ...")
        
        best_return = -999.0
        best_params = None
        best_portfolio = None
        results = []
        
        # 2. จำลองการเทรดทีละรูปแบบ
        for entry, sl, tp in combinations:
            entries = df[prob_col] >= entry
            # สังเกตว่าเราไม่ใช้ exit_threshold แล้ว แต่จะให้ SL/TP ทำงานแทน!
            
            portfolio = vbt.Portfolio.from_signals(
                close=df[price_col],
                entries=entries,
                sl_stop=sl,      # <--- เกราะป้องกันพอร์ต
                tp_stop=tp,      # <--- หุ่นยนต์เก็บกำไรอัตโนมัติ
                fees=self.fees,
                init_cash=self.init_cash,
                freq='15T',
                direction='longonly'
            )
            
            # ดึงสถิติ
            ret_pct = portfolio.total_return() * 100
            win_rate = portfolio.trades.win_rate() * 100
            pf = portfolio.trades.profit_factor()
            
            results.append({
                'Entry >': entry, 
                'SL [%]': sl * 100, 
                'TP [%]': tp * 100,
                'Return [%]': ret_pct,
                'Win Rate [%]': win_rate if not pd.isna(win_rate) else 0,
                'Profit Factor': pf if not pd.isna(pf) else 0
            })
            
            # เก็บอันดับ 1 ไว้
            if ret_pct > best_return:
                best_return = ret_pct
                best_params = (entry, sl, tp)
                best_portfolio = portfolio
                
        # 3. สรุปผลลัพธ์
        res_df = pd.DataFrame(results).sort_values(by='Return [%]', ascending=False).round(2)
        
        logger.success(f"🎯 พบ Best Parameters: Entry > {best_params[0]}, SL = {best_params[1]*100}%, TP = {best_params[2]*100}%")
        
        return res_df, best_portfolio