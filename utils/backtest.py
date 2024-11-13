import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from scipy.stats import gmean

def choose_position(roi, trade_threshold=0):
    # Hàm đơn giản để xác định tín hiệu mua/bán
    if roi > trade_threshold:
        return 1  # Tín hiệu mua
    elif roi < -trade_threshold:
        return -1  # Tín hiệu bán
    else:
        return 0  # Không mở vị thế
        
class ModelBasedStrategy(Strategy):
    trade_threshold = 0
    feat = []  # Khai báo biến lớp feat
    model = None  # Khai báo biến lớp model
    point_value = 1  # Giá trị mỗi điểm
    forecast_steps = 3  # Số bước dự báo
    take_profit = 0.02  # Ngưỡng chốt lời (2%)
    stop_loss = 0.02  # Ngưỡng cắt lỗ (2%)
    trailing_stop_pct = 0.02  # Trailing stop 2% (2% dưới mức giá cao nhất đạt được)

    def init(self):
        # Lưu trữ dự báo tín hiệu mua/bán từ mô hình
        self.pred = self.I(self.predict_model, self.data.Close)
        self.pos = self.calculate_positions(self.pred)
        self.i = 0
        self.entry_price = None  # Giá vào lệnh
        self.highest_price = None  # Lưu trữ giá cao nhất đạt được trong quá trình giữ vị thế

    def next(self):
        if self.i < len(self.pos):
            current_pos = self.pos[self.i]
            self.i += 1
            current_price = self.data.Close[-1] * self.point_value

            # Kiểm tra nếu không có vị thế mở
            if not self.position:
                if current_pos == 1:
                    self.buy(size=0.4)
                    self.entry_price = current_price
                    self.highest_price = current_price  # Thiết lập giá cao nhất ban đầu
                elif current_pos == -1:
                    self.sell(size=0.4)
                    self.entry_price = current_price
                    self.highest_price = current_price  # Thiết lập giá cao nhất ban đầu

            # Nếu đã có vị thế mở, kiểm tra chốt lời/cắt lỗ và trailing stop
            else:
                # Cập nhật giá cao nhất (hoặc thấp nhất nếu bán) kể từ khi mở vị thế
                if self.position.is_long:
                    self.highest_price = max(self.highest_price, current_price)
                else:
                    self.highest_price = min(self.highest_price, current_price)

                # Tính tỷ suất sinh lời hiện tại
                if self.position.is_long:
                    tssl = (current_price - self.entry_price) / self.entry_price
                else:
                    tssl = (self.entry_price - current_price) / self.entry_price

                # Đóng vị thế nếu đạt mức TP hoặc SL
                if tssl >= self.take_profit or tssl <= -self.stop_loss:
                    self.position.close()
                    self.entry_price = None
                    self.highest_price = None
                    return

                # Tính trailing stop dựa trên giá cao nhất đạt được
                trailing_stop = self.highest_price * (1 - self.trailing_stop_pct) if self.position.is_long else self.highest_price * (1 + self.trailing_stop_pct)
# Đóng vị thế nếu giá đi ngược lại mức trailing stop
                if (self.position.is_long and current_price <= trailing_stop) or (self.position.is_short and current_price >= trailing_stop):
                    self.position.close()
                    self.entry_price = None
                    self.highest_price = None
                    return

                # Lấy dự báo cho các bước tiếp theo
                future_positions = self.pos[self.i:self.i + self.forecast_steps]

                # Kiểm tra tín hiệu đảo chiều trong 2 bước tới (thay vì 1 bước)
                if self.check_reversal(future_positions, current_pos):
                    self.position.close()  # Đóng vị thế hiện tại
                    if current_pos == 1:
                        self.buy(size=0.4)
                    elif current_pos == -1:
                        self.sell(size=0.4)
                    self.entry_price = current_price
                    self.highest_price = current_price  # Thiết lập giá cao nhất mới cho vị thế tiếp theo

    def predict_model(self, price):
        # Dự báo từ mô hình đã cung cấp dựa trên các đặc trưng hiện tại
        test_data = self.data.df[self.feat]  # Sử dụng feat trong predict_model
        pred = self.model.predict(test_data)  # Sử dụng model để dự báo
        return pred

    def calculate_positions(self, predictions):
        # Xác định tín hiệu mua/bán dựa trên ngưỡng dự đoán
        return [choose_position(roi, self.trade_threshold) for roi in predictions]

    def check_reversal(self, future_positions, current_pos):
        """
        Kiểm tra tín hiệu đảo chiều:
        - Đảm bảo tín hiệu thay đổi trong 2 bước liên tiếp mới đảo chiều.
        """
        # Nếu hiện tại đang mua mà có tín hiệu bán trong 2 bước tiếp theo (hoặc ngược lại)
        if current_pos == 1 and -1 in future_positions[:2]:  # Kiểm tra 2 bước tiếp theo
            return True
        elif current_pos == -1 and 1 in future_positions[:2]:  # Kiểm tra 2 bước tiếp theo
            return True
        return False
    
class CustomBacktest(Backtest):
    def __init__(self, *args, cash=1000, margin=0.13, **kwargs):
        super().__init__(*args, cash=cash, margin=margin, **kwargs)
        self._initial_cash = cash
        self.margin = margin  # Tỷ lệ margin

    def calculate_total_fees(self, entry_price, exit_price, size, entry_day, exit_day):
        platform_fee = 2000 * size
        exchange_fee = 2700 * size
        overnight_days = max(0, (exit_day - entry_day).days)
        overnight_fee = 2000 * overnight_days * size
        personal_income_tax = (exit_price * 100000 * self.margin) * 0.1 / 100 / 2 * size
        total_fee = (platform_fee + exchange_fee + overnight_fee + personal_income_tax) / 100000
        return total_fee

    def geometric_mean(self, returns: pd.Series) -> float:
        returns = returns.fillna(0) + 1
        if np.any(returns <= 0):
            return 0
        return np.exp(np.log(returns).sum() / (len(returns) or np.nan)) - 1

    def run(self, **strategy_args):
        stats = super().run(**strategy_args)
        fees_for_trades = []
        margin_required_list = []
        current_equity_list = []

        current_equity = self._initial_cash

        for idx, trade in stats._trades.iterrows():
            entry_price = trade['EntryPrice']
            exit_price = trade['ExitPrice']
            size = abs(trade['Size'])
            entry_day = trade['EntryTime'].date()
            exit_day = trade['ExitTime'].date()

            # Tính ký quỹ cần thiết cho giao dịch
            margin_required = entry_price * self.margin
            margin_required_list.append(margin_required)

            # Kiểm tra nếu equity >= margin_required
            if current_equity < margin_required:
                # print(f"Không đủ ký quỹ cho giao dịch tại {idx}. Equity hiện tại: {current_equity}, Ký quỹ yêu cầu: {margin_required}")
                continue

            # Tính phí cho giao dịch này
            total_fee = self.calculate_total_fees(entry_price, exit_price, size, entry_day, exit_day)
            fees_for_trades.append(total_fee)

            # Cập nhật vốn hiện tại sau khi trừ phí và tính lãi/lỗ của giao dịch
            pnl_after_fees = trade['PnL'] - total_fee
            current_equity += pnl_after_fees
            current_equity_list.append(current_equity)

            # Cập nhật PnL sau khi trừ phí
            stats._trades.at[idx, 'PnL_after_fees'] = pnl_after_fees

        # Thêm cột 'Fees', 'Margin Required', và 'Current Equity' vào _trades và tổng phí đã áp dụng
        if len(fees_for_trades) < len(stats._trades.index):
            # Extend fees_for_trades with zeros
            fees_for_trades += [0] * (len(stats._trades.index) - len(fees_for_trades))
        elif len(fees_for_trades) > len(stats._trades.index):
            # Trim fees_for_trades if it's too long
            fees_for_trades = fees_for_trades[:len(stats._trades.index)]
        
        # Repeat the same for margin_required_list and current_equity_list
        if len(margin_required_list) < len(stats._trades.index):
            margin_required_list += [0] * (len(stats._trades.index) - len(margin_required_list))
        elif len(margin_required_list) > len(stats._trades.index):
            margin_required_list = margin_required_list[:len(stats._trades.index)]
        
        if len(current_equity_list) < len(stats._trades.index):
            current_equity_list += [current_equity] * (len(stats._trades.index) - len(current_equity_list))
        elif len(current_equity_list) > len(stats._trades.index):
            current_equity_list = current_equity_list[:len(stats._trades.index)]
        stats._trades['Fees'] = fees_for_trades
        stats._trades['Margin Required'] = margin_required_list
        stats._trades['Current Equity'] = current_equity_list
        stats['Custom Total Fees'] = sum(fees_for_trades)

        # Cập nhật vốn cuối cùng sau khi trừ phí
        final_cash = stats['Equity Final [$]'] - stats['Custom Total Fees']
        stats['Equity Final [$]'] = final_cash
        stats['Return [%]'] = ((final_cash - self._initial_cash) / self._initial_cash) * 100

        # Tính toán Return (Ann.) [%] và Volatility (Ann.) [%] sử dụng công thức chuẩn hóa
        equity_df = stats._equity_curve
        index = equity_df.index
        day_returns = equity_df['Equity'].resample('D').last().dropna().pct_change().dropna()

        if len(day_returns) > 0:
            gmean_day_return = self.geometric_mean(day_returns)
            annual_trading_days = 365 if index.dayofweek.to_series().between(5, 6).mean() > 2/7 * .6 else 252
            annualized_return = (1 + gmean_day_return) ** annual_trading_days - 1
            stats['Return (Ann.) [%]'] = annualized_return * 100

            # Áp dụng công thức chuẩn cho Volatility (Ann.) [%]
            daily_var = day_returns.var(ddof=1)
            stats['Volatility (Ann.) [%]'] = np.sqrt(daily_var * annual_trading_days) * 100
        else:
            stats['Return (Ann.) [%]'] = np.nan
            stats['Volatility (Ann.) [%]'] = np.nan

        self._update_metrics(stats)
        return stats

    def _update_metrics(self, stats):
        equity_df = stats._equity_curve
        index = equity_df.index
        day_returns = equity_df['Equity'].resample('D').last().dropna().pct_change().dropna()

        if len(day_returns) > 0:
            gmean_day_return = gmean(1 + day_returns) - 1
            annual_trading_days = 365 if index.dayofweek.to_series().between(5, 6).mean() > 2/7 * .6 else 252
            annualized_return = (1 + gmean_day_return) ** annual_trading_days - 1
            stats['Return (Ann.) [%]'] = annualized_return * 100

            # Volatility (Ann.) based on standard formula
            daily_var = day_returns.var(ddof=1)
            stats['Volatility (Ann.) [%]'] = np.sqrt(daily_var * annual_trading_days) * 100
        else:
            stats['Return (Ann.) [%]'] = np.nan
            stats['Volatility (Ann.) [%]'] = np.nan


        # Sharpe Ratio
        risk_free_rate = 0
        stats['Sharpe Ratio'] = (
            (stats['Return (Ann.) [%]'] - risk_free_rate) / stats['Volatility (Ann.) [%]']
            if stats['Volatility (Ann.) [%]'] > 0 else np.nan
        )

        # Sortino Ratio
        downside_risk = day_returns[day_returns < 0].std() * np.sqrt(annual_trading_days) * 100
        stats['Sortino Ratio'] = (stats['Return (Ann.) [%]'] - risk_free_rate) / downside_risk if downside_risk > 0 else np.nan

        # Calmar Ratio
        equity_curve = stats._equity_curve['Equity']
        max_drawdown = (equity_curve.cummax() - equity_curve).max()
        stats['Calmar Ratio'] = stats['Return (Ann.) [%]'] / max_drawdown if max_drawdown > 0 else np.nan

        # Win Rate
        stats['Win Rate [%]'] = (stats._trades['PnL_after_fees'] > 0).mean() * 100 if not stats._trades.empty else 0
        # Các chỉ số giao dịch khác
        stats['Best Trade [%]'] = stats._trades['ReturnPct'].max() * 100 if not stats._trades.empty else 0
        stats['Worst Trade [%]'] = stats._trades['ReturnPct'].min() * 100 if not stats._trades.empty else 0
        stats['Avg. Trade [%]'] = stats._trades['ReturnPct'].mean() * 100 if not stats._trades.empty else 0

        try:
            stats['Profit Factor'] = stats._trades[stats._trades['PnL_after_fees'] > 0]['PnL_after_fees'].sum() / abs(stats._trades[stats._trades['PnL_after_fees'] < 0]['PnL_after_fees'].sum())
        except:
            stats['Profit Factor'] = 0

        stats['Expectancy [%]'] = stats._trades['ReturnPct'].mean() * 100 if not stats._trades.empty else 0

        try:
            stats['SQN'] = np.sqrt(len(stats._trades)) * stats._trades['PnL_after_fees'].mean() / stats._trades['PnL_after_fees'].std()
        except:
            stats['SQN'] = 0

def run_model_backtest(df, selected_features, model):
    # Chạy backtest với CustomBacktest
    bt = CustomBacktest(df, ModelBasedStrategy, cash=1000, commission=0, margin=0.13, hedging=True, exclusive_orders=False)
    stats = bt.run(feat=selected_features, model=model)
    return stats

