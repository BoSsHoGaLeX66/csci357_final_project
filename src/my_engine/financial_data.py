import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import RobustScaler, StandardScaler
import torch
from torch.utils.data import TensorDataset


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd_hist(close: pd.Series) -> pd.Series:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()

    return macd - signal


def download_ohlcv(tickers, start="2018-01-01", end=None) -> pd.DataFrame:
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    return data


def get_ticker_frame(raw_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Handles both single-ticker and multi-ticker yfinance outputs.
    """
    if isinstance(raw_data.columns, pd.MultiIndex):
        df = raw_data[ticker].copy()
    else:
        df = raw_data.copy()

    df.columns = [col.lower() for col in df.columns]
    return df


def build_stock_feature_df(
    target_ticker: str,
    sector_etf: str,
    peer_tickers: list[str],
    start: str = "2018-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """
    Builds a feature dataframe for stock movement prediction.

    Target:
        next_day_log_return

    Features include:
        target stock price action
        volume/liquidity
        technical indicators
        market context
        relative strength
        peer basket features
        regime features
    """
    market_tickers = ["SPY", "QQQ", "^VIX", "^TNX", "DX-Y.NYB"]
    tickers = list(
        dict.fromkeys(
            [target_ticker, sector_etf] + peer_tickers + market_tickers
        )
    )

    raw_data = download_ohlcv(tickers, start=start, end=end)

    target = get_ticker_frame(raw_data, target_ticker)

    df = pd.DataFrame(index=target.index)

    close = target["close"]
    open_ = target["open"]
    high = target["high"]
    low = target["low"]
    volume = target["volume"]

    # -------------------------
    # Target stock price action
    # -------------------------
    df["log_return_1d"] = np.log(close / close.shift(1))
    df["return_5d"] = close.pct_change(5, fill_method=None)
    df["return_20d"] = close.pct_change(20, fill_method=None)

    df["rolling_vol_10d"] = df["log_return_1d"].rolling(10).std()

    df["intraday_range"] = (high - low) / close
    df["gap_return"] = (open_ - close.shift(1)) / close.shift(1)

    # -------------------------
    # Volume / liquidity
    # -------------------------
    df["log_volume"] = np.log1p(volume)
    df["relative_volume_20d"] = volume / volume.rolling(20).mean()
    df["dollar_volume"] = close * volume

    # -------------------------
    # Technical indicators
    # -------------------------
    df["rsi_14"] = rsi(close, window=14)
    df["macd_hist"] = macd_hist(close)

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()

    df["dist_from_ma20"] = close / ma20 - 1
    df["dist_from_ma50"] = close / ma50 - 1

    rolling_mean_20 = close.rolling(20).mean()
    rolling_std_20 = close.rolling(20).std()

    upper_band = rolling_mean_20 + 2 * rolling_std_20
    lower_band = rolling_mean_20 - 2 * rolling_std_20

    df["bollinger_position"] = (
        (close - lower_band) / (upper_band - lower_band)
    )

    # -------------------------
    # Market context
    # -------------------------
    spy = get_ticker_frame(raw_data, "SPY")["close"]
    qqq = get_ticker_frame(raw_data, "QQQ")["close"]
    vix = get_ticker_frame(raw_data, "^VIX")["close"]
    tnx = get_ticker_frame(raw_data, "^TNX")["close"]
    dxy = get_ticker_frame(raw_data, "DX-Y.NYB")["close"]

    df["spy_return_1d"] = np.log(spy / spy.shift(1))
    df["qqq_return_1d"] = np.log(qqq / qqq.shift(1))
    df["vix_return_1d"] = np.log(vix / vix.shift(1))
    df["tnx_change_1d"] = tnx.diff()
    df["dxy_return_1d"] = np.log(dxy / dxy.shift(1))

    # -------------------------
    # Relative strength
    # -------------------------
    sector = get_ticker_frame(raw_data, sector_etf)["close"]
    sector_return = np.log(sector / sector.shift(1))

    df["stock_minus_spy"] = df["log_return_1d"] - df["spy_return_1d"]
    df["stock_minus_sector"] = df["log_return_1d"] - sector_return

    # -------------------------
    # Peer features
    # -------------------------
    peer_returns = []

    for peer in peer_tickers:
        peer_close = get_ticker_frame(raw_data, peer)["close"]
        peer_return = np.log(peer_close / peer_close.shift(1))

        df[f"{peer.lower()}_return_1d"] = peer_return
        peer_returns.append(peer_return)

    if len(peer_returns) > 0:
        peer_return_df = pd.concat(peer_returns, axis=1)
        df["peer_basket_return"] = peer_return_df.mean(axis=1)

        lead_peer = peer_tickers[0]
        df["lead_peer_return"] = df[f"{lead_peer.lower()}_return_1d"]

    # -------------------------
    # Regime features
    # -------------------------
    df["rolling_corr_spy_20d"] = (
        df["log_return_1d"]
        .rolling(20)
        .corr(df["spy_return_1d"])
    )

    df["trend_regime"] = (ma20 > ma50).astype(int)

    # -------------------------
    # Prediction target
    # -------------------------
    df["target_next_log_return"] = df["log_return_1d"].shift(-1)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    return df

def train_val_test_split_time_series(
    df: pd.DataFrame,
    target_col: str = "target_next_log_return",
    train_frac: float = 0.7,
    val_frac: float = 0.15,
):
    n = len(df)

    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    feature_cols = [col for col in df.columns if col != target_col]

    X_train = train_df[feature_cols].values
    y_train = train_df[[target_col]].values

    X_val = val_df[feature_cols].values
    y_val = val_df[[target_col]].values

    X_test = test_df[feature_cols].values
    y_test = test_df[[target_col]].values

    x_scaler = RobustScaler()
    y_scaler = StandardScaler()

    X_train = x_scaler.fit_transform(X_train)
    X_val = x_scaler.transform(X_val)
    X_test = x_scaler.transform(X_test)

    y_train = y_scaler.fit_transform(y_train)
    y_val = y_scaler.transform(y_val)
    y_test = y_scaler.transform(y_test)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "feature_cols": feature_cols,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
    }

def make_sequences(X, y, sequence_length: int = 30, horizon: int = 1):
    X_seq = []
    y_seq = []

    for i in range(len(X) - sequence_length - horizon + 1):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y[i+sequence_length + horizon - 1])

    X_seq = torch.tensor(np.array(X_seq), dtype=torch.float32)
    y_seq = torch.tensor(np.array(y_seq), dtype=torch.float32)

    return X_seq, y_seq

def make_single_stock_df(ticker: str, period:str = "5y", train_split: float = 0.8, val_split:float = 0.1, window_size:int = 30):
    df = yf.download(ticker, period=period)
    df['log_returns'] = np.log(df['Close'].shift(1) / df['Close'])
    df['log_volume'] = np.log(df['Volume'])
    df['log_intraday_chng'] = np.log(df['High'] / df['Low'])
    df['log_variance'] = df['log_intraday_chng'].rolling(5).mean()
    df['10_log_returns_ma'] = df['log_returns'].rolling(10).mean()
    df['20_log_returns_ma'] = df['log_returns'].rolling(20).mean()
    df['50_log_returns_ma'] = df['log_returns'].rolling(50).mean()

    df.dropna(inplace=True)

    val_idx = int(len(df) * train_split)
    test_idx = None
    if train_split + val_split < 1:
        test_idx = int(len(df) * val_split) + val_idx

    train_df = df.iloc[:val_idx]
    val_df = df.iloc[val_idx:]
    test_df = None
    if test_idx:
        test_df = df.iloc[test_idx:]

    features = ['log_returns', 'log_volume', 'log_intraday_chng', 'log_variance', '10_log_returns_ma',
                '20_log_returns_ma', '50_log_returns_ma']
    scaler = StandardScaler()

    train_scaled = pd.DataFrame(scaler.fit_transform(train_df[features]), columns=train_df[features].columns)
    val_scaled = pd.DataFrame(scaler.transform(val_df[features]), columns=val_df[features].columns)
    test_scaled = pd.DataFrame(scaler.transform(test_df[features]), columns=test_df[features].columns)

    def generate_sequence(scaled_df: pd.DataFrame):
        target = 'log_returns'
        X = []
        y = []
        for i in range(scaled_df.shape[0] - window_size - 1):
            data = scaled_df.iloc[i:i + window_size][features]
            y_temp = scaled_df.iloc[i + window_size][target]
            X.append(data)
            y.append(y_temp)
        X = np.array(X)
        y = np.array(y)
        return TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))

    train_ds = generate_sequence(train_scaled)
    val_ds = generate_sequence(val_scaled)
    test_ds = None
    if test_idx:
        test_ds = generate_sequence(test_scaled)

    return train_ds, val_ds, test_ds, scaler