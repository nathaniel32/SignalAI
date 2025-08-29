import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn import preprocessing
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import mplfinance as mpf
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
from collections import Counter
import config

def plot_dataset_all(df):
    n = len(df.columns)
    figsize = (18, 2*n)
    df.plot(subplots=True, figsize=figsize, title="Dataset")
    plt.tight_layout()
    plt.savefig("data/dataset.png")
    plt.close()

def plot_dataset_chart(df, filename="data/chart.png"):
    df_plot = df.copy()
    colors = {"BUY": "green", "SELL": "red", "HOLD": "blue"}

    # Data preparation
    if not isinstance(df_plot.index, pd.DatetimeIndex):
        df_plot.index = pd.to_datetime(df_plot.index)
    df_plot.index = df_plot.index.tz_localize(None)
    df_plot = df_plot.sort_index()

    df_plot.rename(columns={
        'open': 'Open', 
        'high': 'High',
        'low': 'Low', 
        'close': 'Close'
    }, inplace=True)

    df_plot = df_plot.dropna(subset=['Open','High','Low','Close'])

    # figure
    fig = plt.figure(figsize=(26, 12))
    
    # SUBPLOT 1: CANDLESTICK CHART
    ax1 = plt.subplot(2, 1, 1)
    
    # Market colors
    mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc)
    
    # Plot candlestick
    mpf.plot(
        df_plot[['Open','High','Low','Close']],
        type='candle',
        style=s,
        ax=ax1,
        ylabel='Price',
        warn_too_much_data=len(df_plot)+1
    )
    
    ax1.set_title("Candlestick Chart", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # SUBPLOT 2: SCATTER PLOT SIGNALS
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    
    # Plot close price
    positions = np.arange(len(df_plot))
    ax2.plot(positions, df_plot['Close'].values, color='gray', alpha=0.3, linewidth=1, label='Close Price')
    
    # Scatter plot
    for label, color in colors.items():
        mask = df_plot["Label"] == label
        positions_label = np.where(mask)[0]
        prices_label = df_plot.loc[mask, "Close"].values
        
        ax2.scatter(positions_label, prices_label, label=f'{label} Signal', color=color, s=60, marker='o', alpha=0.8, edgecolors='black', linewidth=0.5)
    
    ax2.set_title("Trading Signals", fontsize=14, fontweight='bold')
    ax2.set_ylabel('Price')
    ax2.set_xlabel('Time Index')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Format dan styling
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def show_conf_matrix(preds_array, solution_array, label, title, binary=False, plot=False):
    def calculate_accuracy(preds_array, solution_array):
        preds_tensor = torch.tensor(preds_array)
        solution_tensor = torch.tensor(solution_array)
        accuracy = (preds_tensor == solution_tensor).sum().item() / len(solution_tensor)
        return round(accuracy * 100, 1)

    if len(label) > 1:
        title = title + "\nAccuracy: " + str(calculate_accuracy(preds_array, solution_array)) + "%"
        print("\nConfusion matrix " + title)
        
        if binary:
            confmat_metric = ConfusionMatrix(task='binary', num_classes=2)
        else:
            confmat_metric = ConfusionMatrix(task='multiclass', num_classes=len(label))
            
        conf_matrix = confmat_metric(torch.tensor(preds_array), torch.tensor(solution_array))
        print(conf_matrix)

        if plot:
            conf_matrix_np = conf_matrix.cpu().numpy()
            fig, ax = plot_confusion_matrix(conf_mat=conf_matrix_np, class_names=label, figsize=(8, 8), cmap="Blues")
            plt.tight_layout(pad=3.0)
            plt.title(title)
            plt.xlabel("Prediction")
            plt.ylabel("Solution")
            plt.savefig("data/cmatrix.png")
            plt.close()

        return conf_matrix
    else:
        return None

def to_yhat(logits):
    logits = logits.view(-1, logits.shape[-1]).cpu().detach()
    probs = torch.softmax(logits, dim=1)
    y_hat = torch.argmax(probs, dim=1)
    return probs.numpy(), y_hat.numpy()

class DatasetManager(torch.utils.data.Dataset):
    def __init__(self, sequences, market_ids, periods, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.market_ids = torch.LongTensor(market_ids)
        self.periods = torch.LongTensor(periods)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'sequence': self.sequences[idx],
            'market_id': self.market_ids[idx],
            'period': self.periods[idx],
            'label': self.labels[idx]
        }

def print_table_info(df, title):
    print("="*50)
    print(title)
    print(df)
    
    if 'Label' in df.columns:
        label_counts = df['Label'].value_counts()
        print("\nLabel Counts")
        for label, count in label_counts.items():
            print(f"- {label}: {count}")
    
    print("="*50)

def create_labels(df, hold=True):
    df = df.copy()
    # Label
    # Geser harga close ke depan
    df["Future_Close"] = df["close"].shift(-config.HORIZON_LABEL)

    # Hitung return masa depan
    df["Future_Return"] = (df["Future_Close"] - df["close"]) / df["close"]

    # Buat label berdasarkan aturan
    def create_label(x):
        if x > config.THRESHOLD_LABEL:
            return "BUY"
        elif x < -config.THRESHOLD_LABEL:
            return "SELL"
        else:
            return "HOLD" if hold else None

    # hapus label yg nan
    df.dropna(subset=['Future_Return'], axis=0, inplace=True)

    # dapatkan signal
    df["Label"] = df["Future_Return"].apply(create_label)
    
    df.drop(columns=["Future_Close"], inplace=True)
    df.drop(columns=["Future_Return"], inplace=True)
    return df

def create_indicators(df):
    """
    Create advanced technical indicators for trading analysis.
    Returns the original user features and new AI-derived features.
    """
    # Make an explicit copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # ===== PRICE CHANGE FEATURES (Percentage based) =====
    
    # Hourly price change percentage
    df["Price_Change_Pct"] = (df["close"] - df["open"]) / df["open"] * 100
    
    # Price change from previous period
    df["Price_Change_Prev_Pct"] = df["close"].pct_change() * 100
    
    # High-Low range as percentage of close
    df["HL_Range_Pct"] = (df["high"] - df["low"]) / df["close"] * 100
    
    
    # ===== MOVING AVERAGES (Relative to current price) =====
    
    # SMA deviation from current price
    df["SMA_5"] = df["close"].rolling(window=5).mean()
    df["SMA_5_Deviation_Pct"] = (df["close"] - df["SMA_5"]) / df["SMA_5"] * 100
    
    df["SMA_10"] = df["close"].rolling(window=10).mean()
    df["SMA_10_Deviation_Pct"] = (df["close"] - df["SMA_10"]) / df["SMA_10"] * 100
    
    # EMA deviation from current price
    df["EMA_5"] = df["close"].ewm(span=5, adjust=False).mean()
    df["EMA_5_Deviation_Pct"] = (df["close"] - df["EMA_5"]) / df["EMA_5"] * 100
    
    df["EMA_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["EMA_12_Deviation_Pct"] = (df["close"] - df["EMA_12"]) / df["EMA_12"] * 100
    
    
    # ===== MOMENTUM INDICATORS (Already normalized 0-100) =====
    
    # RSI (already 0-100 scale)
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    roll_up = gain.rolling(14).mean()
    roll_down = loss.rolling(14).mean()
    rs = roll_up / roll_down
    df["RSI_14"] = 100 - (100 / (1 + rs))
    
    # RSI normalized to -1 to 1 range
    df["RSI_Normalized"] = (df["RSI_14"] - 50) / 50
    
    
    # ===== MACD (Normalized) =====
    
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    
    # MACD as percentage of price
    df["MACD_Pct"] = df["MACD"] / df["close"] * 100
    df["Signal_Pct"] = df["Signal"] / df["close"] * 100
    df["MACD_Histogram_Pct"] = (df["MACD"] - df["Signal"]) / df["close"] * 100
    
    
    # ===== VOLATILITY INDICATORS (Normalized) =====
    
    # ATR as percentage of price
    df["H-L"] = df["high"] - df["low"]
    df["H-C"] = abs(df["high"] - df["close"].shift())
    df["L-C"] = abs(df["low"] - df["close"].shift())
    df["TR"] = df[["H-L", "H-C", "L-C"]].max(axis=1)
    df["ATR_14"] = df["TR"].rolling(14).mean()
    df["ATR_Pct"] = df["ATR_14"] / df["close"] * 100
    
    
    # ===== BOLLINGER BANDS (Position based) =====
    
    df["SMA_20"] = df["close"].rolling(window=20).mean()
    df["STDDEV_20"] = df["close"].rolling(window=20).std()
    df["UpperBB"] = df["SMA_20"] + (df["STDDEV_20"] * 2)
    df["LowerBB"] = df["SMA_20"] - (df["STDDEV_20"] * 2)
    
    # Bollinger Band position (0 = at lower band, 1 = at upper band)
    df["BB_Position"] = (df["close"] - df["LowerBB"]) / (df["UpperBB"] - df["LowerBB"])
    
    # Distance from bands as percentage
    df["BB_Upper_Distance_Pct"] = (df["UpperBB"] - df["close"]) / df["close"] * 100
    df["BB_Lower_Distance_Pct"] = (df["close"] - df["LowerBB"]) / df["close"] * 100
    
    
    # ===== VOLUME INDICATORS =====
    
    if "volume" in df.columns:
        # Volume change percentage
        df["Volume_Change_Pct"] = np.where(df["volume"] == 0, 0, df["volume"].pct_change() * 100)
        
        # Volume moving average deviation
        df["Volume_MA_5"] = df["volume"].rolling(5).mean()
        df["Volume_MA_Deviation_Pct"] = np.where(df["volume"] == 0, 0, (df["volume"] - df["Volume_MA_5"]) / df["Volume_MA_5"] * 100)
        df["Volume_Flag"] = np.where(df["volume"] == 0, 0, 1)
    
    
    # ===== TREND STRENGTH INDICATORS =====
    
    # Price momentum over different periods
    df["Momentum_3"] = (df["close"] - df["close"].shift(3)) / df["close"].shift(3) * 100
    df["Momentum_5"] = (df["close"] - df["close"].shift(5)) / df["close"].shift(5) * 100
    df["Momentum_10"] = (df["close"] - df["close"].shift(10)) / df["close"].shift(10) * 100
    
    # Trend consistency (how many of last N periods were up/down)
    df["Up_Periods_5"] = (df["close"].diff() > 0).rolling(5).sum() / 5
    df["Down_Periods_5"] = (df["close"].diff() < 0).rolling(5).sum() / 5
    
    
    # ===== PATTERN RECOGNITION FEATURES =====

    # Doji pattern (open close to close)
    df["Doji_Pattern"] = abs(df["close"] - df["open"]) / (df["high"] - df["low"])
    
    # Hammer/Shooting star patterns
    body_size = abs(df["close"] - df["open"])
    upper_shadow = df["high"] - df[["close", "open"]].max(axis=1)
    lower_shadow = df[["close", "open"]].min(axis=1) - df["low"]
    
    df["Upper_Shadow_Ratio"] = upper_shadow / body_size
    df["Lower_Shadow_Ratio"] = lower_shadow / body_size
    
    # ===== User features =====
    user_features = [
        "open", 
        "high", 
        "low", 
        "close", 
        "RSI_14", 
        "H-L", 
        "H-C", 
        "L-C", 
        "TR", 
        "SMA_5", 
        "SMA_10", 
        "SMA_20", 
        "EMA_5", 
        "EMA_12", 
        "STDDEV_20", 
        "UpperBB", 
        "LowerBB", 
        "MACD", 
        "Signal", 
        "ATR_14"
    ]
    
    if "volume" in df.columns:
        user_features.append("Volume_MA_5")
    
    # ===== AI features =====
    ai_features = [
        "Price_Change_Pct",           # Hourly price change
        "Price_Change_Prev_Pct",      # Previous period change
        "HL_Range_Pct",               # Volatility measure
        "SMA_5_Deviation_Pct",        # Short-term trend
        "SMA_10_Deviation_Pct",       # Medium-term trend
        "EMA_5_Deviation_Pct",        # Responsive trend
        "RSI_Normalized",             # Momentum (-1 to 1)
        "MACD_Histogram_Pct",         # MACD signal
        "ATR_Pct",                    # Volatility
        "BB_Position",                # Bollinger band position
        "Momentum_3",                 # Short momentum
        "Momentum_5",                 # Medium momentum
        "Up_Periods_5",               # Trend consistency
        "Doji_Pattern",               # Pattern recognition
        "Signal_Pct",
        "Down_Periods_5",
        "Momentum_10",
        "MACD_Pct",
        "BB_Upper_Distance_Pct",
        "BB_Lower_Distance_Pct",
        "Upper_Shadow_Ratio",
        "Lower_Shadow_Ratio"
    ]
    
    # Volume features
    if "volume" in df.columns:
        ai_features.extend(["Volume_Change_Pct", "Volume_MA_Deviation_Pct", "Volume_Flag"])

    return df, user_features, ai_features

def create_sequences(df, sequence_length=config.SEQUENCE_CANDLE_LENGTH):
    X_sequences = []
    X_market_ids = []
    X_periods = []
    Y_labels = []
    
    # Group by market_id dan period
    for (market_id, period), group in df.groupby(['market_id', 'period']):
        group = group.sort_index()  # Sort by timestamp
        print_table_info(group, "Dataset")
        
        if len(group) < sequence_length + 1:
            continue
        
        for i in range(len(group) - sequence_length):
            # Get n candles sequence
            group_candle_sequence = group.iloc[i:i+sequence_length] #group[i:i+sequence_length]
            
            # Add Indicators
            group_candle_sequence_indicator, user_features, ai_features = create_indicators(df=group_candle_sequence)
            
            # Add Labels
            group_candle_sequence_indicator = create_labels(df=group_candle_sequence_indicator)

            # Clean Data
            group_candle_sequence_indicator = group_candle_sequence_indicator.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

            #sequence = group_candle_sequence[config.PRICE_COLUMNS].values
            sequence = group_candle_sequence_indicator[ai_features].values
            
            # Percentage change normalization
            #normalized_sequence = (sequence / sequence[0]) - 1
            #X_sequences.append(normalized_sequence)
            market_id = group_candle_sequence_indicator['market_id'].tail(1).item()
            period = group_candle_sequence_indicator['period'].tail(1).item()

            X_sequences.append(sequence)
            X_market_ids.append(market_id)
            X_periods.append(period)
            
            # Target Label
            target_label = group_candle_sequence_indicator['Label'].tail(1).item()
            Y_labels.append(target_label)

            #print(sequence.shape)
            #print_table_info(df=group_candle_sequence_indicator, title=f"Signal: {target_label}\nMarket ID: {market_id}\nPeriod: {period}")
    
    max_rows = max(arr.shape[0] for arr in X_sequences)
    
    X_sequences_padding_np = np.array([
        np.pad(arr, ((max_rows - arr.shape[0], 0), (0, 0)), mode='constant', constant_values=0)
        for arr in X_sequences
    ])

    X_market_ids_np = np.array(X_market_ids)
    X_periods_np = np.array(X_periods)
    Y_labels_np = np.array(Y_labels)

    print(f"\nData shapes:")
    print(f"- X_sequences: {X_sequences_padding_np.shape}")
    print(f"- X_market_ids: {X_market_ids_np.shape}")  
    print(f"- X_periods: {X_periods_np.shape}")
    print(f"- Y_labels: {Y_labels_np.shape}")
    
    print(f"\nLabel distribution:")
    unique, counts = np.unique(Y_labels_np, return_counts=True)
    for signal, count in zip(unique, counts):
        print(f"{signal}: {count} ({count/len(Y_labels_np)*100:.1f}%)")

    return X_sequences_padding_np, X_market_ids_np, X_periods_np, Y_labels_np

def prepare_data(train_df, val_df):
    encoder_market_ids = preprocessing.LabelEncoder()
    encoder_periods = preprocessing.LabelEncoder()
    encoder_labels = preprocessing.LabelEncoder()

    print("\n======== Train Dataset ========\n")
    X_sequences_train, X_market_ids_train, X_periods_train, Y_labels_train = create_sequences(df=train_df)
    X_market_ids_encoded_train = encoder_market_ids.fit_transform(X_market_ids_train)
    X_periods_encoded_train = encoder_periods.fit_transform(X_periods_train)
    Y_labels_encoded_train = encoder_labels.fit_transform(Y_labels_train)

    print("\n======== Val Dataset ========\n")
    X_sequences_val, X_market_ids_val, X_periods_val, Y_labels_val = create_sequences(df=val_df)
    X_market_ids_encoded_val = encoder_market_ids.transform(X_market_ids_val)
    X_periods_encoded_val = encoder_periods.transform(X_periods_val)
    Y_labels_encoded_val = encoder_labels.transform(Y_labels_val)
    
    return (X_sequences_train, X_market_ids_encoded_train, X_periods_encoded_train, Y_labels_encoded_train), (X_sequences_val, X_market_ids_encoded_val, X_periods_encoded_val, Y_labels_encoded_val), (encoder_market_ids, encoder_periods, encoder_labels)

def balance_data(features, labels, method, random_state):
    if method == 'smote':
        # SMOTE - Synthetic Minority Oversampling Technique
        smote = SMOTE(random_state=random_state, k_neighbors=min(5, Counter(labels).most_common()[-1][1]-1))
        features_balanced, labels_balanced = smote.fit_resample(features, labels)
        
    elif method == 'adasyn':
        # ADASYN - Adaptive Synthetic Sampling
        adasyn = ADASYN(random_state=random_state)
        features_balanced, labels_balanced = adasyn.fit_resample(features, labels)
        
    elif method == 'oversample':
        # Random Oversampling
        ros = RandomOverSampler(random_state=random_state)
        features_balanced, labels_balanced = ros.fit_resample(features, labels)
        
    elif method == 'undersample':
        # Random Undersampling
        rus = RandomUnderSampler(random_state=random_state)
        features_balanced, labels_balanced = rus.fit_resample(features, labels)
        
    elif method == 'combine':
        # Combined approach: SMOTE + Tomek links
        smote_tomek = SMOTETomek(random_state=random_state)
        features_balanced, labels_balanced = smote_tomek.fit_resample(features, labels)
        
    elif method == 'smoteenn':
        # SMOTE + Edited Nearest Neighbours
        smote_enn = SMOTEENN(random_state=random_state)
        features_balanced, labels_balanced = smote_enn.fit_resample(features, labels)
         
    else:
        print(f"Unknown method: {method}. Using original data.")
        features_balanced, labels_balanced = features, labels
    
    return features_balanced, labels_balanced

def train_fn(data_loader, model, optimizer, criterion):
    model.train()
    final_loss = 0

    for batch in data_loader:
        sequences = batch['sequence'].to(config.DEVICE)
        market_ids = batch['market_id'].to(config.DEVICE)
        periods = batch['period'].to(config.DEVICE)
        labels = batch['label'].to(config.DEVICE)

        optimizer.zero_grad()
        output = model(sequences, market_ids, periods)
        loss = criterion(output, labels)
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        final_loss += loss.item()

    return final_loss/len(data_loader)

def val_fn(data_loader, model, criterion):
    
    model.eval()
    final_loss = 0
    preds_array = []
    solution_array = []

    with torch.no_grad():
        for batch in data_loader:
            sequences = batch['sequence'].to(config.DEVICE)
            market_ids = batch['market_id'].to(config.DEVICE)
            periods = batch['period'].to(config.DEVICE)
            labels = batch['label'].to(config.DEVICE)

            output =  model(sequences, market_ids, periods)
            loss =  criterion(output, labels)
 
            final_loss += loss.item()

            _, preds_labels = to_yhat(output)

            preds_array.extend(preds_labels)
            solution_array.extend(labels)

    return final_loss/len(data_loader), preds_array, solution_array