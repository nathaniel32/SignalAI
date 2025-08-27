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
        ylabel='Price'
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

""" class DatasetManager(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        labels = torch.tensor(self.labels[idx], dtype=torch.int64)
        return features, labels """

class DatasetManager(torch.utils.data.Dataset):
    def __init__(self, features, labels, sequence_length):
        self.sequence_length = sequence_length
        self.features = []
        self.labels = []
        
        for i in range(len(features) - sequence_length + 1):
            self.features.append(features[i:i + sequence_length])
            self.labels.append(labels[i + sequence_length - 1])
        
        self.features = torch.FloatTensor(np.array(self.features))
        self.labels = torch.LongTensor(np.array(self.labels))
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def create_labels(df):
    # Label
    THRESHOLD_LABEL = 0.001  # 0.1% ambang batas

    # Geser harga close ke depan
    df["Future_Close"] = df["close"].shift(-config.HORIZON_LABEL)

    # Hitung return masa depan
    df["Future_Return"] = (df["Future_Close"] - df["close"]) / df["close"]

    # Buat label berdasarkan aturan
    def create_label(x):
        if x > THRESHOLD_LABEL:
            return "BUY"
        elif x < -config.THRESHOLD_LABEL:
            return "SELL"
        else:
            return "HOLD"

    df["Label"] = df["Future_Return"].apply(create_label)

def create_advanced_features(df):
    #df["open"] =df["open"].astype(float)
    #df["close"] =df["close"].astype(float)
    #df["low"] =df["low"].astype(float)
    #df["high"] =df["high"].astype(float)
    #df["volume"] =df["volume"].astype(float)
    #df["adjusted_close"] =df["adjusted_close"].astype(float)
    
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
        df["Volume_Change_Pct"] = df["volume"].pct_change() * 100
        
        # Volume moving average deviation
        df["Volume_MA_5"] = df["volume"].rolling(5).mean()
        df["Volume_MA_Deviation_Pct"] = (df["volume"] - df["Volume_MA_5"]) / df["Volume_MA_5"] * 100
    
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
    user_features = ["open", "high", "low", "close", "H-L", "H-C", "L-C", "TR", "SMA_5", "SMA_10", "SMA_20", "EMA_5", "EMA_12", "STDDEV_20", "UpperBB", "LowerBB", "MACD", "Signal", "ATR_14"]
    
    """ if "Volume_MA_5" in df.columns:
        user_features.append("Volume_MA_5") """
    
    #df = df.drop(columns=[col for col in user_features if col in df.columns])
    
    # ===== ai features =====
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
    ]
    
    # volume features
    if "Volume_Change_Pct" in df.columns:
        ai_features.extend(["Volume_Change_Pct", "Volume_MA_Deviation_Pct"])
        
    return user_features, ai_features

def prepare_data(df, balance_method='smote', random_state=42):
    user_features, ai_features = create_advanced_features(df=df)
    create_labels(df=df)
    
    plot_dataset_all(df=df[ai_features])
    plot_dataset_chart(df=df[user_features + ["Label"]])

    # label encoding
    encoder_name = preprocessing.LabelEncoder()
    df['Label'] = encoder_name.fit_transform(df["Label"])

    # ambil hanya feature yg penting
    df = df[ai_features + ["Label"]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    
    labels = df['Label'].values
    features = df.drop(columns=["Label"]).values

    # Apply balancing technique
    features_balanced, labels_balanced = balance_data(features, labels, method=balance_method, random_state=random_state)
    
    return features_balanced, labels_balanced, encoder_name

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

loss_fn = nn.CrossEntropyLoss()

def train_fn(data_loader, model, optimizer, device):
    model.train()
    final_loss = 0

    for batch in data_loader:
        features, labels = batch
        features = features.to(device)
        labels = labels.to(device)

        # zero
        optimizer.zero_grad()

        # Forward
        output = model(features)
        
        # loss
        loss = loss_fn(output, labels)

        # Backward
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        final_loss += loss.item()

    return final_loss/len(data_loader)

def val_fn(data_loader, model, device):
    
    model.eval()
    final_loss = 0
    preds_array = []
    solution_array = []

    with torch.no_grad():
        for batch in data_loader:
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)

            # Forward
            output =  model(features)
            
            # loss
            loss =  loss_fn(output, labels)
 
            final_loss += loss.item()

            _, preds_labels = to_yhat(output)

            preds_array.extend(preds_labels)
            solution_array.extend(labels)

    return final_loss/len(data_loader), preds_array, solution_array