import json
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn import preprocessing
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
import torch
import matplotlib.pyplot as plt
import mplfinance as mpf
import matplotlib.dates as mdates
from sklearn.utils import resample

def plot_dataset_all(df):
    n = len(df.columns)
    figsize = (18, 2*n)
    df.plot(subplots=True, figsize=figsize, title="Dataset")
    plt.tight_layout()
    plt.savefig("data/dataset.png")
    plt.close()

def plot_dataset_chart(df):
    # index datetime
    df.index = pd.to_datetime(df.index)

    colors = {"BUY": "green", "SELL": "red", "HOLD": "blue"}

    # market colors
    mc = mpf.make_marketcolors(up="g", down="r", inherit=True)
    s  = mpf.make_mpf_style(marketcolors=mc)

    # plot candlestick
    fig, ax = mpf.plot(df, type="candle", style=s,
                       ylabel="Price",
                       returnfig=True, figsize=(18,8))

    # scatter label
    for label, color in colors.items():
        idx = df[df["Label"] == label].index
        price = df.loc[idx, "Close"]
        ax[0].scatter(idx, price, label=label, color=color, s=30, marker="o")

    # x-axis = date
    ax[0].xaxis_date()
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
    fig.autofmt_xdate() # rotate

    ax[0].legend()
    plt.title("Candlestick + BUY/SELL/HOLD")
    plt.savefig("data/chart.png")
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
    horizon = 3   # lihat 3 candle ke depan
    threshold = 0.001  # 0.1% ambang batas

    # Geser harga close ke depan
    df["Future_Close"] = df["Close"].shift(-horizon)

    # Hitung return masa depan
    df["Future_Return"] = (df["Future_Close"] - df["Close"]) / df["Close"]

    # Buat label berdasarkan aturan
    def create_label(x):
        if x > threshold:
            return "BUY"
        elif x < -threshold:
            return "SELL"
        else:
            return "HOLD"

    df["Label"] = df["Future_Return"].apply(create_label)

""" def create_advanced_features(df):
    # Simple Moving Average (SMA)
    df["SMA_5"] = df["Close"].rolling(window=5).mean()

    # Exponential Moving Average (EMA)
    df["EMA_5"] = df["Close"].ewm(span=5, adjust=False).mean()

    # Relative Strength Index (RSI)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    roll_up = pd.Series(gain).rolling(14).mean()
    roll_down = pd.Series(loss).rolling(14).mean()
    rs = roll_up / roll_down
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # ATR (Average True Range)
    df["H-L"] = df["High"] - df["Low"]
    df["H-C"] = abs(df["High"] - df["Close"].shift())
    df["L-C"] = abs(df["Low"] - df["Close"].shift())
    df["TR"] = df[["H-L", "H-C", "L-C"]].max(axis=1)
    df["ATR_14"] = df["TR"].rolling(14).mean()

    # Bollinger Bands
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["STDDEV_20"] = df["Close"].rolling(window=20).std()
    df["UpperBB"] = df["SMA_20"] + (df["STDDEV_20"] * 2)
    df["LowerBB"] = df["SMA_20"] - (df["STDDEV_20"] * 2) """

def create_advanced_features(df):
    # ===== PRICE CHANGE FEATURES (Percentage based) =====
    
    # Hourly price change percentage
    df["Price_Change_Pct"] = (df["Close"] - df["Open"]) / df["Open"] * 100
    
    # Price change from previous period
    df["Price_Change_Prev_Pct"] = df["Close"].pct_change() * 100
    
    # High-Low range as percentage of close
    df["HL_Range_Pct"] = (df["High"] - df["Low"]) / df["Close"] * 100
    
    
    # ===== MOVING AVERAGES (Relative to current price) =====
    
    # SMA deviation from current price
    df["SMA_5"] = df["Close"].rolling(window=5).mean()
    df["SMA_5_Deviation_Pct"] = (df["Close"] - df["SMA_5"]) / df["SMA_5"] * 100
    
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_10_Deviation_Pct"] = (df["Close"] - df["SMA_10"]) / df["SMA_10"] * 100
    
    # EMA deviation from current price
    df["EMA_5"] = df["Close"].ewm(span=5, adjust=False).mean()
    df["EMA_5_Deviation_Pct"] = (df["Close"] - df["EMA_5"]) / df["EMA_5"] * 100
    
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_12_Deviation_Pct"] = (df["Close"] - df["EMA_12"]) / df["EMA_12"] * 100
    
    
    # ===== MOMENTUM INDICATORS (Already normalized 0-100) =====
    
    # RSI (already 0-100 scale)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    roll_up = gain.rolling(14).mean()
    roll_down = loss.rolling(14).mean()
    rs = roll_up / roll_down
    df["RSI_14"] = 100 - (100 / (1 + rs))
    
    # RSI normalized to -1 to 1 range
    df["RSI_Normalized"] = (df["RSI_14"] - 50) / 50
    
    
    # ===== MACD (Normalized) =====
    
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    
    # MACD as percentage of price
    df["MACD_Pct"] = df["MACD"] / df["Close"] * 100
    df["Signal_Pct"] = df["Signal"] / df["Close"] * 100
    df["MACD_Histogram_Pct"] = (df["MACD"] - df["Signal"]) / df["Close"] * 100
    
    
    # ===== VOLATILITY INDICATORS (Normalized) =====
    
    # ATR as percentage of price
    df["H-L"] = df["High"] - df["Low"]
    df["H-C"] = abs(df["High"] - df["Close"].shift())
    df["L-C"] = abs(df["Low"] - df["Close"].shift())
    df["TR"] = df[["H-L", "H-C", "L-C"]].max(axis=1)
    df["ATR_14"] = df["TR"].rolling(14).mean()
    df["ATR_Pct"] = df["ATR_14"] / df["Close"] * 100
    
    
    # ===== BOLLINGER BANDS (Position based) =====
    
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["STDDEV_20"] = df["Close"].rolling(window=20).std()
    df["UpperBB"] = df["SMA_20"] + (df["STDDEV_20"] * 2)
    df["LowerBB"] = df["SMA_20"] - (df["STDDEV_20"] * 2)
    
    # Bollinger Band position (0 = at lower band, 1 = at upper band)
    df["BB_Position"] = (df["Close"] - df["LowerBB"]) / (df["UpperBB"] - df["LowerBB"])
    
    # Distance from bands as percentage
    df["BB_Upper_Distance_Pct"] = (df["UpperBB"] - df["Close"]) / df["Close"] * 100
    df["BB_Lower_Distance_Pct"] = (df["Close"] - df["LowerBB"]) / df["Close"] * 100
    
    
    # ===== VOLUME INDICATORS =====
    
    """ if "Volume" in df.columns:
        # Volume change percentage
        df["Volume_Change_Pct"] = df["Volume"].pct_change() * 100
        
        # Volume moving average deviation
        df["Volume_MA_5"] = df["Volume"].rolling(5).mean()
        df["Volume_MA_Deviation_Pct"] = (df["Volume"] - df["Volume_MA_5"]) / df["Volume_MA_5"] * 100 """
    
    # ===== TREND STRENGTH INDICATORS =====
    
    # Price momentum over different periods
    df["Momentum_3"] = (df["Close"] - df["Close"].shift(3)) / df["Close"].shift(3) * 100
    df["Momentum_5"] = (df["Close"] - df["Close"].shift(5)) / df["Close"].shift(5) * 100
    df["Momentum_10"] = (df["Close"] - df["Close"].shift(10)) / df["Close"].shift(10) * 100
    
    # Trend consistency (how many of last N periods were up/down)
    df["Up_Periods_5"] = (df["Close"].diff() > 0).rolling(5).sum() / 5
    df["Down_Periods_5"] = (df["Close"].diff() < 0).rolling(5).sum() / 5
    
    
    # ===== PATTERN RECOGNITION FEATURES =====

    # Doji pattern (open close to close)
    df["Doji_Pattern"] = abs(df["Close"] - df["Open"]) / (df["High"] - df["Low"])
    
    # Hammer/Shooting star patterns
    body_size = abs(df["Close"] - df["Open"])
    upper_shadow = df["High"] - df[["Close", "Open"]].max(axis=1)
    lower_shadow = df[["Close", "Open"]].min(axis=1) - df["Low"]
    
    df["Upper_Shadow_Ratio"] = upper_shadow / body_size
    df["Lower_Shadow_Ratio"] = lower_shadow / body_size
    
    # ===== User features =====
    user_features = ["Open", "High", "Low", "Close", "H-L", "H-C", "L-C", "TR", "SMA_5", "SMA_10", "SMA_20", "EMA_5", "EMA_12", "STDDEV_20", "UpperBB", "LowerBB", "MACD", "Signal", "ATR_14"]
    
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
    """ if "Volume_Change_Pct" in df.columns:
        ai_features.extend(["Volume_Change_Pct", "Volume_MA_Deviation_Pct"]) """
        
    return user_features, ai_features

def get_data(data_path):
    with open(data_path, "r") as f:
        data = json.load(f)
    
    candles = data["Candles"][0]["Candles"]
    df = pd.DataFrame(candles)
    df["FromDate"] = pd.to_datetime(df["FromDate"])
    df.set_index("FromDate", inplace=True)

    user_features, ai_features = create_advanced_features(df=df)
    create_labels(df=df)
    
    plot_dataset_all(df=df)
    plot_dataset_chart(df=df[user_features + ["Label"]])

    # label encoding
    encoder_name = preprocessing.LabelEncoder()
    df['Label'] = encoder_name.fit_transform(df["Label"])

    # ambil hanya feature yg penting
    df = df[ai_features + ["Label"]].dropna(axis=0)
    print(df)
    
    labels = df['Label'].values
    features = df.apply(pd.to_numeric, errors="coerce").values
    
    return features, labels, encoder_name

""" 
# balance data
def get_data(data_path):
    with open(data_path, "r") as f:
        data = json.load(f)
    
    candles = data["Candles"][0]["Candles"]
    df = pd.DataFrame(candles)
    df["FromDate"] = pd.to_datetime(df["FromDate"])
    df.set_index("FromDate", inplace=True)

    create_advanced_features(df=df)
    create_labels(df=df)
    
    indicators = ["Open", "High", "Low", "Close", "SMA_5", "EMA_5", "RSI_14", "MACD", "Signal", "ATR_14", "UpperBB", "LowerBB"]
    df = df[indicators + ["Label"]].dropna(axis=0)
    
    # Encode label
    encoder_name = preprocessing.LabelEncoder()
    df['Label'] = encoder_name.fit_transform(df["Label"])
    
    # =====================
    # Balancing dataset
    # =====================
    # Pisahkan per kelas
    classes = df['Label'].unique()
    dfs = [df[df['Label'] == c] for c in classes]
    
    # Tentukan jumlah data target (kelas mayoritas)
    max_size = max(len(d) for d in dfs)
    
    # Oversample semua kelas ke jumlah max_size
    dfs_upsampled = [resample(d, replace=True, n_samples=max_size, random_state=42) for d in dfs]
    
    # Gabungkan semua kelas
    df_balanced = pd.concat(dfs_upsampled)
    
    # Shuffle dataset
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    important_indicators = ["RSI_14", "MACD", "Signal", "ATR_14"]

    labels = df_balanced['Label'].values
    features = df_balanced[important_indicators].apply(pd.to_numeric, errors="coerce").values
    
    return features, labels, encoder_name """

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

def val_fn( data_loader,
            model,
            device):
    
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