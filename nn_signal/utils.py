import torch
import numpy as np
import pandas as pd
from sklearn import preprocessing
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import mplfinance as mpf
from sklearn.utils import resample
import config
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import random

class DatasetManager(torch.utils.data.Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'sequence': self.sequences[idx],
            'label': self.labels[idx]
        }
    
def to_yhat(logits):
    logits = logits.view(-1, logits.shape[-1]).cpu().detach()
    probs = torch.softmax(logits, dim=1)
    y_hat = torch.argmax(probs, dim=1)
    return probs.numpy(), y_hat.numpy()

#########################################################################################

def plot_dataset_all(df, filename="data/dataset.png"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    n = len(df.columns)
    figsize = (18, 2*n)
    df.plot(subplots=True, figsize=figsize, title="Dataset")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_dataset_chart(df, filename="data/chart.png"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

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

def show_conf_matrix(preds_array, solution_array, label, title, binary=False, plot=False, filename="data/cmatrix.png"):
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
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            conf_matrix_np = conf_matrix.cpu().numpy()
            fig, ax = plot_confusion_matrix(conf_mat=conf_matrix_np, class_names=label, figsize=(8, 8), cmap="Blues")
            plt.tight_layout(pad=3.0)
            plt.title(title)
            plt.xlabel("Prediction")
            plt.ylabel("Solution")
            plt.savefig(filename)
            plt.close()

        return conf_matrix
    else:
        return None

def print_table_info(df, title, filename="data/log_df.csv"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)

    print("="*50)
    print(title)
    print(df)
    
    if 'Label' in df.columns:
        label_counts = df['Label'].value_counts()
        print("\nLabel Counts")
        for label, count in label_counts.items():
            print(f"- {label}: {count}")
    
    print("="*50)

#########################################################################################

def create_indicators(df, normalize=True, scaler_type='minmax'):
    # Copy dataframe
    data = df.copy()
    
    data['volume'] = data['volume'].fillna(0)
    data['adjusted_close'] = data['adjusted_close'].fillna(0)
    
    # ========== INDICATORS ==========
    
    # Moving Averages - convert to ratios
    data['sma_5_ratio'] = data['close'] / data['close'].rolling(5).mean()
    data['sma_10_ratio'] = data['close'] / data['close'].rolling(10).mean()
    data['sma_20_ratio'] = data['close'] / data['close'].rolling(20).mean()
    data['ema_5_ratio'] = data['close'] / data['close'].ewm(span=5).mean()
    data['ema_10_ratio'] = data['close'] / data['close'].ewm(span=10).mean()
    data['ema_20_ratio'] = data['close'] / data['close'].ewm(span=20).mean()
    
    # RSI (already 0-100)
    def calculate_rsi(prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))
    
    data['rsi'] = calculate_rsi(data['close']) / 100  # Normalize to 0-1
    
    # MACD - normalize by close price
    ema_12 = data['close'].ewm(span=12).mean()
    ema_26 = data['close'].ewm(span=26).mean()
    data['macd_norm'] = (ema_12 - ema_26) / data['close']
    data['macd_signal_norm'] = data['macd_norm'].ewm(span=9).mean()
    data['macd_hist_norm'] = data['macd_norm'] - data['macd_signal_norm']
    
    # Bollinger Bands Position (already 0-1)
    bb_sma = data['close'].rolling(20).mean()
    bb_std = data['close'].rolling(20).std()
    bb_upper = bb_sma + (bb_std * 2)
    bb_lower = bb_sma - (bb_std * 2)
    data['bb_position'] = (data['close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)
    data['bb_position'] = data['bb_position'].clip(0, 1)  # Clip to 0-1 range
    
    # Price Features (already normalized)
    data['price_change'] = data['close'].pct_change()
    data['price_position'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    data['hl_spread'] = (data['high'] - data['low']) / data['close']
    
    # Volume Features
    volume_sma = data['volume'].rolling(20).mean()
    data['volume_ratio'] = data['volume'] / (volume_sma + 1e-8)
    # Cap volume ratio to reasonable range
    data['volume_ratio'] = data['volume_ratio'].clip(0, 5)
    
    # ATR normalized by close
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - data['close'].shift())
    tr3 = abs(data['low'] - data['close'].shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    data['atr_norm'] = true_range.rolling(14).mean() / data['close']
    
    # Lagged Features - as ratios
    data['close_lag_1_ratio'] = data['close'] / data['close'].shift(1)
    data['close_lag_2_ratio'] = data['close'] / data['close'].shift(2)
    data['rsi_lag_1'] = calculate_rsi(data['close']).shift(1) / 100
    
    # Volume lag (handle zero volume)
    data['volume_lag_1_ratio'] = data['volume'] / (data['volume'].shift(1) + 1e-8)
    data['volume_lag_1_ratio'] = data['volume_lag_1_ratio'].clip(0, 5)
    
    # Rolling Stats - normalized
    data['close_volatility_5'] = data['close'].rolling(5).std() / data['close']
    data['close_volatility_20'] = data['close'].rolling(20).std() / data['close']
    
    # Momentum (already as ratios)
    data['momentum_5'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_10'] = data['close'] / data['close'].shift(10) - 1
    
    # Additional useful indicators
    data['rsi_momentum'] = data['rsi'] - calculate_rsi(data['close']).shift(1) / 100
    data['price_acceleration'] = data['price_change'] - data['price_change'].shift(1)
    
    # List indikator untuk LSTM (sudah dalam bentuk yang lebih normalized)
    ai_indicators = [
        'sma_5_ratio', 'sma_10_ratio', 'sma_20_ratio', 
        'ema_5_ratio', 'ema_10_ratio', 'ema_20_ratio',
        'rsi', 'macd_norm', 'macd_signal_norm', 'macd_hist_norm',
        'bb_position', 'price_change', 'price_position', 'hl_spread',
        'volume_ratio', 'atr_norm', 
        'close_lag_1_ratio', 'close_lag_2_ratio', 
        'rsi_lag_1', 'volume_lag_1_ratio',
        'close_volatility_5', 'close_volatility_20',
        'momentum_5', 'momentum_10',
        'rsi_momentum', 'price_acceleration'
    ]
    
    """ if normalize:
        scaler = MinMaxScaler() if scaler_type == 'minmax' else StandardScaler()
        
        # Only normalize the indicator columns
        indicator_data = data[ai_indicators].copy()
        
        # Handle infinite and NaN values
        indicator_data = indicator_data.replace([np.inf, -np.inf], np.nan)
        indicator_data = indicator_data.ffill().bfill()
        indicator_data = indicator_data.fillna(0)
        
        # Apply scaling
        scaled_data = scaler.fit_transform(indicator_data)
        
        # Replace in original dataframe
        for i, col in enumerate(ai_indicators):
            data[col + '_scaled'] = scaled_data[:, i]
        
        # Update indicators list to use scaled version
        ai_indicators = [col + '_scaled' for col in ai_indicators]
        
        # Store scaler for inverse transform later
        data.scaler = scaler """
    
    if normalize:
        scaler = MinMaxScaler() if scaler_type == 'minmax' else StandardScaler()
        
        # Only normalize the indicator columns
        indicator_data = data[ai_indicators].copy()
        
        # Handle infinite values only, keep NaN as NaN
        indicator_data = indicator_data.replace([np.inf, -np.inf], np.nan)
        
        # Create a mask to track original NaN positions
        nan_mask = indicator_data.isna()
        
        # For scaling, we need to temporarily fill NaN values
        # We'll use forward fill and backward fill only for scaling
        temp_data = indicator_data.ffill().bfill().fillna(0)
        
        # Apply scaling on the temporarily filled data
        scaled_data = scaler.fit_transform(temp_data)
        
        # Convert back to DataFrame to easily apply the NaN mask
        scaled_df = pd.DataFrame(scaled_data, columns=ai_indicators, index=indicator_data.index)
        
        # Restore NaN values at their original positions
        scaled_df[nan_mask] = np.nan
        
        # Replace in original dataframe
        for i, col in enumerate(ai_indicators):
            data[col + '_scaled'] = scaled_df.iloc[:, i]
        
        # Update indicators list to use scaled version
        ai_indicators = [col + '_scaled' for col in ai_indicators]
        
        # Store scaler for inverse transform later
        data.scaler = scaler
    
    return data, ai_indicators

def create_labels(df):
    df = df.copy()
    # Label
    df["Future_Close"] = df["close"].shift(-config.HORIZON_LABEL)
    df["Future_Return"] = (df["Future_Close"] - df["close"]) / df["close"]

    def create_label(x):
        if np.isnan(x):
            return np.nan
        elif x > config.THRESHOLD_LABEL:
            return "BUY"
        elif x < -config.THRESHOLD_LABEL:
            return "SELL"
        else:
            if config.THRESHOLD_LABEL == 0:
                return random.choice(["BUY", "SELL"])
            else:
                return "HOLD"

    # hapus label yg nan
    df.dropna(subset=['Future_Return'], axis=0, inplace=True)

    # dapatkan signal
    df["Label"] = df["Future_Return"].apply(create_label)
    
    df.drop(columns=["Future_Close"], inplace=True)
    df.drop(columns=["Future_Return"], inplace=True)
    return df

def create_sequences(df, sequence_length=config.SEQUENCE_CANDLE_LENGTH):
    X_sequences = []
    Y_labels = []
    
    # Group by market_id dan period
    for (market_id, period), group in df.groupby(['market_id', 'period']):
        group = group.sort_index()  # Sort by timestamp
        print_table_info(group, "Dataset")
        
        if len(group) < sequence_length + 1:
            continue
        
        for i in range(len(group) - sequence_length):
            # Get n candles sequence
            group_candle_sequence = group.iloc[i:i+sequence_length]
            
            # Add Labels
            group_candle_sequence_label = create_labels(df=group_candle_sequence)

            # Add Indicators
            group_candle_sequence_indicator, ai_features = create_indicators(df=group_candle_sequence_label)

            # Clean Data
            group_candle_sequence_indicator = group_candle_sequence_indicator.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

            sequence = group_candle_sequence_indicator[ai_features].iloc[-1]
            target_label = group_candle_sequence_indicator['Label'].iloc[-1]

            X_sequences.append(sequence)
            Y_labels.append(target_label)

            #print(sequence.shape)
            #print(sequence)
            #print(mask)
            #print(market_id)
            #print(target_label)
            #print_table_info(df=group_candle_sequence_indicator, title=f"Signal: {target_label}\nMarket ID: {market_id}\nPeriod: {period}")
            #break
    
    X_sequences_np = np.array(X_sequences)
    Y_labels_np = np.array(Y_labels)

    print(f"\nData shapes:")
    print(f"- X_sequences: {X_sequences_np.shape}")
    print(f"- Y_labels: {Y_labels_np.shape}")
    
    print(f"\nLabel distribution:")
    unique, counts = np.unique(Y_labels_np, return_counts=True)
    for signal, count in zip(unique, counts):
        print(f"- {signal}: {count} ({count/len(Y_labels_np)*100:.1f}%)")

    return X_sequences_np, Y_labels_np

def prepare_data(train_df, val_df):
    encoder_labels = preprocessing.LabelEncoder()

    print("\n======== Train Dataset ========\n")
    X_sequences_train, Y_labels_train = create_sequences(df=train_df)

    Y_labels_encoded_train = encoder_labels.fit_transform(Y_labels_train)

    train_labels = set(encoder_labels.classes_)

    print("\n======== Val Dataset ========\n")
    X_sequences_val, Y_labels_val = create_sequences(df=val_df)

    mask_labels = np.array([label in train_labels for label in Y_labels_val])

    X_sequences_val_filtered = X_sequences_val[mask_labels]
    Y_labels_val_filtered = Y_labels_val[mask_labels]


    Y_labels_encoded_val_filtered = encoder_labels.transform(Y_labels_val_filtered)
    
    return (X_sequences_train, Y_labels_encoded_train), (X_sequences_val_filtered, Y_labels_encoded_val_filtered), encoder_labels