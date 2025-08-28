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

def create_labels(df, hold=True):
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

    df["Label"] = df["Future_Return"].apply(create_label)
    
    df.drop(columns=["Future_Close"], inplace=True)
    df.drop(columns=["Future_Return"], inplace=True)

def print_table_info(df, title):
    print("\n", title)
    
    print(df)
    
    label_counts = df['Label'].value_counts()
    print(label_counts)

def create_sequences(df, sequence_length=config.SEQUENCE_CANDLE_LENGTH):
    X_sequences = []
    X_market_ids = []
    X_periods = []
    Y_labels = []
    
    # Group by market_id dan period
    for (market_id, period), group in df.groupby(['market_id', 'period']):
        group = group.sort_index()  # Sort by timestamp
        create_labels(df=group)
        print_table_info(group, "Dataset")
        
        if len(group) < sequence_length + 1:
            continue
        
        ohlc_data = group[config.PRICE_COLUMNS].values
        
        for i in range(len(group) - sequence_length):
            # Get n candles sequence
            sequence = ohlc_data[i:i+sequence_length]
            
            # Percentage change normalization
            normalized_sequence = (sequence / sequence[0]) - 1
            
            X_sequences.append(sequence)
            #X_sequences.append(normalized_sequence)
            X_market_ids.append(market_id)
            X_periods.append(period)
            
            # Target Label
            target_label = group.iloc[i + sequence_length - 1]['Label']
            Y_labels.append(target_label)
    
    return (np.array(X_sequences), np.array(X_market_ids), np.array(X_periods), np.array(Y_labels))

def prepare_data(df):
    X_sequences, X_market_ids, X_periods, Y_labels = create_sequences(df)

    print(f"Data shapes:")
    print(f"X_sequences: {X_sequences.shape}")
    print(f"X_market_ids: {X_market_ids.shape}")  
    print(f"X_periods: {X_periods.shape}")
    print(f"Y_labels: {Y_labels.shape}")
    
    print(f"\nLabel distribution:")
    unique, counts = np.unique(Y_labels, return_counts=True)
    for signal, count in zip(unique, counts):
        print(f"{signal}: {count} ({count/len(Y_labels)*100:.1f}%)")

    # encoding
    encoder_market_ids = preprocessing.LabelEncoder()
    encoder_periods = preprocessing.LabelEncoder()
    encoder_labels = preprocessing.LabelEncoder()

    X_market_ids_encoded = encoder_market_ids.fit_transform(X_market_ids)
    X_periods_encoded = encoder_periods.fit_transform(X_periods)
    Y_labels_encoded = encoder_labels.fit_transform(Y_labels)

    return (X_sequences, X_market_ids_encoded, X_periods_encoded, Y_labels_encoded), (encoder_market_ids, encoder_periods, encoder_labels)

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