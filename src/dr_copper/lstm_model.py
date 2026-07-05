import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_sequences(
    X: pd.DataFrame,
    y: pd.Series,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    feat_cols = list(X.columns)

    arr = X[feat_cols].values.astype(np.float32)
    labels = y.values.astype(np.float32)
    dates = y.index

    n = len(arr) - seq_len + 1
    X_seq = np.stack([arr[i : i + seq_len] for i in range(n)])
    y_seq = labels[seq_len - 1 :]
    d_seq = dates[seq_len - 1 :]

    return X_seq, y_seq, d_seq  # type: ignore


@dataclass
class LSTMConfig:
    input_size: int
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 500
    val_fraction: float = 0.15
    patience: int = 30


@dataclass
class LSTMTrainerData:
    X_seq_train: np.ndarray
    y_seq_train: np.ndarray
    d_seq_train: pd.DatetimeIndex
    X_seq_test: np.ndarray
    y_seq_test: np.ndarray
    d_seq_test: pd.DatetimeIndex


class CopperLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super(CopperLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.layer_norm(out)
        out = self.dropout(out)

        return self.sigmoid(self.linear(out)).squeeze(-1)


class CopperLSTMTrainer:
    def __init__(self, cfg: LSTMConfig, data: LSTMTrainerData, seed: int = 42):
        self.cfg = cfg
        self.data = data
        torch.manual_seed(seed)
        np.random.seed(seed)

    def train(self):
        cfg = self.cfg
        data = self.data

        val_n = int(len(data.X_seq_train) * cfg.val_fraction)
        X_val, y_val = data.X_seq_train[-val_n:], data.y_seq_train[-val_n:]
        X_train, y_train = data.X_seq_train[:-val_n], data.y_seq_train[:-val_n]

        model = CopperLSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        criterion = nn.BCELoss()

        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_train, device=DEVICE),
                torch.tensor(y_train, device=DEVICE),
            ),
            batch_size=cfg.batch_size,
            shuffle=True,
        )

        X_val_t = torch.tensor(X_val, device=DEVICE)
        y_val_t = torch.tensor(y_val, device=DEVICE)
        X_test_t = torch.tensor(data.X_seq_test, device=DEVICE)
        y_test_t = torch.tensor(data.y_seq_test, device=DEVICE)

        best_val_loss = float("inf")
        best_weights = copy.deepcopy(model.state_dict())
        patience_counter = 0
        best_epoch = 0

        for epoch in range(cfg.max_epochs):
            epoch_loss = 0.0
            model.train()
            for features, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels.view(-1))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(labels)

            model.eval()
            with torch.no_grad():
                val_out = model(X_val_t)
                val_loss = criterion(val_out, y_val_t).item()

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(model.state_dict())
                patience_counter = 0
                best_epoch = epoch
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    print(
                        f"Training stopped on {epoch} epoch. The best epoch is {best_epoch}."
                    )
                    break

            epoch_loss /= len(y_train)
            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{cfg.max_epochs}], train BCELoss: {epoch_loss:.4f}, val BCELoss: {val_loss:.4f}"
                )

        model.load_state_dict(best_weights)
        model.eval()
        with torch.no_grad():
            test_out = model(X_test_t)
            test_loss = criterion(test_out, y_test_t).item()

        print(f"Test BCELoss: {test_loss:.4f}")

        return model


def lstm_test_score(data: LSTMTrainerData, fit_model: CopperLSTM):
    fit_model.eval()
    with torch.no_grad():
        test_preds = (
            fit_model(torch.tensor(data.X_seq_test, device=DEVICE)).cpu().numpy()
        )

    test_pred_series = pd.Series(test_preds, index=data.d_seq_test)

    pred_signs = (test_pred_series >= 0.5).astype(int)
    actual_signs = data.y_seq_test

    print(f"Accuracy score: {float(accuracy_score(actual_signs, pred_signs)):.2%}")
    print(f"ROC AUC score: {float(roc_auc_score(actual_signs, test_preds)):.2%}")
