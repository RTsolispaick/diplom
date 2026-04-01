import torch
import torch.nn as nn


class AttentionModule(nn.Module):
    """Attention модуль для лучшей локализации паттерна"""
    
    def __init__(self, hidden_size):
        super(AttentionModule, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x shape: (batch, seq_len, hidden_size)
        attn_out, _ = self.attention(x, x, x)
        attn_out = attn_out + x  # Residual connection
        return attn_out


class CNNLSTMModel(nn.Module):
    """CNN-LSTM модель для бинарной классификации паттерна "Голова и Плечи"."""

    def __init__(self):
        super(CNNLSTMModel, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)

        self.pool = nn.MaxPool1d(kernel_size=2)

        self.lstm1 = nn.LSTM(input_size=64, hidden_size=64, batch_first=True, dropout=0.3)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)

        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.2)

        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.pool(x)
        x = x.transpose(1, 2)

        x, _ = self.lstm1(x)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = x[:, -1, :]

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)

        x = self.fc2(x)
        x = self.sigmoid(x)

        return x


class CNNLSTMLocalizationModel(nn.Module):
    """
    Улучшенная CNN-LSTM модель для классификации и локализации начала/конца паттерна.
    
    Улучшения:
    - Bidirectional LSTM для лучшей контекстуализации
    - Attention механизм для фокусировки на релевантных временных шагов
    - Более глубокие специализированные ветви для локализации
    - Skip connections
    """

    def __init__(self):
        super(CNNLSTMLocalizationModel, self).__init__()

        # === CNN слои для извлечения признаков ===
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=96, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(96)

        self.pool = nn.MaxPool1d(kernel_size=2)

        # === Bidirectional LSTM ===
        self.lstm1 = nn.LSTM(input_size=96, hidden_size=96, batch_first=True, 
                            dropout=0.3, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=192, hidden_size=64, batch_first=True, 
                            dropout=0.2, bidirectional=True)

        # === Attention механизм ===
        self.attention = AttentionModule(128)  # 64 * 2 (bidirectional)

        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.2)

        # === Shared backbone ===
        self.fc_shared = nn.Linear(128, 128)
        self.bn_shared = nn.BatchNorm1d(128)

        # === Классификация (1 output) ===
        self.fc_class1 = nn.Linear(128, 64)
        self.bn_class1 = nn.BatchNorm1d(64)
        self.fc_class2 = nn.Linear(64, 32)
        self.fc_class3 = nn.Linear(32, 1)

        # === Локализация НАЧАЛА (более глубокая сеть) ===
        self.fc_start1 = nn.Linear(128, 96)
        self.bn_start1 = nn.BatchNorm1d(96)
        self.fc_start2 = nn.Linear(96, 64)
        self.bn_start2 = nn.BatchNorm1d(64)
        self.fc_start3 = nn.Linear(64, 32)
        self.fc_start4 = nn.Linear(32, 1)

        # === Локализация КОНЦА (более глубокая сеть) ===
        self.fc_end1 = nn.Linear(128, 96)
        self.bn_end1 = nn.BatchNorm1d(96)
        self.fc_end2 = nn.Linear(96, 64)
        self.bn_end2 = nn.BatchNorm1d(64)
        self.fc_end3 = nn.Linear(64, 32)
        self.fc_end4 = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # === CNN Feature Extraction ===
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.pool(x)
        
        # Transpose для LSTM: (batch, seq_len, channels)
        x = x.transpose(1, 2)

        # === Bidirectional LSTM ===
        x, _ = self.lstm1(x)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = self.dropout2(x)

        # === Attention ===
        x_attn = self.attention(x)
        x = x + x_attn  # Skip connection

        # Берем последний временной шаг
        x = x[:, -1, :]

        # === Shared backbone ===
        shared = self.fc_shared(x)
        shared = self.bn_shared(shared)
        shared = self.relu(shared)
        shared = self.dropout3(shared)

        # === Классификация ===
        class_out = self.fc_class1(shared)
        class_out = self.bn_class1(class_out)
        class_out = self.relu(class_out)
        class_out = self.dropout3(class_out)
        
        class_out = self.fc_class2(class_out)
        class_out = self.relu(class_out)
        class_out = self.dropout3(class_out)
        
        class_out = self.fc_class3(class_out)
        class_out = self.sigmoid(class_out)

        # === Локализация НАЧАЛА ===
        start_out = self.fc_start1(shared)
        start_out = self.bn_start1(start_out)
        start_out = self.relu(start_out)
        start_out = self.dropout2(start_out)
        
        start_out = self.fc_start2(start_out)
        start_out = self.bn_start2(start_out)
        start_out = self.relu(start_out)
        start_out = self.dropout2(start_out)
        
        start_out = self.fc_start3(start_out)
        start_out = self.relu(start_out)
        start_out = self.dropout2(start_out)
        
        start_out = self.fc_start4(start_out)
        start_out = self.sigmoid(start_out)

        # === Локализация КОНЦА ===
        end_out = self.fc_end1(shared)
        end_out = self.bn_end1(end_out)
        end_out = self.relu(end_out)
        end_out = self.dropout2(end_out)
        
        end_out = self.fc_end2(end_out)
        end_out = self.bn_end2(end_out)
        end_out = self.relu(end_out)
        end_out = self.dropout2(end_out)
        
        end_out = self.fc_end3(end_out)
        end_out = self.relu(end_out)
        end_out = self.dropout2(end_out)
        
        end_out = self.fc_end4(end_out)
        end_out = self.sigmoid(end_out)

        return class_out, start_out, end_out
