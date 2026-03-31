import torch.nn as nn


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
    """CNN-LSTM модель для классификации и локализации начала/конца паттерна."""

    def __init__(self):
        super(CNNLSTMLocalizationModel, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)

        self.pool = nn.MaxPool1d(kernel_size=2)

        self.lstm1 = nn.LSTM(input_size=64, hidden_size=64, batch_first=True, dropout=0.3)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)

        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.2)

        self.fc_shared = nn.Linear(32, 64)

        self.fc_class1 = nn.Linear(64, 32)
        self.fc_class2 = nn.Linear(32, 1)

        self.fc_start1 = nn.Linear(64, 32)
        self.fc_start2 = nn.Linear(32, 1)

        self.fc_end1 = nn.Linear(64, 32)
        self.fc_end2 = nn.Linear(32, 1)

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

        shared = self.fc_shared(x)
        shared = self.relu(shared)
        shared = self.dropout2(shared)

        class_out = self.fc_class1(shared)
        class_out = self.relu(class_out)
        class_out = self.fc_class2(class_out)
        class_out = self.sigmoid(class_out)

        start_out = self.fc_start1(shared)
        start_out = self.relu(start_out)
        start_out = self.fc_start2(start_out)
        start_out = self.sigmoid(start_out)

        end_out = self.fc_end1(shared)
        end_out = self.relu(end_out)
        end_out = self.fc_end2(end_out)
        end_out = self.sigmoid(end_out)

        return class_out, start_out, end_out
