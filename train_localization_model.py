"""
Скрипт для обучения CNN-LSTM модели с локализацией паттерна "Голова и Плечи"
Модель предсказывает: наличие паттерна, начало и конец паттерна
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings

from model_architecture import CNNLSTMLocalizationModel

warnings.filterwarnings('ignore')

# Конфигурация
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 15
LEARNING_RATE = 0.001
LAMBDA_LOC = 0.5  # Вес локализационной потери относительно классификации
WINDOW_SIZE = 120

print(f"Устройство: {DEVICE}")
print(f"=" * 80)


def load_data(train_file, test_file):
    """Загрузка данных с координатами паттернов"""
    print("Загрузка данных...")
    
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    # Разделение признаков и меток
    feature_cols = [col for col in train_df.columns if col.startswith('t')]
    
    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df['has_pattern'].values.astype(np.float32)
    start_train = train_df['pattern_start'].values.astype(np.float32)
    end_train = train_df['pattern_end'].values.astype(np.float32)
    
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df['has_pattern'].values.astype(np.float32)
    start_test = test_df['pattern_start'].values.astype(np.float32)
    end_test = test_df['pattern_end'].values.astype(np.float32)
    
    # Преобразование для Conv1D (batch, channels, timesteps)
    X_train = X_train.reshape(-1, 1, X_train.shape[1])
    X_test = X_test.reshape(-1, 1, X_test.shape[1])
    
    print(f"Размер обучающего набора: {X_train.shape}")
    print(f"Размер тестового набора: {X_test.shape}")
    print(f"Распределение классов в обучении: {(y_train == 1).sum()} с паттерном, {(y_train == 0).sum()} без")
    
    return X_train, y_train, start_train, end_train, X_test, y_test, start_test, end_test


def create_dataloaders(X_train, y_train, start_train, end_train, batch_size=32, val_split=0.2):
    """Создание DataLoader'ов для обучения и валидации"""
    # Преобразование в тензоры
    X_train_tensor = torch.FloatTensor(X_train).to(DEVICE)
    y_train_tensor = torch.FloatTensor(y_train).to(DEVICE)
    start_train_tensor = torch.FloatTensor(start_train).to(DEVICE)
    end_train_tensor = torch.FloatTensor(end_train).to(DEVICE)
    
    # Разделение на обучение и валидацию
    val_size = int(len(X_train_tensor) * val_split)
    val_indices = np.random.choice(len(X_train_tensor), val_size, replace=False)
    train_indices = np.array([i for i in range(len(X_train_tensor)) if i not in val_indices])
    
    X_train_split = X_train_tensor[train_indices]
    y_train_split = y_train_tensor[train_indices]
    start_train_split = start_train_tensor[train_indices]
    end_train_split = end_train_tensor[train_indices]
    
    X_val = X_train_tensor[val_indices]
    y_val = y_train_tensor[val_indices]
    start_val = start_train_tensor[val_indices]
    end_val = end_train_tensor[val_indices]
    
    # Создание DataLoader'ов
    train_dataset = TensorDataset(X_train_split, y_train_split, start_train_split, end_train_split)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    print(f"\nРазмер тренировочного батча: {len(X_train_split)}")
    print(f"Размер валидационного батча: {len(X_val)}")
    
    return train_loader, X_val, y_val, start_val, end_val


def train_epoch(model, train_loader, optimizer, criterion_class, criterion_loc, lambda_loc, device):
    """Обучение на одной эпохе"""
    model.train()
    total_loss = 0.0
    class_loss_sum = 0.0
    loc_loss_sum = 0.0
    correct = 0
    total = 0
    
    for X_batch, y_batch, start_batch, end_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        start_batch = start_batch.to(device)
        end_batch = end_batch.to(device)
        
        optimizer.zero_grad()
        
        # Предсказания
        class_pred, start_pred, end_pred = model(X_batch)
        
        # Потеря классификации
        loss_class = criterion_class(class_pred, y_batch.view(-1, 1))
        
        # Потеря локализации (только для образцов с паттерном)
        mask = y_batch.view(-1, 1) > 0.5
        if mask.sum() > 0:
            loss_loc = criterion_loc(start_pred[mask], start_batch.view(-1, 1)[mask]) + \
                       criterion_loc(end_pred[mask], end_batch.view(-1, 1)[mask])
        else:
            loss_loc = torch.tensor(0.0, device=device)
        
        # Общая потеря
        loss = loss_class + lambda_loc * loss_loc
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        class_loss_sum += loss_class.item()
        if mask.sum() > 0:
            loc_loss_sum += loss_loc.item()
        
        # Точность классификации
        correct += ((class_pred > 0.5).float() == y_batch.view(-1, 1)).sum().item()
        total += y_batch.size(0)
    
    avg_loss = total_loss / len(train_loader)
    avg_class_loss = class_loss_sum / len(train_loader)
    avg_loc_loss = loc_loss_sum / len(train_loader) if loc_loss_sum > 0 else 0.0
    accuracy = correct / total
    
    return avg_loss, avg_class_loss, avg_loc_loss, accuracy


def validate(model, X_val, y_val, start_val, end_val, criterion_class, criterion_loc, lambda_loc, device):
    """Валидация модели"""
    model.eval()
    with torch.no_grad():
        X_val = X_val.to(device)
        y_val = y_val.to(device)
        start_val = start_val.to(device)
        end_val = end_val.to(device)
        
        class_pred, start_pred, end_pred = model(X_val)
        
        loss_class = criterion_class(class_pred, y_val.view(-1, 1))
        
        mask = y_val.view(-1, 1) > 0.5
        if mask.sum() > 0:
            loss_loc = criterion_loc(start_pred[mask], start_val.view(-1, 1)[mask]) + \
                       criterion_loc(end_pred[mask], end_val.view(-1, 1)[mask])
        else:
            loss_loc = torch.tensor(0.0, device=device)
        
        total_loss = loss_class + lambda_loc * loss_loc
        
        correct = ((class_pred > 0.5).float() == y_val.view(-1, 1)).sum().item()
        accuracy = correct / y_val.size(0)
        
    return total_loss.item(), loss_class.item(), loss_loc.item() if loss_loc.item() > 0 else 0.0, accuracy


def train_model(model, train_loader, X_val, y_val, start_val, end_val, 
                epochs, patience, learning_rate, lambda_loc, device, checkpoint_path='best_model_loc.pt'):
    """Обучение модели с ранней остановкой"""
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion_class = nn.BCELoss()
    criterion_loc = nn.SmoothL1Loss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [],
        'train_class_loss': [],
        'train_loc_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_class_loss': [],
        'val_loc_loss': [],
        'val_acc': []
    }
    
    print(f"\n{'=' * 80}")
    print(f"Обучение модели (эпохи: {epochs}, patience: {patience})")
    print(f"{'=' * 80}\n")
    
    for epoch in range(epochs):
        # Обучение
        train_loss, train_class_loss, train_loc_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion_class, criterion_loc, lambda_loc, device
        )
        
        # Валидация
        val_loss, val_class_loss, val_loc_loss, val_acc = validate(
            model, X_val, y_val, start_val, end_val, criterion_class, criterion_loc, lambda_loc, device
        )
        
        # Сохранение в историю
        history['train_loss'].append(train_loss)
        history['train_class_loss'].append(train_class_loss)
        history['train_loc_loss'].append(train_loc_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_class_loss'].append(val_class_loss)
        history['val_loc_loss'].append(val_loc_loss)
        history['val_acc'].append(val_acc)
        
        # Ранняя остановка
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Эпоха {epoch + 1:3d} | "
                  f"Loss: {train_loss:.4f} (C:{train_class_loss:.4f} L:{train_loc_loss:.4f}) | "
                  f"Val Loss: {val_loss:.4f} (C:{val_class_loss:.4f} L:{val_loc_loss:.4f}) | "
                  f"Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} [✓ SAVED]")
        else:
            patience_counter += 1
            if (epoch + 1) % 10 == 0:
                print(f"Эпоха {epoch + 1:3d} | "
                      f"Loss: {train_loss:.4f} (C:{train_class_loss:.4f} L:{train_loc_loss:.4f}) | "
                      f"Val Loss: {val_loss:.4f} (C:{val_class_loss:.4f} L:{val_loc_loss:.4f}) | "
                      f"Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        if patience_counter >= patience:
            print(f"\n⚠ Остановка на эпохе {epoch + 1} (patience={patience})")
            break
    
    # Загрузка лучшей модели
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"\n✓ Обучение завершено! Обучено эпох: {len(history['train_loss'])}")
    
    return model, history


def evaluate_model(model, X_test, y_test, start_test, end_test, device):
    """Оценка модели на тестовом наборе"""
    print(f"\n{'=' * 80}")
    print("Оценка модели на тестовом наборе")
    print(f"{'=' * 80}\n")
    
    model.eval()
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)
    
    with torch.no_grad():
        class_pred, start_pred, end_pred = model(X_test_tensor)
        
        class_pred_np = class_pred.squeeze().cpu().numpy()
        start_pred_np = start_pred.squeeze().cpu().numpy()
        end_pred_np = end_pred.squeeze().cpu().numpy()
        
        y_pred_binary = (class_pred_np > 0.5).astype(int)
    
    # Метрики классификации
    accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary, zero_division=0)
    recall = recall_score(y_test, y_pred_binary, zero_division=0)
    f1 = f1_score(y_test, y_pred_binary, zero_division=0)
    auc = roc_auc_score(y_test, class_pred_np)
    
    print("Метрики классификации:")
    print(f"  Точность (Accuracy):  {accuracy:.4f}")
    print(f"  Precision:            {precision:.4f}")
    print(f"  Recall:               {recall:.4f}")
    print(f"  F1-Score:             {f1:.4f}")
    print(f"  ROC-AUC:              {auc:.4f}")
    
    # Метрики локализации (для образцов с паттерном)
    pattern_mask = y_test == 1
    if pattern_mask.sum() > 0:
        print(f"\nМетрики локализации (для {pattern_mask.sum()} образцов с паттерном):")
        
        start_error = np.abs(start_pred_np[pattern_mask] - start_test[pattern_mask])
        end_error = np.abs(end_pred_np[pattern_mask] - end_test[pattern_mask])
        
        print(f"  Средняя ошибка начала:   {start_error.mean():.4f} ± {start_error.std():.4f}")
        print(f"  Средняя ошибка конца:    {end_error.mean():.4f} ± {end_error.std():.4f}")
        print(f"  Макс ошибка начала:      {start_error.max():.4f}")
        print(f"  Макс ошибка конца:       {end_error.max():.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'predictions': {
            'class': class_pred_np,
            'start': start_pred_np,
            'end': end_pred_np
        }
    }


def main():
    """Главная функция"""
    # Загрузка данных
    X_train, y_train, start_train, end_train, X_test, y_test, start_test, end_test = \
        load_data('train.csv', 'test.csv')
    
    # Создание DataLoader'ов
    train_loader, X_val, y_val, start_val, end_val = \
        create_dataloaders(X_train, y_train, start_train, end_train, batch_size=BATCH_SIZE)
    
    # Создание модели
    print(f"\n{'=' * 80}")
    print("Создание модели CNN-LSTM с локализацией")
    print(f"{'=' * 80}\n")
    model = CNNLSTMLocalizationModel().to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Всего параметров: {total_params:,}")
    print(f"Обучаемых параметров: {trainable_params:,}")
    print(f"\nВыходы модели:")
    print(f"  1. class_pred - вероятность паттерна (0-1)")
    print(f"  2. start_pred - нормализованное начало (0-1)")
    print(f"  3. end_pred   - нормализованный конец (0-1)")
    
    # Обучение
    model, history = train_model(
        model, train_loader, X_val, y_val, start_val, end_val,
        epochs=EPOCHS,
        patience=PATIENCE,
        learning_rate=LEARNING_RATE,
        lambda_loc=LAMBDA_LOC,
        device=DEVICE,
        checkpoint_path='best_model_loc.pt'
    )
    
    # Оценка
    results = evaluate_model(model, X_test, y_test, start_test, end_test, DEVICE)
    
    print(f"\n{'=' * 80}")
    print("✓ Обучение и оценка завершены!")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
