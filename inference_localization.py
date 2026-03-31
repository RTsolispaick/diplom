"""
Скрипт для инференса модели локализации паттерна
Загружает обученную модель и делает предсказания на новых данных
"""

import numpy as np
import pandas as pd
import torch
from model_architecture import CNNLSTMLocalizationModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WINDOW_SIZE = 120


class PatternPredictor:
    """Класс для предсказания паттернов и их локализации"""
    
    def __init__(self, model_path='best_model_loc.pt', device=None):
        """Инициализация с загрузкой модели"""
        self.device = device or DEVICE
        self.model = CNNLSTMLocalizationModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"✓ Модель загружена из {model_path}")
    
    def predict_sample(self, time_series):
        """
        Предсказание для одного временного ряда
        
        Args:
            time_series: numpy array размером (120,) или (1, 120)
        
        Returns:
            dict с ключами:
            - has_pattern: вероятность наличия паттерна (0-1)
            - pattern_start: нормализованное начало (0-1)
            - pattern_end: нормализованный конец (0-1)
            - pattern_start_idx: индекс начала (0-120)
            - pattern_end_idx: индекс конца (0-120)
        """
        if time_series.ndim == 1:
            time_series = time_series.reshape(1, 120)
        
        # Преобразование для модели (batch, channels, timesteps)
        x = time_series.reshape(1, 1, 120).astype(np.float32)
        x_tensor = torch.FloatTensor(x).to(self.device)
        
        with torch.no_grad():
            class_pred, start_pred, end_pred = self.model(x_tensor)
            
            class_prob = class_pred.squeeze().cpu().numpy()
            start_norm = start_pred.squeeze().cpu().numpy()
            end_norm = end_pred.squeeze().cpu().numpy()
        
        # Денормализация координат
        start_idx = int(np.clip(start_norm * WINDOW_SIZE, 0, WINDOW_SIZE - 1))
        end_idx = int(np.clip(end_norm * WINDOW_SIZE, 0, WINDOW_SIZE - 1))
        
        return {
            'has_pattern': float(class_prob),
            'has_pattern_label': 'Да' if class_prob > 0.5 else 'Нет',
            'pattern_start': float(start_norm),
            'pattern_end': float(end_norm),
            'pattern_start_idx': start_idx,
            'pattern_end_idx': end_idx,
            'confidence': max(class_prob, 1 - class_prob)
        }
    
    def predict_batch(self, time_series_batch):
        """
        Предсказание для батча временных рядов
        
        Args:
            time_series_batch: numpy array размером (N, 120)
        
        Returns:
            list с предсказаниями для каждого образца
        """
        batch_size = time_series_batch.shape[0]
        x = time_series_batch.reshape(batch_size, 1, 120).astype(np.float32)
        x_tensor = torch.FloatTensor(x).to(self.device)
        
        with torch.no_grad():
            class_pred, start_pred, end_pred = self.model(x_tensor)
            
            class_pred_np = class_pred.squeeze().cpu().numpy()
            start_pred_np = start_pred.squeeze().cpu().numpy()
            end_pred_np = end_pred.squeeze().cpu().numpy()
        
        results = []
        for i in range(batch_size):
            class_prob = class_pred_np[i] if batch_size > 1 else class_pred_np
            start_norm = start_pred_np[i] if batch_size > 1 else start_pred_np
            end_norm = end_pred_np[i] if batch_size > 1 else end_pred_np
            
            start_idx = int(np.clip(start_norm * WINDOW_SIZE, 0, WINDOW_SIZE - 1))
            end_idx = int(np.clip(end_norm * WINDOW_SIZE, 0, WINDOW_SIZE - 1))
            
            results.append({
                'has_pattern': float(class_prob),
                'has_pattern_label': 'Да' if class_prob > 0.5 else 'Нет',
                'pattern_start': float(start_norm),
                'pattern_end': float(end_norm),
                'pattern_start_idx': start_idx,
                'pattern_end_idx': end_idx,
                'confidence': max(class_prob, 1 - class_prob)
            })
        
        return results


def example_single_prediction():
    """Пример предсказания для одного образца"""
    print("=" * 80)
    print("Пример 1: Предсказание для одного временного ряда")
    print("=" * 80 + "\n")
    
    # Загрузка данных
    test_df = pd.read_csv('test.csv')
    feature_cols = [col for col in test_df.columns if col.startswith('t')]
    
    # Берем первый образец
    sample = test_df[feature_cols].iloc[0].values
    actual_label = test_df['has_pattern'].iloc[0]
    actual_start = test_df['pattern_start'].iloc[0]
    actual_end = test_df['pattern_end'].iloc[0]
    
    # Предсказание
    predictor = PatternPredictor('best_model_loc.pt')
    prediction = predictor.predict_sample(sample)
    
    # Вывод результатов
    print(f"Фактические значения:")
    print(f"  - Паттерн: {actual_label} (1=да, 0=нет)")
    print(f"  - Начало: {actual_start:.4f}")
    print(f"  - Конец: {actual_end:.4f}")
    
    print(f"\nПредсказания модели:")
    print(f"  - Паттерн: {prediction['has_pattern']:.4f} ({prediction['has_pattern_label']})")
    print(f"  - Начало: {prediction['pattern_start']:.4f} (индекс: {prediction['pattern_start_idx']})")
    print(f"  - Конец: {prediction['pattern_end']:.4f} (индекс: {prediction['pattern_end_idx']})")
    print(f"  - Уверенность: {prediction['confidence']:.4f}")


def example_batch_prediction():
    """Пример предсказания для батча образцов"""
    print("\n" + "=" * 80)
    print("Пример 2: Предсказание для батча образцов")
    print("=" * 80 + "\n")
    
    # Загрузка данных
    test_df = pd.read_csv('test.csv')
    feature_cols = [col for col in test_df.columns if col.startswith('t')]
    
    # Берем первые 5 образцов
    samples = test_df[feature_cols].iloc[:5].values
    labels = test_df['has_pattern'].iloc[:5].values
    
    # Предсказание
    predictor = PatternPredictor('best_model_loc.pt')
    predictions = predictor.predict_batch(samples)
    
    # Вывод результатов
    print(f"{'Idx':<5} {'Фактич':<10} {'Предсказ':<12} {'Уверен':<10} {'Нач':<8} {'Конец':<8}")
    print("-" * 60)
    
    for i, (pred, actual_label) in enumerate(zip(predictions, labels)):
        print(f"{i:<5} {actual_label:<10} {pred['has_pattern']:.4f}     "
              f"{pred['confidence']:.4f}    "
              f"{pred['pattern_start_idx']:<8} {pred['pattern_end_idx']:<8}")


def example_csv_inference():
    """Пример: обработка всего тестового набора и сохранение результатов"""
    print("\n" + "=" * 80)
    print("Пример 3: Инференс на всем тестовом наборе и сохранение результатов")
    print("=" * 80 + "\n")
    
    # Загрузка данных
    test_df = pd.read_csv('test.csv')
    feature_cols = [col for col in test_df.columns if col.startswith('t')]
    
    samples = test_df[feature_cols].values
    actual_labels = test_df['has_pattern'].values
    
    # Предсказание
    predictor = PatternPredictor('best_model_loc.pt')
    predictions = predictor.predict_batch(samples)
    
    # Создание результирующего датасета
    results_df = pd.DataFrame({
        'actual_pattern': actual_labels,
        'predicted_prob': [p['has_pattern'] for p in predictions],
        'predicted_label': [p['has_pattern_label'] for p in predictions],
        'confidence': [p['confidence'] for p in predictions],
        'start_actual': test_df['pattern_start'].values,
        'start_predicted': [p['pattern_start'] for p in predictions],
        'end_actual': test_df['pattern_end'].values,
        'end_predicted': [p['pattern_end'] for p in predictions],
    })
    
    # Сохранение
    results_df.to_csv('predictions_results.csv', index=False)
    print(f"✓ Результаты сохранены в predictions_results.csv")
    
    # Статистика
    correct = (results_df['actual_pattern'] == (results_df['predicted_prob'] > 0.5)).sum()
    accuracy = correct / len(results_df)
    
    print(f"\nОбщая статистика:")
    print(f"  - Всего образцов: {len(results_df)}")
    print(f"  - Правильные предсказания: {correct}/{len(results_df)}")
    print(f"  - Точность: {accuracy:.4f}")
    
    # Показываем примеры
    print(f"\nПримеры предсказаний:")
    print(results_df.head(10).to_string())


def main():
    """Главная функция"""
    print(f"{'=' * 80}")
    print("Инференс модели локализации паттерна \"Голова и Плечи\"")
    print(f"{'=' * 80}\n")
    
    # Примеры
    example_single_prediction()
    example_batch_prediction()
    example_csv_inference()
    
    print(f"\n{'=' * 80}")
    print("✓ Все примеры завершены!")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
