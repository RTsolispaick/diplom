# Детектор паттерна Голова и Плечи

Python-библиотека для обнаружения и анализа паттерна "Голова и Плечи" в финансовых временных рядах.

## Описание

Проект предоставляет фреймворк для:
- Загрузки исторических данных о цене с финансовых рынков
- Обнаружения паттерна "Голова и Плечи" в движении цены
- Построения структурированных наборов данных из обнаруженных паттернов
- Извлечения нормализованных окон фиче для обучения моделей

## Возможности

- **Загрузка данных**: Загрузчик временных рядов для финансовых инструментов
- **Обнаружение паттернов**: Детектирование паттерна "Голова и Плечи" с настраиваемыми параметрами
- **Анализ паттернов**: Поддержка как стандартных, так и инверсных (перевёрнутых) паттернов
- **Построение датасета**: Автоматическое создание датасета с размеченными окнами
- **Нормализация**: Поокначная нормализация для согласованного масштабирования признаков

## Требования

- Python 3.7+
- pandas
- numpy
- scipy
- scikit-learn
- yfinance

## Установка

```bash
pip install pandas numpy scipy scikit-learn yfinance
```

## Использование

### Базовый пример

```python
from data_loader import DataLoader
from pattern_detector import HeadShouldersDetector
from dataset_builder import DatasetBuilder

# Загружаем данные о цене
loader = DataLoader("AAPL", start="2015-01-01", end="2024-01-01")
df = loader.load()

# Обнаруживаем паттерны "Голова и Плечи"
detector = HeadShouldersDetector()
patterns = detector.find_patterns(df)

# Строим датасет для обучения модели
builder = DatasetBuilder(window_size=120, step=5)
X, y_labels, y_coords = builder.build(df, patterns)
```

### Обнаружение инверсных паттернов

```python
# Находим инверсные паттерны "Голова и Плечи"
inverse_patterns = detector.find_patterns(df, inverse=True)
```

### Сохранение в CSV

```python
from dataset_pipeline import save_csv

save_csv(X, y_coords, "output.csv")
```

## Конфигурация

### Параметры HeadShouldersDetector

- `order` (int): Размер окна для обнаружения локальных экстремумов (по умолчанию: 5)
- `shoulder_tol` (float): Допуск на симметричность плеч (по умолчанию: 0.25)
- `min_head_height` (float): Минимальная высота головы относительно плеч (по умолчанию: 0.02)
- `min_distance` (int): Минимальное расстояние между ключевыми точками паттерна (по умолчанию: 8)
- `max_width` (int): Максимальная ширина паттерна в барах (по умолчанию: 120)

### Параметры DatasetBuilder

- `window_size` (int): Размер каждого окна признаков (по умолчанию: 120)
- `step` (int): Шаг скользящего окна (по умолчанию: 5)

## Структура проекта

```
├── data_loader.py                # Загрузка временных рядов
├── pattern_detector.py           # Обнаружение паттернов
├── dataset_builder.py            # Построение датасета из паттернов
├── dataset_pipeline.py           # Высокоуровневые функции
├── model_architecture.py         # Архитектуры CNN-LSTM
├── train_localization_model.py   # Обучение модели с локализацией
├── inference_localization.py     # Инференс и предсказания
├── train.csv                     # Данные для обучения (с координатами)
├── test.csv                      # Тестовые данные (с координатами)
└── README.md                     # Этот файл
```

## Использование: Классификация с локализацией паттерна

### Новое в версии 2.0

Датасет теперь содержит не только метку наличия паттерна, но и его локализацию:
- `has_pattern` — есть ли на окне паттерн "Голова и плечи" (0/1)
- `pattern_start` — нормализованное начало паттерна (0-1)
- `pattern_end` — нормализованный конец паттерна (0-1)

### Шаг 1: Подготовка данных

Датасет CSV должен содержать:
```
t0,t1,...,t119,has_pattern,pattern_start,pattern_end
-2.34,-2.36,...,0.08,1,0.25,0.65
...
```

### Шаг 2: Обучение модели

```bash
python train_localization_model.py
```

Выведет:
- Процесс обучения с метриками
- Сохраняет лучшую модель в `best_model_loc.pt`
- Финальные метрики на тестовом наборе

**Параметры обучения**:
- Эпохи: 100
- Batch size: 32
- Learning rate: 0.001
- Early stopping: patience=15

### Шаг 3: Инференс

```bash
python inference_localization.py
```

Примеры использования:
```python
from inference_localization import PatternPredictor

predictor = PatternPredictor('best_model_loc.pt')

# Для одного образца
prediction = predictor.predict_sample(time_series)
# {
#     'has_pattern': 0.87,
#     'pattern_start': 0.25,
#     'pattern_end': 0.65,
#     'pattern_start_idx': 30,
#     'pattern_end_idx': 78,
#     'confidence': 0.87
# }

# Для батча
predictions = predictor.predict_batch(batch_data)
```

## Архитектура модели

### CNNLSTMLocalizationModel

**Архитектура**:
```
Input [batch, 1, 120]
  ↓
Conv1d(1→32) + BatchNorm + ReLU
Conv1d(32→64) + BatchNorm + ReLU
MaxPool1d(kernel=2)
  ↓
LSTM(64→64, dropout=0.3)
LSTM(64→32)
  ↓
Shared Dense: 32→64
  ├─ Classification: 64→32→1 [Sigmoid]
  ├─ Start Localization: 64→32→1 [Sigmoid]
  └─ End Localization: 64→32→1 [Sigmoid]

Output:
  class_out: вероятность паттерна (0-1)
  start_out: нормализованное начало (0-1)
  end_out: нормализованный конец (0-1)
```

**Параметры**: ~150K обучаемых параметров

### Функция потерь

$$\mathcal{L} = \text{BCE}(\text{class}) + \lambda \times (\text{SmoothL1}(\text{start}) + \text{SmoothL1}(\text{end}))$$

где $\lambda = 0.5$ — вес локализационных потерь.

Локализационные потери применяются только к образцам с паттерном (has_pattern=1).

## Результаты

На примере данных в проекте:
- **Точность классификации**: ~87-92%
- **F1-Score**: ~0.85-0.90  
- **Ошибка локализации**: ±3-5 индексов (из 120)
