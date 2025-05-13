<<<<<<< HEAD
# TradeFusion
=======
# TradeFusion Simulation

A Python simulation of a crypto trading rewards platform that models user economics, project economics, and platform financials over time.

## Overview

TradeFusion is a platform where:
1. Token projects pay subscription fees to create trading challenges
2. Traders connect their exchange accounts to earn rewards for trading activity
3. 40% of subscription fees go directly to reward pools for traders
4. Different subscription tiers offer different reward amounts
5. Traders can earn from multiple challenges simultaneously

## Project Structure

- `trade_fusion_simulation.py`: Core simulation classes and logic
- `main.py`: Example usage and scenario runner

## Key Components

### Типы трейдеров

- **Casual**: $5,000-$10,000 месячный объем
  - Начинающие трейдеры или трейдеры с частичной занятостью
  - Низкий уровень участия в челленджах (1-3 челленджа)
  - Средний ROI и доходность

- **Active**: $50,000-$100,000 месячный объем
  - Регулярные трейдеры со средним объемом
  - Средний уровень участия в челленджах (3-5 челленджей)
  - Выше среднего ROI и доходность

- **Professional**: $500,000-$1,000,000 месячный объем
  - Профессиональные трейдеры, для которых торговля - основной доход
  - Высокий уровень участия в челленджах (5-8 челленджей)
  - Высокий ROI и доходность

- **Whale**: $5,000,000-$10,000,000 месячный объем
  - Институциональные игроки с очень высоким объемом
  - Очень высокий уровень участия в челленджах (8-10 челленджей)
  - Очень высокий ROI и доходность

### Subscription Tiers
- **Standard**: $2,000/month ($800 to reward pool)
- **Premium**: $5,000/month ($2,000 to reward pool)
- **Enterprise**: $10,000/month ($4,000 to reward pool)

### Reward Distribution Mechanisms
- **Volume Leaderboards** (60% of rewards): Top traders based on volume
- **Random Draws** (20% of rewards): Weighted by volume with minimum threshold
- **Milestone Achievements** (15% of rewards): Fixed rewards for volume thresholds
- **Special Events** (5% of rewards): Project-specific rewards

## Usage

```python
# Run the simulation with default parameters
python main.py
```

The simulation will generate visualizations and analysis in the `output` directory.

## Simulation Parameters

Симуляция включает следующие настраиваемые параметры:

- Начальное количество трейдеров: 1,000 (распределение: 70% casual, 20% active, 8% professional, 2% whale)
- Начальное количество проектов: 10 (распределение: 60% standard, 30% premium, 10% enterprise)
- Фиксированное количество новых проектов в месяц:
  - 1-й месяц: 3 проекта
  - с 3-го месяца: 5 проектов
  - с 6-го месяца: 8 проектов
  - с 12-го месяца: 10 проектов
  - с 18-го месяца: 15 проектов
- Новые пользователи привлекаются проектами:
  - Standard проекты: 15 пользователей
  - Premium проекты: 30 пользователей
  - Enterprise проекты: 50 пользователей
- Отток пользователей: 5% в месяц
- Отток проектов: 8% в месяц
- Коэффициент конверсии рефералов: 10%
- Среднее количество челленджей на проект: 1.5

## Visualizations

The simulation generates various visualizations including:

1. Monthly financial metrics (revenue, costs, profit)
2. User growth by trader type
3. Project growth by subscription tier
4. Average earnings per trader type
5. Revenue breakdown by subscription tier
6. Reward pool distribution
7. Sensitivity analysis on key parameters
8. Market shock impact analysis

## Scenario Analysis

The simulation includes several scenario analyses:

1. **Base Scenario**: Default parameters
2. **Sensitivity Analysis**: Varying key parameters to test impact
3. **Market Shock Scenario**: Simulating a market crash and recovery
4. **Alternative Business Model**: Testing different reward pool allocations

## Requirements

- Python 3.6+
- NumPy
- Pandas
- Matplotlib
- Seaborn

## License

MIT
>>>>>>> 296a3be (Initial commit)
