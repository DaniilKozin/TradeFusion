import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from trade_fusion_simulation import TradeFusionSimulation, TraderType, SubscriptionTier, RewardType

# Настройка страницы
st.set_page_config(page_title="TradeFusion Simulator", layout="wide")

# Заголовок приложения
st.title("TradeFusion Simulator")
st.markdown("Интерактивная модель для симуляции и анализа платформы TradeFusion")

# Создание вкладок
tab1, tab2, tab3, tab4 = st.tabs(["Параметры симуляции", "Результаты", "Анализ чувствительности", "Импакт на проекты"])

# Вкладка с параметрами симуляции
with tab1:
    st.header("Настройка параметров симуляции")
    
    # Создание колонок для лучшей организации интерфейса
    col1, col2 = st.columns(2)
    
    # Добавляем выбор сценария симуляции
    scenario = st.radio(
        "Выберите сценарий симуляции",
        ["Нейтральный", "Оптимистичный", "Пессимистичный"],
        horizontal=True
    )
    
    with col1:
        st.subheader("Основные параметры")
        simulation_months = st.slider("Длительность симуляции (месяцы)", 6, 60, 24)
        
        # Устанавливаем значения по умолчанию в зависимости от сценария
        if scenario == "Оптимистичный":
            default_traders = 60
            default_projects = 2
            default_user_growth = 0.18
            default_user_churn = 0.04
            default_project_churn = 0.06
            default_referral_rate = 0.12
        elif scenario == "Пессимистичный":
            default_traders = 40
            default_projects = 1
            default_user_growth = 0.10
            default_user_churn = 0.08
            default_project_churn = 0.12
            default_referral_rate = 0.07
        else:  # Нейтральный
            default_traders = 50
            default_projects = 1
            default_user_growth = 0.15
            default_user_churn = 0.06
            default_project_churn = 0.08
            default_referral_rate = 0.10
        
        initial_traders = st.number_input("Начальное количество трейдеров", 10, 1000, default_traders)
        initial_projects = st.number_input("Начальное количество проектов", 1, 10, default_projects)
        
        st.subheader("Параметры роста")
        monthly_user_growth_rate = st.slider("Месячный темп роста пользователей", 0.01, 0.50, default_user_growth, 0.01, format="%.2f")
        user_churn_rate = st.slider("Отток пользователей", 0.01, 0.30, default_user_churn, 0.01, format="%.2f", help="Процент пользователей, которые перестают пользоваться платформой каждый месяц")
        project_churn_rate = st.slider("Отток проектов", 0.01, 0.30, default_project_churn, 0.01, format="%.2f", help="Процент проектов, которые перестают пользоваться платформой каждый месяц")
        referral_conversion_rate = st.slider("Конверсия рефералов", 0.01, 0.50, default_referral_rate, 0.01, format="%.2f")
        
        st.subheader("Новые проекты в месяц")
        # Уменьшаем количество новых проектов согласно требованиям
        new_projects_month_1 = st.number_input("Новые проекты в 1-й месяц", 0, 5, 1, 1)
        new_projects_month_3 = st.number_input("Новые проекты с 3-го месяца", 0, 10, 1, 1)
        new_projects_month_6 = st.number_input("Новые проекты с 6-го месяца", 0, 15, 2, 1)
        new_projects_month_12 = st.number_input("Новые проекты с 12-го месяца", 0, 20, 3, 1)
        new_projects_month_18 = st.number_input("Новые проекты с 18-го месяца", 0, 25, 4, 1)
    
    with col2:
        st.subheader("Финансовые параметры")
        # Устанавливаем значения по умолчанию в зависимости от сценария
        if scenario == "Оптимистичный":
            default_dev_cost = 50000
            default_monthly_cost = 10000
            default_marketing_pct = 0.18
            default_cac_retail = 45
            default_cac_b2b = 900
            default_marketing_efficiency = 0.9
        elif scenario == "Пессимистичный":
            default_dev_cost = 60000
            default_monthly_cost = 12000
            default_marketing_pct = 0.12
            default_cac_retail = 60
            default_cac_b2b = 1200
            default_marketing_efficiency = 1.2
        else:  # Нейтральный
            default_dev_cost = 50000
            default_monthly_cost = 10000
            default_marketing_pct = 0.15
            default_cac_retail = 50
            default_cac_b2b = 1000
            default_marketing_efficiency = 1.0
            
        development_cost = st.number_input("Стоимость разработки ($)", 10000, 1000000, default_dev_cost, 5000, help="Первоначальные затраты на разработку платформы")
        monthly_operational_cost = st.number_input("Ежемесячные операционные расходы ($)", 5000, 100000, default_monthly_cost, 1000, help="Ежемесячные затраты на поддержку платформы, включая зарплаты, серверы и т.д.")
        marketing_cost_percentage = st.slider("Процент расходов на маркетинг", 0.05, 0.50, default_marketing_pct, 0.01, format="%.2f")
        
        # Добавляем метрику CAC (стоимость привлечения клиента)
        st.subheader("Метрики эффективности маркетинга")
        base_cac_retail = st.number_input("Базовая стоимость привлечения розничного пользователя ($)", 5, 1000, default_cac_retail, 5)
        base_cac_b2b = st.number_input("Базовая стоимость привлечения B2B клиента ($)", 100, 10000, default_cac_b2b, 100)
        marketing_efficiency = st.slider("Эффективность маркетинговых расходов", 0.5, 2.0, default_marketing_efficiency, 0.1, format="%.1f", 
                                      help="Коэффициент, показывающий как увеличение/уменьшение расходов влияет на CAC. Значение < 1 означает повышение эффективности, > 1 - снижение.")
        
        st.subheader("Параметры вознаграждений")
        reward_pool_allocation = st.slider("Распределение пула вознаграждений", 0.1, 0.8, 0.4, 0.05, format="%.2f")
        referral_commission_percentage = st.slider("Процент комиссии за рефералов", 0.05, 0.30, 0.10, 0.01, format="%.2f")
        
        st.subheader("Параметры челленджей")
        average_challenges_per_project = st.slider("Среднее количество челленджей на проект", 0.5, 5.0, 1.5, 0.1, format="%.1f")
    
    # Настройка цен подписки
    st.subheader("Цены подписки")
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        standard_fee = st.number_input("Standard подписка ($)", 500, 5000, 2000, 100)
    with col_s2:
        premium_fee = st.number_input("Premium подписка ($)", 1000, 10000, 5000, 100)
    with col_s3:
        enterprise_fee = st.number_input("Enterprise подписка ($)", 5000, 20000, 10000, 100)
    
    # Настройка распределения вознаграждений
    st.subheader("Распределение вознаграждений")
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    with col_r1:
        volume_leaderboard_pct = st.slider("Лидерборд объема (%)", 10, 80, 55, 5)
    with col_r2:
        random_draw_pct = st.slider("Случайные розыгрыши (%)", 5, 40, 15, 5)
    with col_r3:
        milestone_pct = st.slider("Достижение целей (%)", 5, 50, 25, 5)
    with col_r4:
        special_event_pct = st.slider("Специальные события (%)", 1, 20, 5, 1)
    
    # Проверка, что сумма процентов равна 100
    total_reward_pct = volume_leaderboard_pct + random_draw_pct + milestone_pct + special_event_pct
    if total_reward_pct != 100:
        st.warning(f"Сумма процентов распределения вознаграждений должна быть равна 100%. Текущая сумма: {total_reward_pct}%")
    
    # Распределение типов трейдеров
    st.subheader("Распределение типов трейдеров")
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    with col_t1:
        casual_traders_pct = st.slider("Casual трейдеры (%)", 10, 90, 70, 5)
    with col_t2:
        active_traders_pct = st.slider("Active трейдеры (%)", 5, 50, 20, 5)
    with col_t3:
        professional_traders_pct = st.slider("Professional трейдеры (%)", 1, 30, 8, 1)
    with col_t4:
        whale_traders_pct = st.slider("Whale трейдеры (%)", 1, 20, 2, 1)
    
    # Проверка, что сумма процентов равна 100
    total_trader_pct = casual_traders_pct + active_traders_pct + professional_traders_pct + whale_traders_pct
    if total_trader_pct != 100:
        st.warning(f"Сумма процентов типов трейдеров должна быть равна 100%. Текущая сумма: {total_trader_pct}%")
    
    # Распределение типов проектов
    st.subheader("Распределение типов проектов")
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        standard_projects_pct = st.slider("Standard проекты (%)", 10, 90, 60, 5)
    with col_p2:
        premium_projects_pct = st.slider("Premium проекты (%)", 5, 70, 30, 5)
    with col_p3:
        enterprise_projects_pct = st.slider("Enterprise проекты (%)", 5, 50, 10, 5)
    
    # Проверка, что сумма процентов равна 100
    total_project_pct = standard_projects_pct + premium_projects_pct + enterprise_projects_pct
    if total_project_pct != 100:
        st.warning(f"Сумма процентов типов проектов должна быть равна 100%. Текущая сумма: {total_project_pct}%")
    
    # Инициализация состояния для автообновления
    if 'auto_update_enabled' not in st.session_state:
        st.session_state.auto_update_enabled = False
    
    # Добавляем опцию для автоматического обновления при изменении параметров
    auto_update = st.checkbox("Автоматически обновлять при изменении параметров", value=st.session_state.auto_update_enabled, key="auto_update")
    
    # Обновляем состояние автообновления
    if auto_update != st.session_state.auto_update_enabled:
        st.session_state.auto_update_enabled = auto_update
        st.session_state.params_changed = True
    
    # Кнопка для запуска симуляции
    run_simulation = st.button("Запустить симуляцию", type="primary")

# Функция для запуска симуляции с заданными параметрами
def run_simulation_with_params(params):
    simulation = TradeFusionSimulation(params)
    simulation.run_simulation()
    return simulation

# Функция для проверки изменения параметров
def params_changed(new_params):
    if 'last_params' not in st.session_state:
        return True
    return new_params != st.session_state.last_params

# Собираем параметры из интерфейса
params = {
    # Simulation parameters
    'simulation_months': simulation_months,
    
    # Initial state parameters
    'initial_traders': initial_traders,
    'initial_projects': initial_projects,
    
    # Growth parameters
    'monthly_user_growth_rate': monthly_user_growth_rate,
    'user_churn_rate': user_churn_rate,
    'project_churn_rate': project_churn_rate,
    'referral_conversion_rate': referral_conversion_rate,
    
    # Новые проекты в месяц
    'month_to_projects': {
        1: new_projects_month_1,
        3: new_projects_month_3,
        6: new_projects_month_6,
        12: new_projects_month_12,
        18: new_projects_month_18
    },
    
    # Challenge parameters
    'average_challenges_per_project': average_challenges_per_project,
    
    # Financial parameters
    'development_cost': development_cost,
    'monthly_operational_cost': monthly_operational_cost,
    'marketing_cost_percentage': marketing_cost_percentage,
    'reward_pool_allocation': reward_pool_allocation,
    'referral_commission_percentage': referral_commission_percentage,
    
    # Маркетинговые метрики
    'base_cac_retail': base_cac_retail,
    'base_cac_b2b': base_cac_b2b,
    'marketing_efficiency': marketing_efficiency,
    
    # Subscription fees
    'subscription_fees': {
        SubscriptionTier.STANDARD: standard_fee,
        SubscriptionTier.PREMIUM: premium_fee,
        SubscriptionTier.ENTERPRISE: enterprise_fee
    },
    
    # Reward distribution
    'reward_distribution': {
        RewardType.VOLUME_LEADERBOARD: volume_leaderboard_pct / 100,
        RewardType.RANDOM_DRAW: random_draw_pct / 100,
        RewardType.MILESTONE: milestone_pct / 100,
        RewardType.SPECIAL_EVENT: special_event_pct / 100
    },
    
    # Trader distribution
    'trader_distribution': {
        TraderType.CASUAL: casual_traders_pct / 100,
        TraderType.ACTIVE: active_traders_pct / 100,
        TraderType.PROFESSIONAL: professional_traders_pct / 100,
        TraderType.WHALE: whale_traders_pct / 100
    },
    
    # Project distribution
    'project_distribution': {
        SubscriptionTier.STANDARD: standard_projects_pct / 100,
        SubscriptionTier.PREMIUM: premium_projects_pct / 100,
        SubscriptionTier.ENTERPRISE: enterprise_projects_pct / 100
    }
}

# Запускаем симуляцию если кнопка нажата или включено автообновление и параметры изменились
if run_simulation or (st.session_state.auto_update_enabled and params_changed(params)):
    # Сохраняем текущие параметры для отслеживания изменений
    st.session_state.last_params = params.copy()
    
    # Запускаем симуляцию с прогресс-баром
    with st.spinner('Запуск симуляции...'):
        st.session_state.simulation = run_simulation_with_params(params)
        st.session_state.params = params
    
    st.success('Симуляция успешно завершена!')

# Вкладка с результатами
with tab2:
    if 'simulation' in st.session_state:
        simulation = st.session_state.simulation
        
        st.header("Результаты симуляции")
        
        # Основные метрики
        col1, col2, col3, col4 = st.columns(4)
        
        # Получаем последний месяц данных
        last_month = simulation.monthly_metrics.iloc[-1]
        first_month = simulation.monthly_metrics.iloc[0]
        
        # Рассчитываем изменение показателей с начала симуляции
        traders_change = last_month['active_traders'] - first_month['active_traders']
        traders_change_pct = (traders_change / first_month['active_traders']) * 100 if first_month['active_traders'] > 0 else 0
        
        projects_change = last_month['active_projects'] - first_month['active_projects']
        projects_change_pct = (projects_change / first_month['active_projects']) * 100 if first_month['active_projects'] > 0 else 0
        
        # Отображаем метрики с изменениями
        with col1:
            st.metric(
                "Активные трейдеры", 
                f"{int(last_month['active_traders']):,}", 
                f"{int(traders_change):+,} ({traders_change_pct:.1f}%)"
            )
        with col2:
            st.metric(
                "Активные проекты", 
                f"{int(last_month['active_projects']):,}",
                f"{int(projects_change):+,} ({projects_change_pct:.1f}%)"
            )
        with col3:
            st.metric("Месячная выручка", f"${int(last_month['monthly_revenue']):,}")
        with col4:
            st.metric("Месячная прибыль", f"${int(last_month['monthly_profit']):,}")
        
        # Добавляем информацию об оттоке
        st.subheader("Статистика оттока")
        col_churn1, col_churn2, col_churn3 = st.columns(3)
        
        # Рассчитываем средний отток за всю симуляцию
        active_traders = simulation.monthly_metrics['active_traders'].values
        trader_churn_rates = []
        for i in range(1, len(active_traders)):
            if active_traders[i-1] > 0:
                diff = active_traders[i-1] - active_traders[i]
                if diff > 0:  # Был отток
                    churn_rate = diff / active_traders[i-1]
                    trader_churn_rates.append(churn_rate)
        
        avg_trader_churn = sum(trader_churn_rates) / len(trader_churn_rates) if trader_churn_rates else 0
        
        # Аналогично для проектов
        active_projects = simulation.monthly_metrics['active_projects'].values
        project_churn_rates = []
        for i in range(1, len(active_projects)):
            if active_projects[i-1] > 0:
                diff = active_projects[i-1] - active_projects[i]
                if diff > 0:  # Был отток
                    churn_rate = diff / active_projects[i-1]
                    project_churn_rates.append(churn_rate)
        
        avg_project_churn = sum(project_churn_rates) / len(project_churn_rates) if project_churn_rates else 0
        
        with col_churn1:
            st.metric("Средний отток трейдеров", f"{avg_trader_churn:.1%}")
        with col_churn2:
            st.metric("Средний отток проектов", f"{avg_project_churn:.1%}")
        with col_churn3:
            # Рассчитываем средний LTV (Lifetime Value) трейдера
            avg_lifetime = sum(simulation.monthly_metrics['active_traders']) / len(simulation.monthly_metrics)
            avg_revenue_per_trader = sum(simulation.monthly_metrics['monthly_revenue']) / sum(simulation.monthly_metrics['active_traders'])
            avg_ltv = avg_lifetime * avg_revenue_per_trader / avg_trader_churn if avg_trader_churn > 0 else 0
            st.metric("Средний LTV трейдера", f"${avg_ltv:.2f}")
        
        # Точка безубыточности
        break_even_month = simulation.calculate_break_even_point()
        if break_even_month:
            st.success(f"Точка безубыточности достигнута в месяце {break_even_month}")
        else:
            st.error("Платформа не достигает точки безубыточности в течение периода симуляции")
        
        # Добавляем вкладки для разных типов графиков
        result_tabs = st.tabs(["Финансовые показатели", "Рост пользователей", "Рост проектов", "Распределение доходов"])
        
        with result_tabs[0]:
            # Графики финансовых показателей
            st.subheader("Финансовые показатели")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(simulation.monthly_metrics['month'], simulation.monthly_metrics['monthly_revenue'], 'b-', label='Месячная выручка')
            ax.plot(simulation.monthly_metrics['month'], simulation.monthly_metrics['monthly_profit'], 'g-', label='Месячная прибыль')
            ax.plot(simulation.monthly_metrics['month'], simulation.monthly_metrics['cumulative_profit'], 'r-', label='Накопленная прибыль')
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.set_xlabel('Месяц')
            ax.set_ylabel('Сумма ($)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Добавляем информацию о метриках CAC
            st.subheader("Метрики эффективности маркетинга")
            col_m1, col_m2, col_m3 = st.columns(3)
            
            # Рассчитываем текущий CAC на основе параметров
            base_cac_retail = st.session_state.params.get('base_cac_retail', 50)
            base_cac_b2b = st.session_state.params.get('base_cac_b2b', 1000)
            marketing_efficiency = st.session_state.params.get('marketing_efficiency', 1.0)
            marketing_percentage = st.session_state.params.get('marketing_cost_percentage', 0.15)
            marketing_multiplier = (marketing_percentage / 0.15) ** marketing_efficiency
            
            current_cac_retail = base_cac_retail / marketing_multiplier
            current_cac_b2b = base_cac_b2b / marketing_multiplier
            
            with col_m1:
                st.metric("CAC розничных пользователей", f"${current_cac_retail:.2f}")
            with col_m2:
                st.metric("CAC B2B клиентов", f"${current_cac_b2b:.2f}")
            with col_m3:
                marketing_budget = st.session_state.params.get('monthly_operational_cost', 30000) * marketing_percentage
                st.metric("Месячный бюджет маркетинга", f"${marketing_budget:.2f}")
            
            # Добавляем информацию о влиянии маркетинга на привлечение пользователей
            st.info(
                "**Влияние маркетинга на привлечение пользователей:**\n\n"
                f"При текущем бюджете маркетинга (${marketing_budget:.2f}) и эффективности ({marketing_efficiency:.1f}), "
                f"платформа может привлечь примерно {int(marketing_budget * 0.8 / current_cac_retail)} розничных пользователей "
                f"и {int(marketing_budget * 0.2 / current_cac_b2b)} B2B клиентов в месяц через маркетинговые кампании.\n\n"
                "Реферальная программа дополнительно привлекает пользователей без прямых маркетинговых затрат."
            )
            
            # Таблица с финансовыми показателями
            st.subheader("Финансовые показатели по месяцам")
            financial_data = simulation.monthly_metrics[['month', 'monthly_revenue', 'monthly_profit', 'cumulative_profit']].copy()
            financial_data.columns = ['Месяц', 'Выручка ($)', 'Прибыль ($)', 'Накопленная прибыль ($)']
            st.dataframe(financial_data.style.format({'Выручка ($)': '${:,.2f}', 'Прибыль ($)': '${:,.2f}', 'Накопленная прибыль ($)': '${:,.2f}'}), use_container_width=True)
        
        with result_tabs[1]:
            # Графики роста пользователей
            st.subheader("Рост пользователей")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(simulation.monthly_metrics['month'], simulation.monthly_metrics['active_traders'], 'b-', label='Активные трейдеры')
            
            # Рассчитываем изменение количества активных трейдеров между месяцами
            # для отображения динамики вместо отсутствующих столбцов new_traders и churned_traders
            active_traders = simulation.monthly_metrics['active_traders'].values
            months = simulation.monthly_metrics['month'].values
            
            # Создаем массивы для хранения данных о приросте и оттоке
            trader_growth = np.zeros_like(active_traders)
            trader_churn = np.zeros_like(active_traders)
            
            # Заполняем массивы данными о приросте и оттоке
            for i in range(1, len(active_traders)):
                diff = active_traders[i] - active_traders[i-1]
                if diff > 0:
                    trader_growth[i] = diff  # Прирост
                else:
                    trader_churn[i] = -diff  # Отток (преобразуем в положительное число)
            
            # Строим графики прироста и оттока
            ax.plot(months, trader_growth, 'g-', label='Прирост трейдеров')
            ax.plot(months, trader_churn, 'r-', label='Отток трейдеров')
            
            ax.set_xlabel('Месяц')
            ax.set_ylabel('Количество трейдеров')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Распределение типов трейдеров
            st.subheader("Распределение типов трейдеров")
            
            # Создаем DataFrame с данными о типах трейдеров из последнего месяца
            trader_types_data = []
            for trader_type in TraderType:
                type_name = trader_type.value
                count = last_month[f'{type_name}_count'] if f'{type_name}_count' in last_month else 0
                trader_types_data.append({'type': type_name, 'count': count})
            
            trader_types = pd.DataFrame(trader_types_data)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='type', y='count', data=trader_types, ax=ax)
            ax.set_xlabel('Тип трейдера')
            ax.set_ylabel('Количество')
            ax.set_title('Распределение трейдеров по типам')
            st.pyplot(fig)
        
        with result_tabs[2]:
            # Графики роста проектов
            st.subheader("Рост проектов")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(simulation.monthly_metrics['month'], simulation.monthly_metrics['active_projects'], 'b-', label='Активные проекты')
            
            # Рассчитываем изменение количества активных проектов между месяцами
            # для отображения динамики вместо отсутствующих столбцов new_projects и churned_projects
            active_projects = simulation.monthly_metrics['active_projects'].values
            months = simulation.monthly_metrics['month'].values
            
            # Создаем массивы для хранения данных о приросте и оттоке
            project_growth = np.zeros_like(active_projects)
            project_churn = np.zeros_like(active_projects)
            
            # Заполняем массивы данными о приросте и оттоке
            for i in range(1, len(active_projects)):
                diff = active_projects[i] - active_projects[i-1]
                if diff > 0:
                    project_growth[i] = diff  # Прирост
                else:
                    project_churn[i] = -diff  # Отток (преобразуем в положительное число)
            
            # Строим графики прироста и оттока
            ax.plot(months, project_growth, 'g-', label='Прирост проектов')
            ax.plot(months, project_churn, 'r-', label='Отток проектов')
            
            ax.set_xlabel('Месяц')
            ax.set_ylabel('Количество проектов')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Распределение типов проектов
            st.subheader("Распределение типов проектов")
            
            # Создаем DataFrame с данными о типах проектов из последнего месяца
            project_types_data = []
            for tier in SubscriptionTier:
                tier_name = tier.value
                count = last_month[f'{tier_name}_count'] if f'{tier_name}_count' in last_month else 0
                project_types_data.append({'tier': tier_name, 'count': count})
            
            project_types = pd.DataFrame(project_types_data)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='tier', y='count', data=project_types, ax=ax)
            ax.set_xlabel('Тип проекта')
            ax.set_ylabel('Количество')
            ax.set_title('Распределение проектов по типам')
            st.pyplot(fig)
        
        with result_tabs[3]:
            # Распределение доходов
            st.subheader("Распределение доходов")
            
            # Выбор месяца для анализа
            selected_month = st.slider("Выберите месяц для анализа", 1, simulation.max_months, min(12, simulation.max_months), key="revenue_month_selector_tab2")
            
            # Получаем данные за выбранный месяц
            month_data = simulation.monthly_metrics[simulation.monthly_metrics['month'] == selected_month].iloc[0]
            
            # Распределение доходов по типам проектов
            st.subheader(f"Распределение доходов по типам проектов (Месяц {selected_month})")
            revenue_by_tier = simulation.get_revenue_by_tier(selected_month)
            
            if revenue_by_tier is not None and not revenue_by_tier.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                revenue_by_tier.plot(kind='pie', y='revenue', labels=revenue_by_tier['tier'], autopct='%1.1f%%', ax=ax)
                ax.set_ylabel('')
                ax.set_title(f'Распределение доходов по типам проектов (Месяц {selected_month})')
                st.pyplot(fig)
            else:
                st.info(f"Нет данных о доходах по типам проектов для месяца {selected_month}")
            
            # Информация о пуле вознаграждений
            st.subheader(f"Пул вознаграждений (Месяц {selected_month})")
            
            # Рассчитываем размер пула вознаграждений для выбранного месяца
            if 'monthly_revenue' in month_data:
                monthly_revenue = month_data['monthly_revenue']
                reward_pool = monthly_revenue * reward_pool_allocation
                
                # Создаем таблицу с информацией о пуле вознаграждений
                reward_pool_data = pd.DataFrame({
                    'Показатель': ['Месячная выручка', 'Процент на пул вознаграждений', 'Размер пула вознаграждений'],
                    'Значение': [f'${monthly_revenue:,.2f}', f'{reward_pool_allocation:.0%}', f'${reward_pool:,.2f}']
                })
                
                st.table(reward_pool_data)
                
                # Рассчитываем распределение пула вознаграждений по типам вознаграждений
                reward_distribution_data = pd.DataFrame({
                    'Тип вознаграждения': [rt.value for rt in RewardType],
                    'Процент': [params['reward_distribution'][rt] * 100 for rt in RewardType],
                    'Сумма': [params['reward_distribution'][rt] * reward_pool for rt in RewardType]
                })
                
                # Форматируем данные для отображения
                reward_distribution_data['Процент'] = reward_distribution_data['Процент'].apply(lambda x: f'{x:.1f}%')
                reward_distribution_data['Сумма'] = reward_distribution_data['Сумма'].apply(lambda x: f'${x:,.2f}')
                
                st.subheader(f"Распределение пула вознаграждений по типам (Месяц {selected_month})")
                st.table(reward_distribution_data)
            
            # Распределение вознаграждений по типам трейдеров
            st.subheader(f"Распределение вознаграждений по типам трейдеров (Месяц {selected_month})")
            
            # Получаем данные о количестве трейдеров каждого типа
            trader_counts = {}
            trader_percentages = {}
            total_traders = 0
            
            for trader_type in TraderType:
                type_name = trader_type.value
                count_key = f'{type_name}_count'
                if count_key in month_data:
                    count = month_data[count_key]
                    trader_counts[type_name] = count
                    total_traders += count
            
            # Рассчитываем процентное соотношение трейдеров
            for trader_type, count in trader_counts.items():
                trader_percentages[trader_type] = count / total_traders if total_traders > 0 else 0
            
            # Оценка среднего вознаграждения на трейдера каждого типа
            # Используем примерное распределение на основе типов трейдеров и их объемов торговли
            volume_weights = {
                'Casual': 1,
                'Active': 10,
                'Professional': 100,
                'Whale': 1000
            }
            
            # Рассчитываем общий взвешенный объем
            total_weighted_volume = sum(volume_weights[t_type] * count for t_type, count in trader_counts.items())
            
            # Рассчитываем долю вознаграждений для каждого типа трейдеров
            reward_shares = {}
            for trader_type, count in trader_counts.items():
                weighted_volume = volume_weights[trader_type] * count
                reward_shares[trader_type] = weighted_volume / total_weighted_volume if total_weighted_volume > 0 else 0
            
            # Рассчитываем общую сумму вознаграждений и среднее на трейдера
            trader_rewards = {}
            avg_rewards = {}
            
            for trader_type, share in reward_shares.items():
                trader_rewards[trader_type] = reward_pool * share
                avg_rewards[trader_type] = trader_rewards[trader_type] / trader_counts[trader_type] if trader_counts[trader_type] > 0 else 0
            
            # Создаем DataFrame для отображения в таблице
            trader_reward_data = pd.DataFrame({
                'Тип трейдера': list(trader_counts.keys()),
                'Количество': list(trader_counts.values()),
                'Процент от общего числа': [f'{p:.1%}' for p in trader_percentages.values()],
                'Доля в пуле вознаграждений': [f'{s:.1%}' for s in reward_shares.values()],
                'Сумма вознаграждений': [f'${r:,.2f}' for r in trader_rewards.values()],
                'Среднее на трейдера': [f'${a:,.2f}' for a in avg_rewards.values()]
            })
            
            st.table(trader_reward_data)
            
            # Визуализация распределения вознаграждений по типам трейдеров
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.pie([r for r in trader_rewards.values()], 
                   labels=[t for t in trader_rewards.keys()],
                   autopct='%1.1f%%')
            plt.title(f'Распределение пула вознаграждений по типам трейдеров (Месяц {selected_month})')
            st.pyplot(fig)
            
            # Сравнение среднего вознаграждения на трейдера по типам
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = plt.bar(list(avg_rewards.keys()), list(avg_rewards.values()))
            
            # Добавляем подписи значений над столбцами
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'${height:,.2f}',
                        ha='center', va='bottom', rotation=0)
            
            plt.title(f'Среднее вознаграждение на трейдера по типам (Месяц {selected_month})')
            plt.ylabel('Среднее вознаграждение ($)')
            plt.xlabel('Тип трейдера')
            st.pyplot(fig)
            
            # Динамика пула вознаграждений по месяцам
            st.subheader("Динамика пула вознаграждений по месяцам")
            
            # Создаем DataFrame с данными о пуле вознаграждений по месяцам
            reward_pool_monthly = simulation.monthly_metrics[['month', 'monthly_revenue']].copy()
            reward_pool_monthly['reward_pool'] = reward_pool_monthly['monthly_revenue'] * reward_pool_allocation
            
            # Таблица с данными о пуле вознаграждений по месяцам
            reward_pool_table = reward_pool_monthly.copy()
            reward_pool_table.columns = ['Месяц', 'Выручка ($)', 'Пул вознаграждений ($)']
            reward_pool_table['Выручка ($)'] = reward_pool_table['Выручка ($)'].apply(lambda x: f'${x:,.2f}')
            reward_pool_table['Пул вознаграждений ($)'] = reward_pool_table['Пул вознаграждений ($)'].apply(lambda x: f'${x:,.2f}')
            
            st.dataframe(reward_pool_table, use_container_width=True)
            
            # График динамики пула вознаграждений
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(reward_pool_monthly['month'], reward_pool_monthly['reward_pool'], 'b-', marker='o')
            ax.set_xlabel('Месяц')
            ax.set_ylabel('Пул вознаграждений ($)')
            ax.set_title('Динамика пула вознаграждений по месяцам')
            ax.grid(True, alpha=0.3)
            
            # Добавляем подписи значений на график
            for i, value in enumerate(reward_pool_monthly['reward_pool']):
                if i % 3 == 0:  # Показываем каждую третью точку, чтобы не перегружать график
                    ax.text(reward_pool_monthly['month'].iloc[i], value, f'${value:,.0f}', 
                           ha='center', va='bottom', fontsize=8)
            
            st.pyplot(fig)
            
            # Таблица с помесячным распределением вознаграждений по типам трейдеров
            st.subheader("Помесячное распределение вознаграждений по типам трейдеров")
            
            # Создаем DataFrame для хранения данных о распределении вознаграждений по месяцам
            monthly_trader_rewards = []
            
            # Проходим по всем месяцам симуляции
            for month in range(1, simulation.max_months + 1):
                # Получаем данные за текущий месяц
                month_data = simulation.monthly_metrics[simulation.monthly_metrics['month'] == month]
                
                if not month_data.empty:
                    month_data = month_data.iloc[0]
                    monthly_revenue = month_data['monthly_revenue']
                    reward_pool = monthly_revenue * reward_pool_allocation
                    
                    # Получаем данные о количестве трейдеров каждого типа
                    trader_counts = {}
                    total_traders = 0
                    
                    for trader_type in TraderType:
                        type_name = trader_type.value
                        count_key = f'{type_name}_count'
                        if count_key in month_data:
                            count = month_data[count_key]
                            trader_counts[type_name] = count
                            total_traders += count
                    
                    # Рассчитываем распределение вознаграждений по типам трейдеров
                    # Используем те же весовые коэффициенты, что и раньше
                    total_weighted_volume = sum(volume_weights[t_type] * count for t_type, count in trader_counts.items() if t_type in volume_weights)
                    
                    for trader_type, count in trader_counts.items():
                        if trader_type in volume_weights and total_weighted_volume > 0:
                            weighted_volume = volume_weights[trader_type] * count
                            reward_share = weighted_volume / total_weighted_volume
                            trader_reward = reward_pool * reward_share
                            avg_reward = trader_reward / count if count > 0 else 0
                            
                            monthly_trader_rewards.append({
                                'Месяц': month,
                                'Тип трейдера': trader_type,
                                'Количество': count,
                                'Доля в пуле': reward_share,
                                'Сумма вознаграждений': trader_reward,
                                'Среднее на трейдера': avg_reward
                            })
            
            # Создаем DataFrame из собранных данных
            monthly_rewards_df = pd.DataFrame(monthly_trader_rewards)
            
            # Создаем сводную таблицу для отображения
            if not monthly_rewards_df.empty:
                # Выбираем месяцы для отображения (например, каждый 3-й месяц)
                months_to_show = sorted(monthly_rewards_df['Месяц'].unique())
                months_to_show = [m for i, m in enumerate(months_to_show) if i % 3 == 0 or m == selected_month]
                
                # Фильтруем данные по выбранным месяцам
                filtered_rewards = monthly_rewards_df[monthly_rewards_df['Месяц'].isin(months_to_show)]
                
                # Создаем сводную таблицу
                pivot_rewards = filtered_rewards.pivot_table(
                    index='Тип трейдера',
                    columns='Месяц',
                    values='Среднее на трейдера',
                    aggfunc='mean'
                )
                
                # Форматируем значения в таблице
                formatted_pivot = pivot_rewards.applymap(lambda x: f'${x:,.2f}')
                
                st.subheader("Среднее вознаграждение на трейдера по месяцам")
                st.table(formatted_pivot)
                
                # Создаем график изменения среднего вознаграждения по типам трейдеров
                plt.figure(figsize=(12, 6))
                
                for trader_type in TraderType:
                    type_name = trader_type.value
                    type_data = monthly_rewards_df[monthly_rewards_df['Тип трейдера'] == type_name]
                    if not type_data.empty:
                        plt.plot(type_data['Месяц'], type_data['Среднее на трейдера'], marker='o', label=type_name)
                
                plt.xlabel('Месяц')
                plt.ylabel('Среднее вознаграждение ($)')
                plt.title('Динамика среднего вознаграждения по типам трейдеров')
                plt.grid(True, alpha=0.3)
                plt.legend()
                st.pyplot(plt)
            
            # Распределение вознаграждений по типам
            st.subheader(f"Распределение вознаграждений по типам (Месяц {selected_month})")
            rewards_by_type = simulation.get_rewards_by_type(selected_month)
            
            if rewards_by_type is not None and not rewards_by_type.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                rewards_by_type.plot(kind='pie', y='amount', labels=rewards_by_type['reward_type'], autopct='%1.1f%%', ax=ax)
                ax.set_ylabel('')
                ax.set_title(f'Распределение вознаграждений по типам (Месяц {selected_month})')
                st.pyplot(fig)
            else:
                st.info(f"Нет данных о вознаграждениях для месяца {selected_month}")
    else:
        st.info("Запустите симуляцию на вкладке 'Параметры симуляции', чтобы увидеть результаты")

# Вкладка с импактом на проекты
with tab4:
    st.header("Влияние на проекты")
    
    if 'simulation' not in st.session_state:
        st.info("Запустите симуляцию на вкладке 'Параметры симуляции', чтобы увидеть влияние на проекты")
    else:
        simulation = st.session_state.simulation
        
        # Выбор месяца для анализа
        selected_month = st.slider("Выберите месяц для анализа", 1, simulation.max_months, min(12, simulation.max_months), key="impact_month_selector_tab4")
        
        # Получаем данные о проектах за выбранный месяц
        project_data = simulation.get_project_data_for_month(selected_month) if hasattr(simulation, 'get_project_data_for_month') else None
        
        if project_data is None:
            # Если метод не реализован, создаем примерные данные на основе имеющейся информации
            month_metrics = simulation.monthly_metrics[simulation.monthly_metrics['month'] == selected_month]
            
            if not month_metrics.empty:
                month_data = month_metrics.iloc[0]
                
                # Основные метрики по проектам
                st.subheader("Основные метрики по проектам")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Активные проекты", f"{int(month_data['active_projects']):,}")
                
                # Рассчитываем примерное распределение проектов по типам
                project_distribution = st.session_state.params.get('project_distribution', {
                    SubscriptionTier.STANDARD: 0.6,
                    SubscriptionTier.PREMIUM: 0.3,
                    SubscriptionTier.ENTERPRISE: 0.1
                })
                
                # Рассчитываем количество проектов каждого типа
                standard_count = int(month_data['active_projects'] * project_distribution[SubscriptionTier.STANDARD])
                premium_count = int(month_data['active_projects'] * project_distribution[SubscriptionTier.PREMIUM])
                enterprise_count = int(month_data['active_projects'] * project_distribution[SubscriptionTier.ENTERPRISE])
                
                with col2:
                    st.metric("Новые проекты в этом месяце", f"{int(month_data.get('new_projects', 0)):,}")
                
                with col3:
                    # Рассчитываем примерный отток проектов
                    churn_rate = st.session_state.params.get('project_churn_rate', 0.08)
                    churned_projects = int(month_data['active_projects'] * churn_rate)
                    st.metric("Отток проектов", f"{churned_projects:,}", f"{churn_rate:.1%}")
                
                # Таблица с распределением проектов по типам
                st.subheader("Распределение проектов по типам")
                
                project_types_data = pd.DataFrame({
                    'Тип проекта': ['Standard', 'Premium', 'Enterprise', 'Всего'],
                    'Количество': [standard_count, premium_count, enterprise_count, standard_count + premium_count + enterprise_count],
                    'Процент': [
                        f"{project_distribution[SubscriptionTier.STANDARD]:.1%}",
                        f"{project_distribution[SubscriptionTier.PREMIUM]:.1%}",
                        f"{project_distribution[SubscriptionTier.ENTERPRISE]:.1%}",
                        "100.0%"
                    ]
                })
                
                st.table(project_types_data)
                
                # Визуализация распределения проектов по типам
                fig, ax = plt.subplots(figsize=(8, 8))
                plt.pie([standard_count, premium_count, enterprise_count], 
                       labels=['Standard', 'Premium', 'Enterprise'],
                       autopct='%1.1f%%',
                       colors=['#ff9999','#66b3ff','#99ff99'])
                plt.title(f'Распределение проектов по типам (Месяц {selected_month})')
                st.pyplot(fig)
                
                # Доход от проектов
                st.subheader("Доход от проектов")
                
                # Рассчитываем доход от каждого типа проектов
                subscription_fees = st.session_state.params.get('subscription_fees', {
                    SubscriptionTier.STANDARD: 2000,
                    SubscriptionTier.PREMIUM: 5000,
                    SubscriptionTier.ENTERPRISE: 10000
                })
                
                standard_revenue = standard_count * subscription_fees[SubscriptionTier.STANDARD]
                premium_revenue = premium_count * subscription_fees[SubscriptionTier.PREMIUM]
                enterprise_revenue = enterprise_count * subscription_fees[SubscriptionTier.ENTERPRISE]
                total_revenue = standard_revenue + premium_revenue + enterprise_revenue
                
                project_revenue_data = pd.DataFrame({
                    'Тип проекта': ['Standard', 'Premium', 'Enterprise', 'Всего'],
                    'Количество': [standard_count, premium_count, enterprise_count, standard_count + premium_count + enterprise_count],
                    'Стоимость подписки': [
                        f"${subscription_fees[SubscriptionTier.STANDARD]:,}",
                        f"${subscription_fees[SubscriptionTier.PREMIUM]:,}",
                        f"${subscription_fees[SubscriptionTier.ENTERPRISE]:,}",
                        ""
                    ],
                    'Доход': [
                        f"${standard_revenue:,}",
                        f"${premium_revenue:,}",
                        f"${enterprise_revenue:,}",
                        f"${total_revenue:,}"
                    ],
                    'Доля в доходе': [
                        f"{standard_revenue/total_revenue:.1%}" if total_revenue > 0 else "0.0%",
                        f"{premium_revenue/total_revenue:.1%}" if total_revenue > 0 else "0.0%",
                        f"{enterprise_revenue/total_revenue:.1%}" if total_revenue > 0 else "0.0%",
                        "100.0%"
                    ]
                })
                
                st.table(project_revenue_data)
                
                # Визуализация распределения дохода по типам проектов
                fig, ax = plt.subplots(figsize=(8, 8))
                plt.pie([standard_revenue, premium_revenue, enterprise_revenue], 
                       labels=['Standard', 'Premium', 'Enterprise'],
                       autopct='%1.1f%%',
                       colors=['#ff9999','#66b3ff','#99ff99'])
                plt.title(f'Распределение дохода по типам проектов (Месяц {selected_month})')
                st.pyplot(fig)
                
                # Динамика роста проектов
                st.subheader("Динамика роста проектов")
                
                # График роста проектов по месяцам
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(simulation.monthly_metrics['month'], simulation.monthly_metrics['active_projects'], 'b-', marker='o')
                ax.set_xlabel('Месяц')
                ax.set_ylabel('Количество проектов')
                ax.set_title('Динамика роста проектов')
                ax.grid(True, alpha=0.3)
                
                # Добавляем подписи значений на график
                for i, value in enumerate(simulation.monthly_metrics['active_projects']):
                    if i % 3 == 0:  # Показываем каждую третью точку, чтобы не перегружать график
                        ax.text(simulation.monthly_metrics['month'].iloc[i], value, f'{int(value):,}', 
                               ha='center', va='bottom', fontsize=8)
                
                st.pyplot(fig)
                
                # Анализ влияния проектов на привлечение трейдеров и объемы торговли
                st.subheader("Влияние проектов на привлечение трейдеров и объемы торговли")
                
                # Создаем данные о влиянии проектов на привлечение трейдеров и объемы торговли
                trader_impact = {
                    SubscriptionTier.STANDARD: {
                        'traders_per_project': 15,
                        'avg_volume_per_trader': 10000
                    },
                    SubscriptionTier.PREMIUM: {
                        'traders_per_project': 30,
                        'avg_volume_per_trader': 50000
                    },
                    SubscriptionTier.ENTERPRISE: {
                        'traders_per_project': 50,
                        'avg_volume_per_trader': 200000
                    }
                }
                
                # Рассчитываем влияние на привлечение трейдеров и объемы торговли
                project_impact_data = []
                total_traders = 0
                total_volume = 0
                
                for tier in [SubscriptionTier.STANDARD, SubscriptionTier.PREMIUM, SubscriptionTier.ENTERPRISE]:
                    tier_name = tier.value
                    count = 0
                    if tier == SubscriptionTier.STANDARD:
                        count = standard_count
                    elif tier == SubscriptionTier.PREMIUM:
                        count = premium_count
                    elif tier == SubscriptionTier.ENTERPRISE:
                        count = enterprise_count
                    
                    traders_attracted = count * trader_impact[tier]['traders_per_project']
                    volume_impact = traders_attracted * trader_impact[tier]['avg_volume_per_trader']
                    
                    total_traders += traders_attracted
                    total_volume += volume_impact
                    
                    project_impact_data.append({
                        'Тип проекта': tier_name,
                        'Количество проектов': count,
                        'Среднее количество привлеченных трейдеров': trader_impact[tier]['traders_per_project'],
                        'Общее количество привлеченных трейдеров': traders_attracted,
                        'Средний объем на трейдера': trader_impact[tier]['avg_volume_per_trader'],
                        'Общий объем торговли': volume_impact
                    })
                
                # Добавляем строку с итогами
                project_impact_data.append({
                    'Тип проекта': 'Всего',
                    'Количество проектов': standard_count + premium_count + enterprise_count,
                    'Среднее количество привлеченных трейдеров': '-',
                    'Общее количество привлеченных трейдеров': total_traders,
                    'Средний объем на трейдера': '-',
                    'Общий объем торговли': total_volume
                })
                
                # Создаем DataFrame
                impact_df = pd.DataFrame(project_impact_data)
                
                # Форматируем числовые столбцы
                for col in ['Общее количество привлеченных трейдеров', 'Общий объем торговли']:
                    impact_df[col] = impact_df[col].apply(lambda x: f"{int(x):,}" if isinstance(x, (int, float)) else x)
                
                # Добавляем столбец с долей в привлечении трейдеров
                trader_percentages = []
                volume_percentages = []
                
                for i, row in enumerate(project_impact_data):
                    if i < len(project_impact_data) - 1:  # Исключаем строку с итогами
                        trader_pct = row['Общее количество привлеченных трейдеров'] / total_traders if total_traders > 0 else 0
                        volume_pct = row['Общий объем торговли'] / total_volume if total_volume > 0 else 0
                        trader_percentages.append(f"{trader_pct:.1%}")
                        volume_percentages.append(f"{volume_pct:.1%}")
                    else:
                        trader_percentages.append("100.0%")
                        volume_percentages.append("100.0%")
                
                impact_df['Доля в привлечении трейдеров'] = trader_percentages
                impact_df['Доля в объеме торговли'] = volume_percentages
                
                st.table(impact_df)
                
                # Визуализация влияния проектов на объемы торговли
                st.subheader("Визуализация влияния проектов на объемы торговли")
                
                # Создаем данные для визуализации
                volume_data = []
                for i, row in enumerate(project_impact_data):
                    if i < len(project_impact_data) - 1:  # Исключаем строку с итогами
                        volume_data.append({
                            'Тип проекта': row['Тип проекта'],
                            'Объем торговли': row['Общий объем торговли'] if isinstance(row['Общий объем торговли'], (int, float)) else 0
                        })
                
                # Создаем DataFrame для визуализации
                volume_df = pd.DataFrame(volume_data)
                
                # Создаем график
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = plt.bar(
                    volume_df['Тип проекта'], 
                    volume_df['Объем торговли'],
                    color=['#ff9999','#66b3ff','#99ff99']
                )
                
                # Добавляем подписи значений над столбцами
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f'${int(height):,}',
                            ha='center', va='bottom', rotation=0)
                
                plt.title(f'Влияние проектов на объемы торговли (Месяц {selected_month})')
                plt.ylabel('Объем торговли ($)')
                plt.xlabel('Тип проекта')
                st.pyplot(fig)
                
                # Визуализация влияния проектов на привлечение трейдеров
                fig, ax = plt.subplots(figsize=(10, 6))
                # Создаем данные для визуализации привлечения трейдеров
                traders_data = []
                for i, row in enumerate(project_impact_data):
                    if i < len(project_impact_data) - 1:  # Исключаем строку с итогами
                        traders_data.append({
                            'Тип проекта': row['Тип проекта'],
                            'Общее количество привлеченных трейдеров': row['Общее количество привлеченных трейдеров'] if isinstance(row['Общее количество привлеченных трейдеров'], (int, float)) else 0
                        })
                
                # Создаем DataFrame для визуализации
                traders_df = pd.DataFrame(traders_data)
                
                bars = plt.bar(
                    traders_df['Тип проекта'], 
                    traders_df['Общее количество привлеченных трейдеров'],
                    color=['#ff9999','#66b3ff','#99ff99']
                )
                
                # Добавляем подписи значений над столбцами
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height):,}',
                            ha='center', va='bottom', rotation=0)
                
                plt.title(f'Влияние проектов на привлечение трейдеров (Месяц {selected_month})')
                plt.ylabel('Количество привлеченных трейдеров')
                plt.xlabel('Тип проекта')
                st.pyplot(fig)
                
                # Прогноз роста проектов
                st.subheader("Прогноз роста проектов")
                
                # Создаем простую линейную модель для прогноза роста проектов
                months = simulation.monthly_metrics['month'].values
                projects = simulation.monthly_metrics['active_projects'].values
                
                if len(months) > 1:
                    # Простая линейная регрессия
                    z = np.polyfit(months, projects, 1)
                    p = np.poly1d(z)
                    
                    # Прогноз на следующие 12 месяцев
                    future_months = np.arange(max(months) + 1, max(months) + 13)
                    future_projects = p(future_months)
                    
                    # Создаем DataFrame с прогнозом
                    forecast_data = pd.DataFrame({
                        'Месяц': future_months,
                        'Прогноз количества проектов': future_projects
                    })
                    
                    # Отображаем таблицу с прогнозом
                    st.dataframe(forecast_data.style.format({
                        'Прогноз количества проектов': '{:,.0f}'
                    }), use_container_width=True)
                    
                    # Визуализация прогноза
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(months, projects, 'b-o', label='Фактические данные')
                    ax.plot(future_months, future_projects, 'r--o', label='Прогноз')
                    ax.set_xlabel('Месяц')
                    ax.set_ylabel('Количество проектов')
                    ax.set_title('Прогноз роста проектов')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                else:
                    st.info("Недостаточно данных для построения прогноза")
            else:
                st.info(f"Нет данных для месяца {selected_month}")

# Вкладка с анализом чувствительности
with tab3:
    st.header("Анализ чувствительности")
    
    if 'simulation' not in st.session_state:
        st.info("Запустите симуляцию на вкладке 'Параметры симуляции', чтобы выполнить анализ чувствительности")
    else:
        st.subheader("Выберите параметры для анализа чувствительности")
        
        # Выбор параметров для анализа
        params_to_analyze = st.multiselect(
            "Выберите параметры для анализа",
            ["monthly_user_growth_rate", "user_churn_rate", "reward_pool_allocation", "marketing_cost_percentage"],
            ["monthly_user_growth_rate", "user_churn_rate"]
        )
        
        if params_to_analyze:
            # Настройка диапазонов для каждого параметра
            parameter_ranges = {}
            
            for param in params_to_analyze:
                st.subheader(f"Диапазон для {param}")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    min_val = st.number_input(f"Минимальное значение для {param}", 0.01, 1.0, 0.05, 0.01, format="%.2f")
                with col2:
                    max_val = st.number_input(f"Максимальное значение для {param}", 0.01, 1.0, 0.25, 0.01, format="%.2f")
                with col3:
                    steps = st.number_input(f"Количество шагов для {param}", 3, 10, 5, 1)
                
                parameter_ranges[param] = np.linspace(min_val, max_val, int(steps)).tolist()
            
            # Кнопка для запуска анализа чувствительности
            run_sensitivity = st.button("Запустить анализ чувствительности", type="primary")
            
            if run_sensitivity:
                with st.spinner('Выполнение анализа чувствительности...'):
                    # Используем текущую симуляцию для анализа чувствительности
                    sensitivity_results = st.session_state.simulation.perform_sensitivity_analysis(parameter_ranges)
                    st.session_state.sensitivity_results = sensitivity_results
                
                st.success('Анализ чувствительности успешно завершен!')
            
            # Отображение результатов анализа чувствительности
            if 'sensitivity_results' in st.session_state:
                st.subheader("Результаты анализа чувствительности")
                
                # Отображаем таблицу с результатами
                st.dataframe(st.session_state.sensitivity_results.style.format({
                    'value': '{:.2f}',
                    'monthly_revenue': '${:,.2f}',
                    'monthly_profit': '${:,.2f}',
                    'active_traders': '{:,.0f}',
                    'active_projects': '{:,.0f}'
                }), use_container_width=True)
                
                # Визуализация результатов для каждого параметра
                for param in params_to_analyze:
                    st.subheader(f"Влияние {param} на ключевые метрики")
                    
                    param_data = st.session_state.sensitivity_results[st.session_state.sensitivity_results['parameter'] == param]
                    
                    # График влияния на выручку и прибыль
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(param_data['value'], param_data['monthly_revenue'], 'b-o', label='Месячная выручка')
                    ax.plot(param_data['value'], param_data['monthly_profit'], 'g-o', label='Месячная прибыль')
                    ax.set_xlabel(param)
                    ax.set_ylabel('Сумма ($)')
                    ax.set_title(f'Влияние {param} на финансовые показатели')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    # График влияния на количество трейдеров и проектов
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(param_data['value'], param_data['active_traders'], 'b-o', label='Активные трейдеры')
                    ax.plot(param_data['value'], param_data['active_projects'], 'g-o', label='Активные проекты')
                    ax.set_xlabel(param)
                    ax.set_ylabel('Количество')
                    ax.set_title(f'Влияние {param} на количество трейдеров и проектов')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
        else:
            st.warning("Выберите хотя бы один параметр для анализа чувствительности")

# Добавляем информацию о запуске приложения
st.sidebar.header("О приложении")
st.sidebar.markdown("""
### TradeFusion Simulator

Это интерактивное приложение для моделирования и анализа платформы TradeFusion.

Вы можете настроить различные параметры симуляции и увидеть, как они влияют на рост и финансовые показатели платформы.

#### Инструкция по использованию:
1. Настройте параметры на вкладке "Параметры симуляции"
2. Нажмите кнопку "Запустить симуляцию"
3. Просмотрите результаты на вкладке "Результаты"
4. Выполните анализ чувствительности на вкладке "Анализ чувствительности"
""")

# Добавляем информацию о запуске приложения
st.sidebar.markdown("""
#### Запуск приложения:
```
streamlit run streamlit_app.py
```
""")