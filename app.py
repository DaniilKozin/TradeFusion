import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import base64
from trade_fusion_simulation import TradeFusionSimulation, TraderType, SubscriptionTier
from scenario_manager import save_scenario_ui, load_scenario_ui, compare_scenarios_ui, ScenarioManager

# Настройка страницы
st.set_page_config(page_title="TradeFusion Simulator", layout="wide")

# Заголовок приложения
st.title("TradeFusion Simulator")
st.markdown("Интерактивная модель для симуляции и анализа платформы TradeFusion")

# Создание вкладок
tab1, tab2, tab3, tab4 = st.tabs(["Параметры симуляции", "Результаты", "Анализ чувствительности", "Управление сценариями"])

# Вкладка с параметрами симуляции
with tab1:
    st.header("Настройка параметров симуляции")
    
    # Создание колонок для лучшей организации интерфейса
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Основные параметры")
        simulation_months = st.slider("Длительность симуляции (месяцы)", 6, 60, 24)
        initial_traders = st.number_input("Начальное количество трейдеров", 100, 10000, 1000)
        initial_projects = st.number_input("Начальное количество проектов", 1, 100, 10)
        
        st.subheader("Параметры роста")
        monthly_user_growth_rate = st.slider("Месячный темп роста пользователей", 0.01, 0.50, 0.15, 0.01, format="%.2f")
        monthly_project_growth_rate = st.slider("Месячный темп роста проектов", 0.01, 0.50, 0.10, 0.01, format="%.2f")
        user_churn_rate = st.slider("Отток пользователей", 0.01, 0.30, 0.05, 0.01, format="%.2f")
        project_churn_rate = st.slider("Отток проектов", 0.01, 0.30, 0.08, 0.01, format="%.2f")
        referral_conversion_rate = st.slider("Конверсия рефералов", 0.01, 0.50, 0.10, 0.01, format="%.2f")
    
    with col2:
        st.subheader("Финансовые параметры")
        development_cost = st.number_input("Стоимость разработки ($)", 50000, 1000000, 225000, 25000)
        monthly_operational_cost = st.number_input("Ежемесячные операционные расходы ($)", 5000, 100000, 30000, 1000)
        marketing_cost_percentage = st.slider("Процент расходов на маркетинг", 0.05, 0.50, 0.15, 0.01, format="%.2f")
        reward_pool_allocation = st.slider("Распределение пула вознаграждений", 0.1, 0.8, 0.4, 0.05, format="%.2f")
        referral_commission_percentage = st.slider("Процент комиссии за рефералов", 0.05, 0.30, 0.10, 0.01, format="%.2f")
        
        st.subheader("Параметры челленджей")
        average_challenges_per_project = st.slider("Среднее количество челленджей на проект", 0.5, 5.0, 1.5, 0.1, format="%.1f")
    
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
    
        # Кнопки для управления симуляцией
    col_buttons = st.columns([1, 1, 1])
    with col_buttons[0]:
        run_simulation = st.button("Запустить симуляцию", type="primary")
    
    with col_buttons[1]:
        if 'simulation' in st.session_state:
            if st.button("Сбросить параметры"):
                # Сбрасываем параметры к значениям по умолчанию
                st.experimental_rerun()
    
    with col_buttons[2]:
        if 'simulation' in st.session_state:
            if st.button("Сохранить сценарий"):
                st.session_state.active_tab = "Управление сценариями"
                st.experimental_rerun()

# Функция для запуска симуляции с заданными параметрами
def run_simulation_with_params(params):
    simulation = TradeFusionSimulation(params)
    simulation.run_simulation()
    return simulation

# Если кнопка нажата, запускаем симуляцию
if 'run_simulation' in locals() and run_simulation:
    # Собираем параметры из интерфейса
    params = {
        # Simulation parameters
        'simulation_months': simulation_months,
        
        # Initial state parameters
        'initial_traders': initial_traders,
        'initial_projects': initial_projects,
        
        # Growth parameters
        'monthly_user_growth_rate': monthly_user_growth_rate,
        'monthly_project_growth_rate': monthly_project_growth_rate,
        'user_churn_rate': user_churn_rate,
        'project_churn_rate': project_churn_rate,
        'referral_conversion_rate': referral_conversion_rate,
        
        # Challenge parameters
        'average_challenges_per_project': average_challenges_per_project,
        
        # Financial parameters
        'development_cost': development_cost,
        'monthly_operational_cost': monthly_operational_cost,
        'marketing_cost_percentage': marketing_cost_percentage,
        'reward_pool_allocation': reward_pool_allocation,
        'referral_commission_percentage': referral_commission_percentage,
        
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
        
        with col1:
            st.metric("Активные трейдеры", f"{int(last_month['active_traders']):,}")
        with col2:
            st.metric("Активные проекты", f"{int(last_month['active_projects']):,}")
        with col3:
            st.metric("Месячная выручка", f"${int(last_month['monthly_revenue']):,}")
        with col4:
            st.metric("Месячная прибыль", f"${int(last_month['monthly_profit']):,}")
        
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
            ax.grid(True)
            st.pyplot(fig)
            
            # Добавляем детализацию расходов
            st.subheader("Структура расходов")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.stackplot(simulation.monthly_metrics['month'], 
                         simulation.monthly_metrics['operational_cost'],
                         simulation.monthly_metrics['marketing_cost'],
                         simulation.monthly_metrics['reward_pool'],
                         simulation.monthly_metrics['referral_commissions'],
                         labels=['Операционные расходы', 'Маркетинг', 'Пул вознаграждений', 'Комиссии рефералов'])
            ax.set_xlabel('Месяц')
            ax.set_ylabel('Сумма ($)')
            ax.legend(loc='upper left')
            ax.grid(True)
            st.pyplot(fig)
        
        with result_tabs[1]:
            # Графики роста пользователей
            st.subheader("Рост пользователей")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(simulation.monthly_metrics['month'], simulation.monthly_metrics['active_traders'], 'b-', linewidth=2)
            ax.set_xlabel('Месяц')
            ax.set_ylabel('Количество активных трейдеров')
            ax.grid(True)
            st.pyplot(fig)
            
            # Рост по типам трейдеров
            st.subheader("Рост по типам трейдеров")
            fig, ax = plt.subplots(figsize=(10, 6))
            for trader_type in TraderType:
                type_name = trader_type.value
                ax.plot(simulation.monthly_metrics['month'], 
                        simulation.monthly_metrics[f'{type_name}_count'], 
                        label=type_name)
            ax.set_xlabel('Месяц')
            ax.set_ylabel('Количество трейдеров')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        
        with result_tabs[2]:
            # Графики роста проектов
            st.subheader("Рост проектов")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(simulation.monthly_metrics['month'], simulation.monthly_metrics['active_projects'], 'g-', linewidth=2)
            ax.set_xlabel('Месяц')
            ax.set_ylabel('Количество активных проектов')
            ax.grid(True)
            st.pyplot(fig)
            
            # Рост по типам проектов
            st.subheader("Рост по типам проектов")
            fig, ax = plt.subplots(figsize=(10, 6))
            for tier in SubscriptionTier:
                tier_name = tier.value
                ax.plot(simulation.monthly_metrics['month'], 
                        simulation.monthly_metrics[f'{tier_name}_count'], 
                        label=tier_name)
            ax.set_xlabel('Месяц')
            ax.set_ylabel('Количество проектов')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        
        with result_tabs[3]:
            # Распределение доходов
            st.subheader("Средний заработок по типам трейдеров")
            fig, ax = plt.subplots(figsize=(10, 6))
            for trader_type in TraderType:
                type_name = trader_type.value
                ax.plot(simulation.monthly_metrics['month'], 
                        simulation.monthly_metrics[f'{type_name}_avg_earnings'], 
                        label=type_name)
            ax.set_xlabel('Месяц')
            ax.set_ylabel('Средний заработок ($)')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            
            # Выбор месяца для анализа распределения доходов
            selected_month = st.slider("Выберите месяц для анализа", 1, len(simulation.monthly_metrics), len(simulation.monthly_metrics))
            month_data = simulation.monthly_metrics.iloc[selected_month-1]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Распределение доходов по типам проектов
                st.subheader(f"Доходы по типам проектов (Месяц {selected_month})")
                tier_revenues = []
                tier_names = []
                
                for tier in SubscriptionTier:
                    tier_name = tier.value
                    tier_names.append(tier_name)
                    tier_revenues.append(month_data[f'{tier_name}_revenue'])
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.bar(tier_names, tier_revenues, color=['#3498db', '#2ecc71', '#e74c3c'])
                ax.set_xlabel('Тип подписки')
                ax.set_ylabel('Доход ($)')
                ax.grid(axis='y')
                
                # Добавляем значения над столбцами
                for i, revenue in enumerate(tier_revenues):
                    ax.text(i, revenue + 100, f'${revenue:,.0f}', ha='center')
                
                st.pyplot(fig)
            
            with col2:
                # Распределение пула вознаграждений
                st.subheader(f"Распределение пула вознаграждений (Месяц {selected_month})")
                reward_types = ['Лидерборд объема', 'Случайный розыгрыш', 'Достижение целей', 'Специальные события']
                percentages = [0.6, 0.2, 0.15, 0.05]
                
                total_reward_pool = month_data['reward_pool']
                amounts = [total_reward_pool * pct for pct in percentages]
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.bar(reward_types, amounts, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'])
                ax.set_xlabel('Тип вознаграждения')
                ax.set_ylabel('Сумма ($)')
                ax.grid(axis='y')
                
                # Добавляем значения над столбцами
                for i, (amount, pct) in enumerate(zip(amounts, percentages)):
                    ax.text(i, amount + 100, f'${amount:,.0f}\n({pct*100:.0f}%)', ha='center')
                
                st.pyplot(fig)
        
        # Таблицы с данными
        st.subheader("Ежемесячные финансовые показатели")
        # Добавляем возможность скачать данные
        csv = simulation.monthly_metrics.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="simulation_results.csv">Скачать данные (CSV)</a>'
        st.markdown(href, unsafe_allow_html=True)

        # Улучшенное отображение таблицы с форматированием
        formatted_df = simulation.monthly_metrics[['month', 'active_traders', 'active_projects', 'monthly_revenue', 'monthly_profit', 'cumulative_profit']].copy()
        formatted_df['monthly_revenue'] = formatted_df['monthly_revenue'].apply(lambda x: f"${x:,.2f}")
        formatted_df['monthly_profit'] = formatted_df['monthly_profit'].apply(lambda x: f"${x:,.2f}")
        formatted_df['cumulative_profit'] = formatted_df['cumulative_profit'].apply(lambda x: f"${x:,.2f}")
        st.dataframe(formatted_df)
        
        # Экономика трейдеров и проектов
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Экономика трейдеров")
            trader_economics = simulation.create_trader_economics_summary()
            # Форматируем числовые значения
            for col in ['Monthly Volume', 'Avg Earnings', 'Lifetime Value']:
                if col in trader_economics.columns:
                    trader_economics[col] = trader_economics[col].apply(lambda x: f"${x:,.2f}")
            st.dataframe(trader_economics)
        
        with col2:
            st.subheader("Экономика проектов")
            project_economics = simulation.create_project_economics_summary()
            # Форматируем числовые значения
            for col in ['Monthly Revenue', 'Reward Pool Contribution', 'Lifetime Value']:
                if col in project_economics.columns:
                    project_economics[col] = project_economics[col].apply(lambda x: f"${x:,.2f}")
            st.dataframe(project_economics)
        
        # Lifetime Values
        st.subheader("Lifetime Values")
        ltv_data = simulation.calculate_lifetime_values()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Trader Lifetime Values:")
            trader_ltv_df = pd.DataFrame({
                'Тип трейдера': [t.value for t in ltv_data['trader_ltv'].keys()],
                'LTV ($)': [f"${ltv:.2f}" for ltv in ltv_data['trader_ltv'].values()]
            })
            st.dataframe(trader_ltv_df)
        
        with col2:
            st.write("Project Lifetime Values:")
            project_ltv_df = pd.DataFrame({
                'Тип проекта': [t.value for t in ltv_data['project_ltv'].keys()],
                'LTV ($)': [f"${ltv:.2f}" for ltv in ltv_data['project_ltv'].values()]
            })
            st.dataframe(project_ltv_df)
    else:
        st.info("Запустите симуляцию на вкладке 'Параметры симуляции', чтобы увидеть результаты")

# Вкладка для управления сценариями
with tab4:
    st.header("Управление сценариями")
    
    # Создаем вкладки для разных функций управления сценариями
    scenario_tabs = st.tabs(["Сохранить сценарий", "Загрузить сценарий", "Сравнить сценарии"])
    
    with scenario_tabs[0]:
        # Интерфейс для сохранения текущего сценария
        if 'simulation' in st.session_state:
            save_scenario_ui()
        else:
            st.info("Запустите симуляцию на вкладке 'Параметры симуляции', чтобы сохранить сценарий")
    
    with scenario_tabs[1]:
        # Интерфейс для загрузки сохраненного сценария
        scenario_data = load_scenario_ui()
        
        if scenario_data and st.button("Применить загруженный сценарий"):
            # Загружаем параметры из сценария
            params = scenario_data["params"]
            
            # Преобразуем строковые значения обратно в enum
            if 'trader_distribution' in params:
                trader_dist = {}
                for k, v in params['trader_distribution'].items():
                    for trader_type in TraderType:
                        if trader_type.name == k:
                            trader_dist[trader_type] = v
                params['trader_distribution'] = trader_dist
            
            if 'project_distribution' in params:
                project_dist = {}
                for k, v in params['project_distribution'].items():
                    for tier in SubscriptionTier:
                        if tier.name == k:
                            project_dist[tier] = v
                params['project_distribution'] = project_dist
            
            # Запускаем симуляцию с загруженными параметрами
            with st.spinner('Запуск симуляции с загруженными параметрами...'):
                st.session_state.simulation = run_simulation_with_params(params)
                st.session_state.params = params
            
            st.success('Симуляция успешно запущена с загруженными параметрами!')
            st.experimental_rerun()
    
    with scenario_tabs[2]:
        # Интерфейс для сравнения сценариев
        compare_scenarios_ui()

# Вкладка с анализом чувствительности
with tab3:
    st.header("Анализ чувствительности")
    
    if 'simulation' in st.session_state:
        simulation = st.session_state.simulation
        
        # Выбор параметров для анализа чувствительности
        st.subheader("Выберите параметры для анализа")
        
        col1, col2 = st.columns(2)
        
        with col1:
            analyze_user_growth = st.checkbox("Темп роста пользователей", True)
            analyze_user_churn = st.checkbox("Отток пользователей", True)
        
        with col2:
            analyze_reward_pool = st.checkbox("Распределение пула вознаграждений", True)
            analyze_marketing_cost = st.checkbox("Процент расходов на маркетинг", True)
        
        # Настройка диапазонов для анализа
        st.subheader("Настройка диапазонов")
        
        parameter_ranges = {}
        
        if analyze_user_growth:
            user_growth_min, user_growth_max = st.slider(
                "Диапазон темпа роста пользователей", 0.05, 0.30, (0.05, 0.25), 0.05
            )
            parameter_ranges['monthly_user_growth_rate'] = np.linspace(user_growth_min, user_growth_max, 5).tolist()
        
        if analyze_user_churn:
            user_churn_min, user_churn_max = st.slider(
                "Диапазон оттока пользователей", 0.02, 0.20, (0.02, 0.15), 0.02
            )
            parameter_ranges['user_churn_rate'] = np.linspace(user_churn_min, user_churn_max, 5).tolist()
        
        if analyze_reward_pool:
            reward_pool_min, reward_pool_max = st.slider(
                "Диапазон распределения пула вознаграждений", 0.2, 0.6, (0.2, 0.6), 0.1
            )
            parameter_ranges['reward_pool_allocation'] = np.linspace(reward_pool_min, reward_pool_max, 5).tolist()
        
        if analyze_marketing_cost:
            marketing_cost_min, marketing_cost_max = st.slider(
                "Диапазон процента расходов на маркетинг", 0.05, 0.25, (0.05, 0.25), 0.05
            )
            parameter_ranges['marketing_cost_percentage'] = np.linspace(marketing_cost_min, marketing_cost_max, 5).tolist()
        
        # Кнопка для запуска анализа чувствительности
        run_sensitivity = st.button("Запустить анализ чувствительности", type="primary")
        
        if run_sensitivity and parameter_ranges:
            with st.spinner('Выполнение анализа чувствительности...'):
                sensitivity_results = simulation.perform_sensitivity_analysis(parameter_ranges)
                st.session_state.sensitivity_results = sensitivity_results
            
            st.success('Анализ чувствительности успешно завершен!')
        
        # Отображение результатов анализа чувствительности
        if 'sensitivity_results' in st.session_state:
            sensitivity_results = st.session_state.sensitivity_results
            
            st.subheader("Результаты анализа чувствительности")
            
            # Отображение графиков для каждого параметра
            for param in parameter_ranges.keys():
                param_data = sensitivity_results[sensitivity_results['parameter'] == param]
                
                st.write(f"**Влияние параметра: {param}**")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                ax1.plot(param_data['value'], param_data['monthly_revenue'], 'b-o')
                ax1.set_title(f'Влияние на месячную выручку')
                ax1.set_xlabel(param)
                ax1.set_ylabel('Месячная выручка ($)')
                ax1.grid(True)
                
                ax2.plot(param_data['value'], param_data['monthly_profit'], 'g-o')
                ax2.set_title(f'Влияние на месячную прибыль')
                ax2.set_xlabel(param)
                ax2.set_ylabel('Месячная прибыль ($)')
                ax2.grid(True)
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # Таблица с результатами
            st.subheader("Таблица результатов анализа чувствительности")
            st.dataframe(sensitivity_results)
    else:
        st.info("Запустите симуляцию на вкладке 'Параметры симуляции', чтобы выполнить анализ чувствительности")

# Добавление информации о проекте в сайдбар
with st.sidebar:
    st.title("О проекте")
    st.markdown("""
    **TradeFusion Simulator** - интерактивный инструмент для моделирования и анализа платформы TradeFusion.
    
    Используйте этот инструмент для:
    - Настройки параметров симуляции
    - Визуализации результатов
    - Анализа чувствительности к изменениям параметров
    - Оптимизации бизнес-модели
    """)
    
    st.markdown("---")
    st.markdown("© 2023 TradeFusion")