import json
import os
import pandas as pd
import streamlit as st
from datetime import datetime

class ScenarioManager:
    """
    Класс для управления сценариями симуляции TradeFusion.
    Позволяет сохранять, загружать и сравнивать различные сценарии.
    """
    def __init__(self, scenarios_dir="scenarios"):
        self.scenarios_dir = scenarios_dir
        os.makedirs(scenarios_dir, exist_ok=True)
    
    def save_scenario(self, params, results=None, name=None, description=None):
        """
        Сохраняет сценарий симуляции в файл JSON.
        
        Args:
            params: Словарь параметров симуляции
            results: Опциональные результаты симуляции (DataFrame)
            name: Имя сценария (если не указано, будет сгенерировано автоматически)
            description: Описание сценария
        
        Returns:
            str: Путь к сохраненному файлу
        """
        # Создаем имя файла, если не указано
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"scenario_{timestamp}"
        
        # Подготавливаем данные для сохранения
        scenario_data = {
            "name": name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "params": {}
        }
        
        # Преобразуем enum-значения в строки для сохранения
        for key, value in params.items():
            if key == 'trader_distribution' or key == 'project_distribution':
                scenario_data["params"][key] = {k.name: v for k, v in value.items()}
            else:
                scenario_data["params"][key] = value
        
        # Сохраняем результаты, если они предоставлены
        if results is not None and isinstance(results, pd.DataFrame):
            results_path = os.path.join(self.scenarios_dir, f"{name}_results.csv")
            results.to_csv(results_path, index=False)
            scenario_data["results_path"] = results_path
        
        # Сохраняем сценарий в JSON файл
        file_path = os.path.join(self.scenarios_dir, f"{name}.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(scenario_data, f, ensure_ascii=False, indent=4)
        
        return file_path
    
    def load_scenario(self, name):
        """
        Загружает сценарий из файла JSON.
        
        Args:
            name: Имя сценария или путь к файлу
        
        Returns:
            dict: Словарь с данными сценария
        """
        # Определяем путь к файлу
        if name.endswith('.json'):
            file_path = name
        else:
            file_path = os.path.join(self.scenarios_dir, f"{name}.json")
        
        # Загружаем данные сценария
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                scenario_data = json.load(f)
            
            # Загружаем результаты, если они есть
            if "results_path" in scenario_data and os.path.exists(scenario_data["results_path"]):
                scenario_data["results"] = pd.read_csv(scenario_data["results_path"])
            
            return scenario_data
        except Exception as e:
            print(f"Ошибка при загрузке сценария: {e}")
            return None
    
    def list_scenarios(self):
        """
        Возвращает список доступных сценариев.
        
        Returns:
            list: Список имен сценариев
        """
        scenarios = []
        for file in os.listdir(self.scenarios_dir):
            if file.endswith('.json'):
                try:
                    scenario_path = os.path.join(self.scenarios_dir, file)
                    with open(scenario_path, 'r', encoding='utf-8') as f:
                        scenario_data = json.load(f)
                    
                    scenarios.append({
                        "name": scenario_data.get("name", file.replace('.json', '')),
                        "description": scenario_data.get("description", ""),
                        "created_at": scenario_data.get("created_at", ""),
                        "file_path": scenario_path
                    })
                except Exception as e:
                    print(f"Ошибка при чтении сценария {file}: {e}")
        
        # Сортируем по дате создания (новые сверху)
        scenarios.sort(key=lambda x: x["created_at"], reverse=True)
        return scenarios
    
    def compare_scenarios(self, scenario_names):
        """
        Сравнивает несколько сценариев и возвращает DataFrame с результатами.
        
        Args:
            scenario_names: Список имен сценариев для сравнения
        
        Returns:
            pd.DataFrame: Таблица сравнения сценариев
        """
        comparison = []
        
        for name in scenario_names:
            scenario = self.load_scenario(name)
            if scenario:
                # Извлекаем ключевые параметры для сравнения
                params = scenario["params"]
                row = {
                    "Сценарий": scenario.get("name", name),
                    "Месяцы симуляции": params.get("simulation_months", "-"),
                    "Начальные трейдеры": params.get("initial_traders", "-"),
                    "Начальные проекты": params.get("initial_projects", "-"),
                    "Рост пользователей": params.get("monthly_user_growth_rate", "-"),
                    "Отток пользователей": params.get("user_churn_rate", "-"),
                    "Пул вознаграждений": params.get("reward_pool_allocation", "-"),
                    "Расходы на маркетинг": params.get("marketing_cost_percentage", "-")
                }
                
                # Добавляем результаты, если они есть
                if "results" in scenario and isinstance(scenario["results"], pd.DataFrame):
                    results = scenario["results"]
                    if not results.empty:
                        last_month = results.iloc[-1]
                        row["Месячная выручка"] = last_month.get("monthly_revenue", "-")
                        row["Месячная прибыль"] = last_month.get("monthly_profit", "-")
                        row["Активные трейдеры"] = last_month.get("active_traders", "-")
                        row["Активные проекты"] = last_month.get("active_projects", "-")
                        
                        # Определяем точку безубыточности
                        break_even_month = None
                        for i, r in results.iterrows():
                            if r.get("cumulative_profit", float('-inf')) >= 0:
                                break_even_month = r.get("month")
                                break
                        
                        row["Точка безубыточности"] = break_even_month if break_even_month else "Не достигнута"
                
                comparison.append(row)
        
        return pd.DataFrame(comparison) if comparison else None

# Функции для использования в Streamlit
def save_scenario_ui():
    """
    Интерфейс для сохранения сценария в Streamlit
    """
    st.subheader("Сохранение сценария")
    
    name = st.text_input("Название сценария", 
                        value=f"Сценарий {datetime.now().strftime('%d.%m.%Y %H:%M')}")
    description = st.text_area("Описание сценария", 
                             placeholder="Опишите особенности этого сценария...")
    
    if st.button("Сохранить сценарий", type="primary"):
        if 'params' in st.session_state and 'simulation' in st.session_state:
            manager = ScenarioManager()
            params = st.session_state.params
            results = st.session_state.simulation.monthly_metrics
            
            file_path = manager.save_scenario(params, results, name, description)
            st.success(f"Сценарий успешно сохранен: {name}")
            return True
        else:
            st.error("Нет данных для сохранения. Сначала запустите симуляцию.")
    
    return False

def load_scenario_ui():
    """
    Интерфейс для загрузки сценария в Streamlit
    """
    st.subheader("Загрузка сценария")
    
    manager = ScenarioManager()
    scenarios = manager.list_scenarios()
    
    if not scenarios:
        st.info("Нет сохраненных сценариев. Сначала сохраните сценарий.")
        return None
    
    # Создаем список для выбора
    scenario_options = {f"{s['name']} ({s['created_at'].split('T')[0]})": s for s in scenarios}
    selected_option = st.selectbox("Выберите сценарий", list(scenario_options.keys()))
    
    if st.button("Загрузить сценарий"):
        selected_scenario = scenario_options[selected_option]
        scenario_data = manager.load_scenario(selected_scenario["file_path"])
        
        if scenario_data:
            st.success(f"Сценарий успешно загружен: {selected_scenario['name']}")
            return scenario_data
        else:
            st.error("Ошибка при загрузке сценария.")
    
    return None

def compare_scenarios_ui():
    """
    Интерфейс для сравнения сценариев в Streamlit
    """
    st.subheader("Сравнение сценариев")
    
    manager = ScenarioManager()
    scenarios = manager.list_scenarios()
    
    if not scenarios or len(scenarios) < 2:
        st.info("Недостаточно сценариев для сравнения. Сохраните как минимум два сценария.")
        return
    
    # Создаем список для выбора
    scenario_options = {s['name']: s['file_path'] for s in scenarios}
    selected_scenarios = st.multiselect("Выберите сценарии для сравнения", 
                                       list(scenario_options.keys()),
                                       max_selections=5)
    
    if selected_scenarios and st.button("Сравнить сценарии"):
        selected_paths = [scenario_options[name] for name in selected_scenarios]
        comparison_df = manager.compare_scenarios(selected_paths)
        
        if comparison_df is not None:
            st.dataframe(comparison_df)
            
            # Визуализация сравнения
            if len(selected_scenarios) > 1 and 'Месячная прибыль' in comparison_df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                comparison_df.plot(x='Сценарий', y='Месячная прибыль', kind='bar', ax=ax)
                ax.set_title('Сравнение месячной прибыли по сценариям')
                ax.set_ylabel('Прибыль ($)')
                ax.grid(axis='y')
                st.pyplot(fig)
        else:
            st.error("Не удалось создать сравнение сценариев.")