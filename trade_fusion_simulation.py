import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import random
from enum import Enum
import os
from datetime import datetime

# Define trader types as an enum for better type safety
class TraderType(Enum):
    CASUAL = "Casual"
    ACTIVE = "Active"
    PROFESSIONAL = "Professional"
    WHALE = "Whale"

# Define project subscription tiers as an enum
class SubscriptionTier(Enum):
    STANDARD = "Standard"
    PREMIUM = "Premium"
    ENTERPRISE = "Enterprise"

# Define reward distribution types
class RewardType(Enum):
    VOLUME_LEADERBOARD = "Volume Leaderboard"
    RANDOM_DRAW = "Random Draw"
    MILESTONE = "Milestone Achievement"
    SPECIAL_EVENT = "Special Event"

@dataclass
class Trader:
    id: int
    type: TraderType
    monthly_volume_min: float
    monthly_volume_max: float
    current_volume: float = 0
    total_rewards: float = 0
    months_active: int = 0
    challenges_participated: int = 0
    referred_by: Optional[int] = None
    referrals: List[int] = None
    is_active: bool = True
    
    def __post_init__(self):
        if self.referrals is None:
            self.referrals = []
        self.generate_monthly_volume()
    
    def generate_monthly_volume(self):
        """Generate a random trading volume within the trader's range"""
        self.current_volume = random.uniform(self.monthly_volume_min, self.monthly_volume_max)
        return self.current_volume
    
    def add_reward(self, amount: float):
        """Add rewards earned by the trader"""
        self.total_rewards += amount
    
    def add_referral(self, trader_id: int):
        """Add a trader that this trader has referred"""
        self.referrals.append(trader_id)
    
    def participate_in_challenge(self):
        """Increment the number of challenges this trader has participated in"""
        self.challenges_participated += 1
    
    def monthly_update(self, churn_rate: float):
        """Update trader stats for a new month and determine if they churn"""
        if not self.is_active:
            return False
        
        self.months_active += 1
        self.generate_monthly_volume()
        
        # Adjust churn rate based on rewards (traders who earn more are less likely to churn)
        adjusted_churn_rate = churn_rate
        if self.total_rewards > 0:
            # Reduce churn rate by up to 50% based on rewards
            churn_factor = min(0.5, self.total_rewards / 1000)  # Cap at 50% reduction
            adjusted_churn_rate = churn_rate * (1 - churn_factor)
        
        # Determine if trader churns
        if random.random() < adjusted_churn_rate:
            self.is_active = False
            return False
        return True

@dataclass
class Project:
    id: int
    tier: SubscriptionTier
    subscription_fee: float
    reward_pool_allocation: float
    months_active: int = 0
    total_paid: float = 0
    challenges: List[int] = None
    referred_by: Optional[int] = None
    referrals: List[int] = None
    is_active: bool = True
    
    def __post_init__(self):
        if self.challenges is None:
            self.challenges = []
        if self.referrals is None:
            self.referrals = []
    
    def add_challenge(self, challenge_id: int):
        """Add a challenge created by this project"""
        self.challenges.append(challenge_id)
    
    def add_referral(self, project_id: int):
        """Add a project that this project has referred"""
        self.referrals.append(project_id)
    
    def pay_subscription(self):
        """Process monthly subscription payment"""
        self.total_paid += self.subscription_fee
        return self.subscription_fee
    
    def get_reward_pool_contribution(self):
        """Calculate how much this project contributes to the reward pool"""
        return self.subscription_fee * self.reward_pool_allocation
    
    def monthly_update(self, churn_rate: float):
        """Update project stats for a new month and determine if they churn"""
        if not self.is_active:
            return False
        
        self.months_active += 1
        
        # Determine if project churns
        if random.random() < churn_rate:
            self.is_active = False
            return False
        return True

class Challenge:
    def __init__(self, id: int, project_id: int, reward_pool: float):
        self.id = id
        self.project_id = project_id
        self.reward_pool = reward_pool
        self.participants: Dict[int, float] = {}  # trader_id -> volume
        self.rewards_distributed: Dict[RewardType, Dict[int, float]] = {
            reward_type: {} for reward_type in RewardType
        }
        self.total_volume = 0
    
    def add_participant(self, trader_id: int, volume: float):
        """Add a trader's participation in this challenge"""
        self.participants[trader_id] = volume
        self.total_volume += volume
    
    def distribute_rewards(self, traders: Dict[int, Trader]):
        """Distribute rewards to participants based on the defined mechanisms"""
        if not self.participants:
            return {}
        
        # Оптимизированное распределение пула вознаграждений
        # Увеличиваем долю для лидерборда объема и достижений, уменьшаем случайные розыгрыши
        volume_leaderboard_pool = self.reward_pool * 0.55  # 55% to volume leaderboards (было 60%)
        random_draw_pool = self.reward_pool * 0.15  # 15% to random draws (было 20%)
        milestone_pool = self.reward_pool * 0.25  # 25% to milestone achievements (было 15%)
        special_event_pool = self.reward_pool * 0.05  # 5% to special events (без изменений)
        
        rewards = {}
        
        # 1. Volume Leaderboard (55% of rewards)
        sorted_participants = sorted(self.participants.items(), key=lambda x: x[1], reverse=True)
        top_count = max(3, int(len(sorted_participants) * 0.15))  # Top 15% or at least 3 traders (было 10%)
        
        # Более справедливое распределение для лидерборда объема
        # Используем логарифмическую шкалу для более равномерного распределения
        total_top_volume = sum(volume for _, volume in sorted_participants[:top_count])
        
        for i, (trader_id, volume) in enumerate(sorted_participants[:top_count]):
            # Распределение на основе доли объема с бонусом для топ-3
            volume_share = volume / total_top_volume if total_top_volume > 0 else 0
            
            if i == 0:  # 1st place - бонус 20%
                reward = volume_leaderboard_pool * (volume_share * 1.2)
            elif i == 1:  # 2nd place - бонус 10%
                reward = volume_leaderboard_pool * (volume_share * 1.1)
            elif i == 2:  # 3rd place - бонус 5%
                reward = volume_leaderboard_pool * (volume_share * 1.05)
            else:  # Остальные трейдеры - по доле объема
                reward = volume_leaderboard_pool * volume_share
            
            rewards[trader_id] = rewards.get(trader_id, 0) + reward
            self.rewards_distributed[RewardType.VOLUME_LEADERBOARD][trader_id] = reward
        
        # 2. Random Draws (15% of rewards)
        # Снижаем порог для участия в случайных розыгрышах
        min_threshold = sum(self.participants.values()) / len(self.participants) * 0.03  # Было 0.05
        eligible_traders = [tid for tid, vol in self.participants.items() if vol >= min_threshold]
        
        if eligible_traders:
            # Веса на основе объема, но с меньшей разницей между большими и малыми объемами
            volumes = [self.participants[tid] for tid in eligible_traders]
            log_volumes = [np.log1p(vol) for vol in volumes]  # Логарифмическая шкала сглаживает разницу
            total_log_volume = sum(log_volumes)
            normalized_weights = [lv/total_log_volume for lv in log_volumes] if total_log_volume > 0 else None
            
            # Увеличиваем количество победителей
            num_winners = max(5, int(len(eligible_traders) * 0.08))  # Было 5% или 3, теперь 8% или 5
            winners = random.choices(eligible_traders, weights=normalized_weights, k=min(num_winners, len(eligible_traders)))
            
            for winner in winners:
                reward = random_draw_pool / len(winners)
                rewards[winner] = rewards.get(winner, 0) + reward
                self.rewards_distributed[RewardType.RANDOM_DRAW][winner] = reward
        
        # 3. Milestone Achievements (25% of rewards)
        # Более реалистичные и достижимые цели для разных типов трейдеров
        milestones = {
            TraderType.CASUAL: 8000,        # $8K volume (было $10K)
            TraderType.ACTIVE: 80000,       # $80K volume (было $100K)
            TraderType.PROFESSIONAL: 800000,    # $800K volume (было $1M)
            TraderType.WHALE: 8000000       # $8M volume (было $10M)
        }
        
        # Добавляем прогрессивные достижения - частичное вознаграждение за достижение части цели
        milestone_achievers = {}
        partial_achievers = {}
        
        for trader_id, volume in self.participants.items():
            trader_type = traders[trader_id].type
            target = milestones[trader_type]
            
            if volume >= target:
                # Полное достижение цели
                milestone_achievers[trader_id] = trader_type
            elif volume >= target * 0.7:  # Достижение 70% от цели
                # Частичное достижение
                partial_achievers[trader_id] = (trader_type, volume / target)  # Сохраняем процент достижения
        
        # Распределяем 80% пула среди полных достижений
        full_achievement_pool = milestone_pool * 0.8
        if milestone_achievers:
            reward_per_achiever = full_achievement_pool / len(milestone_achievers)
            for trader_id, _ in milestone_achievers.items():
                rewards[trader_id] = rewards.get(trader_id, 0) + reward_per_achiever
                self.rewards_distributed[RewardType.MILESTONE][trader_id] = reward_per_achiever
        
        # Распределяем 20% пула среди частичных достижений
        partial_achievement_pool = milestone_pool * 0.2
        if partial_achievers:
            # Распределяем пропорционально проценту достижения
            total_percentage = sum(pct for _, pct in partial_achievers.values())
            for trader_id, (_, percentage) in partial_achievers.items():
                reward = (percentage / total_percentage) * partial_achievement_pool if total_percentage > 0 else 0
                rewards[trader_id] = rewards.get(trader_id, 0) + reward
                self.rewards_distributed[RewardType.MILESTONE][trader_id] = reward
        
        # 4. Special Events (5% of rewards)
        # Более справедливое распределение для специальных событий
        if self.participants:
            # Увеличиваем количество победителей
            special_event_count = max(2, int(len(self.participants) * 0.05))  # 5% участников или минимум 2 (было 3%)
            
            # Выбираем победителей с учетом активности (месяцы на платформе)
            # Трейдеры, которые дольше на платформе, имеют больше шансов
            eligible_traders = list(self.participants.keys())
            weights = [traders[tid].months_active + 1 for tid in eligible_traders]  # +1 чтобы избежать нулевых весов
            
            special_event_winners = random.choices(
                eligible_traders, 
                weights=weights, 
                k=min(special_event_count, len(eligible_traders))
            )
            
            reward_per_winner = special_event_pool / len(special_event_winners)
            for winner in special_event_winners:
                rewards[winner] = rewards.get(winner, 0) + reward_per_winner
                self.rewards_distributed[RewardType.SPECIAL_EVENT][winner] = reward_per_winner
        
        return rewards

class TradeFusionSimulation:
    def __init__(self, params: dict):
        # Initialize simulation parameters
        self.params = params
        self.current_month = 0
        self.max_months = params.get('simulation_months', 24)
        
        # Initialize counters
        self.trader_id_counter = 0
        self.project_id_counter = 0
        self.challenge_id_counter = 0
        
        # Initialize collections
        self.traders: Dict[int, Trader] = {}
        self.projects: Dict[int, Project] = {}
        self.challenges: Dict[int, Challenge] = {}
        
        # Initialize tracking dataframes
        self.monthly_metrics = pd.DataFrame()
        self.trader_metrics = pd.DataFrame()
        self.project_metrics = pd.DataFrame()
        
        # Initialize platform financials
        self.development_cost = params.get('development_cost', 225000)
        self.monthly_operational_cost = params.get('monthly_operational_cost', 30000)
        self.marketing_cost_percentage = params.get('marketing_cost_percentage', 0.15)
        self.reward_pool_allocation = params.get('reward_pool_allocation', 0.4)
        self.referral_commission_percentage = params.get('referral_commission_percentage', 0.1)
        
        self.total_revenue = 0
        self.total_costs = self.development_cost  # Initial development cost
        self.total_profit = -self.development_cost  # Start with negative profit due to development cost
        
        # Initialize the simulation
        self._initialize_simulation()
    
    def _initialize_simulation(self):
        """Set up the initial state of the simulation"""
        # Create initial traders
        initial_traders = self.params.get('initial_traders', 1000)
        trader_distribution = {
            TraderType.CASUAL: 0.7,
            TraderType.ACTIVE: 0.2,
            TraderType.PROFESSIONAL: 0.08,
            TraderType.WHALE: 0.02
        }
        
        for trader_type, percentage in trader_distribution.items():
            count = int(initial_traders * percentage)
            for _ in range(count):
                self._create_trader(trader_type)
        
        # Create initial projects
        initial_projects = self.params.get('initial_projects', 10)
        project_distribution = {
            SubscriptionTier.STANDARD: 0.6,
            SubscriptionTier.PREMIUM: 0.3,
            SubscriptionTier.ENTERPRISE: 0.1
        }
        
        for tier, percentage in project_distribution.items():
            count = int(initial_projects * percentage)
            for _ in range(count):
                self._create_project(tier)
        
        # Create initial challenges
        self._create_initial_challenges()
    
    def _create_trader(self, trader_type: TraderType, referred_by: Optional[int] = None) -> int:
        """Create a new trader and return their ID"""
        volume_ranges = {
            TraderType.CASUAL: (5000, 10000),
            TraderType.ACTIVE: (50000, 100000),
            TraderType.PROFESSIONAL: (500000, 1000000),
            TraderType.WHALE: (5000000, 10000000)
        }
        
        trader_id = self.trader_id_counter
        self.trader_id_counter += 1
        
        min_vol, max_vol = volume_ranges[trader_type]
        trader = Trader(
            id=trader_id,
            type=trader_type,
            monthly_volume_min=min_vol,
            monthly_volume_max=max_vol,
            referred_by=referred_by
        )
        
        self.traders[trader_id] = trader
        
        # If this trader was referred, update the referrer's referrals
        if referred_by is not None and referred_by in self.traders:
            self.traders[referred_by].add_referral(trader_id)
        
        return trader_id
    
    def _create_project(self, tier: SubscriptionTier, referred_by: Optional[int] = None) -> int:
        """Create a new project and return its ID"""
        subscription_fees = {
            SubscriptionTier.STANDARD: 2000,
            SubscriptionTier.PREMIUM: 5000,
            SubscriptionTier.ENTERPRISE: 10000
        }
        
        project_id = self.project_id_counter
        self.project_id_counter += 1
        
        project = Project(
            id=project_id,
            tier=tier,
            subscription_fee=subscription_fees[tier],
            reward_pool_allocation=self.reward_pool_allocation,
            referred_by=referred_by
        )
        
        self.projects[project_id] = project
        
        # If this project was referred, update the referrer's referrals
        if referred_by is not None and referred_by in self.projects:
            self.projects[referred_by].add_referral(project_id)
        
        return project_id
    
    def _create_challenge(self, project_id: int) -> int:
        """Create a new challenge for a project and return its ID"""
        if project_id not in self.projects:
            return None
        
        project = self.projects[project_id]
        challenge_id = self.challenge_id_counter
        self.challenge_id_counter += 1
        
        reward_pool = project.get_reward_pool_contribution()
        challenge = Challenge(id=challenge_id, project_id=project_id, reward_pool=reward_pool)
        
        self.challenges[challenge_id] = challenge
        project.add_challenge(challenge_id)
        
        return challenge_id
    
    def _create_initial_challenges(self):
        """Create initial challenges for projects"""
        avg_challenges_per_project = self.params.get('average_challenges_per_project', 1.5)
        
        for project_id, project in self.projects.items():
            # Determine number of challenges for this project
            num_challenges = max(1, int(random.normalvariate(avg_challenges_per_project, 0.5)))
            for _ in range(num_challenges):
                self._create_challenge(project_id)
    
    def _assign_traders_to_challenges(self):
        """Assign traders to challenges based on participation rates"""
        participation_rates = {
            TraderType.CASUAL: 3,
            TraderType.ACTIVE: 5,
            TraderType.PROFESSIONAL: 8,
            TraderType.WHALE: 10
        }
        
        active_challenges = [c for c in self.challenges.values() 
                           if self.projects[c.project_id].is_active]
        
        if not active_challenges:
            return
        
        for trader_id, trader in self.traders.items():
            if not trader.is_active:
                continue
            
            # Determine how many challenges this trader participates in
            max_participation = participation_rates[trader.type]
            actual_participation = min(max_participation, len(active_challenges))
            
            # Randomly select challenges
            selected_challenges = random.sample(active_challenges, actual_participation)
            
            for challenge in selected_challenges:
                challenge.add_participant(trader_id, trader.current_volume)
                trader.participate_in_challenge()
    
    def _distribute_challenge_rewards(self):
        """Distribute rewards from all active challenges to traders"""
        total_rewards_distributed = 0
        
        for challenge_id, challenge in self.challenges.items():
            if not self.projects[challenge.project_id].is_active:
                continue
            
            # Оптимизация распределения вознаграждений
            # Учитываем объем торговли и активность трейдеров
            rewards = challenge.distribute_rewards(self.traders)
            
            # Более справедливое распределение вознаграждений
            # Трейдеры с большим объемом получают больше, но с уменьшающейся отдачей
            for trader_id, reward_amount in rewards.items():
                if trader_id in self.traders:
                    trader = self.traders[trader_id]
                    
                    # Бонус за лояльность - трейдеры, которые дольше на платформе, получают больше
                    loyalty_bonus = min(0.2, trader.months_active / 24)  # До 20% бонуса за 24 месяца
                    adjusted_reward = reward_amount * (1 + loyalty_bonus)
                    
                    trader.add_reward(adjusted_reward)
                    total_rewards_distributed += adjusted_reward
        
        return total_rewards_distributed
    
    def _process_referral_commissions(self, monthly_subscription_revenue):
        """Process referral commissions for both traders and projects"""
        total_referral_commissions = 0
        
        # Process project referrals (% of subscription fees)
        for project_id, project in self.projects.items():
            if not project.is_active or project.referred_by is None:
                continue
            
            referrer_id = project.referred_by
            if referrer_id in self.projects and self.projects[referrer_id].is_active:
                commission = project.subscription_fee * self.referral_commission_percentage
                total_referral_commissions += commission
        
        # For trader referrals, we'll give a percentage of their earned rewards
        # This is calculated in the _distribute_challenge_rewards method
        
        return total_referral_commissions
    
    def _generate_new_traders(self):
        """Generate new traders based on number of projects and referrals"""
        active_traders = [t for t in self.traders.values() if t.is_active]
        active_projects = [p for p in self.projects.values() if p.is_active]
        
        # Получаем базовые параметры CAC и эффективность маркетинга
        base_cac_retail = self.params.get('base_cac_retail', 50)
        base_cac_b2b = self.params.get('base_cac_b2b', 1000)
        marketing_efficiency = self.params.get('marketing_efficiency', 1.0)
        
        # Рассчитываем текущий CAC на основе процента расходов на маркетинг
        marketing_budget = self.monthly_operational_cost * self.marketing_cost_percentage
        marketing_multiplier = (self.marketing_cost_percentage / 0.15) ** marketing_efficiency
        
        # Корректируем CAC в зависимости от маркетингового бюджета
        current_cac_retail = base_cac_retail / marketing_multiplier
        current_cac_b2b = base_cac_b2b / marketing_multiplier
        
        # Новая модель: каждый проект привлекает определенное количество пользователей
        # Разные типы проектов привлекают разное количество пользователей
        users_per_project = {
            SubscriptionTier.STANDARD: 15,   # Стандартные проекты привлекают 15 пользователей
            SubscriptionTier.PREMIUM: 30,    # Премиум проекты привлекают 30 пользователей
            SubscriptionTier.ENTERPRISE: 50  # Корпоративные проекты привлекают 50 пользователей
        }
        
        # Рассчитываем базовое количество новых пользователей от проектов
        base_new_traders = 0
        for project in active_projects:
            base_new_traders += users_per_project[project.tier]
        
        # Корректировка в зависимости от прибыльности платформы
        profitability_factor = 1.0
        if self.total_profit > 0:
            profitability_factor = min(1.3, 1 + (self.total_profit / (self.total_revenue + 1)) * 0.5)
        elif self.total_profit < 0:
            profitability_factor = max(0.8, 1 - abs(self.total_profit) / (self.total_revenue + 1) * 0.3)
        
        # Применяем корректировку и масштабируем для более реалистичных чисел
        # Чтобы не было слишком много новых пользователей, берем процент от расчетного количества
        scaling_factor = 0.1  # 10% от теоретического максимума
        
        # Корректируем базовое количество новых пользователей с учетом эффективности маркетинга
        marketing_factor = marketing_multiplier
        base_new_traders = int(base_new_traders * scaling_factor * profitability_factor * marketing_factor)
        
        # Рассчитываем, сколько пользователей можем привлечь с текущим бюджетом
        # Предполагаем, что 80% новых пользователей - розничные, 20% - B2B
        retail_ratio = 0.8
        b2b_ratio = 0.2
        
        max_new_retail = int(marketing_budget * retail_ratio / current_cac_retail)
        max_new_b2b = int(marketing_budget * b2b_ratio / current_cac_b2b)
        max_new_from_budget = max_new_retail + max_new_b2b
        
        # Ограничиваем базовое количество новых пользователей бюджетом
        base_new_traders = min(base_new_traders, max_new_from_budget)
        
        # Улучшенная модель рефералов
        # Более активные трейдеры и трейдеры с большим объемом имеют больше шансов привлечь новых пользователей
        referral_conversion_rate = self.params.get('referral_conversion_rate', 0.1)
        referral_new_traders = 0
        
        for trader in active_traders:
            # Корректировка шанса реферала в зависимости от типа трейдера и активности
            trader_type_factor = {
                TraderType.CASUAL: 0.8,
                TraderType.ACTIVE: 1.2,
                TraderType.PROFESSIONAL: 1.5,
                TraderType.WHALE: 2.0
            }.get(trader.type, 1.0)
            
            activity_factor = min(1.5, trader.months_active / 6)  # Более активные трейдеры привлекают больше
            adjusted_referral_rate = referral_conversion_rate * trader_type_factor * activity_factor
            
            # Каждый активный трейдер имеет шанс привлечь новых пользователей
            if random.random() < adjusted_referral_rate:
                # Более активные трейдеры могут привлечь больше пользователей за раз
                new_referrals = 1
                if trader.type in [TraderType.PROFESSIONAL, TraderType.WHALE] and random.random() < 0.3:
                    new_referrals = random.randint(2, 3)
                
                referral_new_traders += new_referrals
        
        total_new_traders = base_new_traders + referral_new_traders
        
        # Distribute new traders according to the distribution
        trader_distribution = {
            TraderType.CASUAL: 0.7,
            TraderType.ACTIVE: 0.2,
            TraderType.PROFESSIONAL: 0.08,
            TraderType.WHALE: 0.02
        }
        
        # Create the new traders
        for trader_type, percentage in trader_distribution.items():
            count = int(total_new_traders * percentage)
            for _ in range(count):
                # Randomly assign a referrer to some percentage of new traders
                if random.random() < 0.3 and active_traders:  # 30% chance of being referred
                    referrer = random.choice(active_traders)
                    self._create_trader(trader_type, referred_by=referrer.id)
                else:
                    self._create_trader(trader_type)
    
    def _generate_new_projects(self):
        """Generate new projects based on fixed monthly additions"""
        # Используем фиксированное количество новых проектов в месяц вместо процентного роста
        # Получаем настройки из параметров или используем значения по умолчанию
        month_to_projects = self.params.get('month_to_projects', {
            1: 3,   # 3 новых проекта в 1-й месяц
            3: 5,   # 5 новых проектов начиная с 3-го месяца
            6: 8,   # 8 новых проектов начиная с 6-го месяца
            12: 10, # 10 новых проектов начиная с 12-го месяца
            18: 15  # 15 новых проектов начиная с 18-го месяца
        })
        
        # Определяем количество новых проектов для текущего месяца
        new_projects_count = 3  # По умолчанию 3 проекта
        for month, count in sorted(month_to_projects.items()):
            if self.current_month >= month:
                new_projects_count = count
        
        # Рассчитываем дополнительные проекты от рефералов (20% от базового количества)
        referral_new_projects = int(new_projects_count * 0.2)
        
        total_new_projects = new_projects_count + referral_new_projects
        
        # Получаем список активных проектов для рефералов
        active_projects = [project for project in self.projects.values() if project.is_active]
        
        # Distribute new projects according to the distribution
        project_distribution = {
            SubscriptionTier.STANDARD: 0.6,
            SubscriptionTier.PREMIUM: 0.3,
            SubscriptionTier.ENTERPRISE: 0.1
        }
        
        # Create the new projects
        for tier, percentage in project_distribution.items():
            count = int(total_new_projects * percentage)
            for _ in range(count):
                # Randomly assign a referrer to some percentage of new projects
                if random.random() < 0.3 and active_projects:  # 30% chance of being referred
                    referrer = random.choice(active_projects)
                    self._create_project(tier, referred_by=referrer.id)
                else:
                    self._create_project(tier)
    
    def _update_traders(self):
        """Update all traders for the current month"""
        churn_rate = self.params.get('user_churn_rate', 0.05)
        for trader in self.traders.values():
            trader.monthly_update(churn_rate)
    
    def _update_projects(self):
        """Update all projects for the current month"""
        churn_rate = self.params.get('project_churn_rate', 0.08)
        for project in self.projects.values():
            project.monthly_update(churn_rate)
    
    def _collect_monthly_subscription_fees(self):
        """Collect subscription fees from all active projects"""
        monthly_revenue = 0
        for project in self.projects.values():
            if project.is_active:
                monthly_revenue += project.pay_subscription()
        return monthly_revenue
    
    def _calculate_monthly_financials(self, monthly_revenue):
        """Calculate monthly financial metrics"""
        # Calculate costs
        operational_cost = self.monthly_operational_cost
        
        # Оптимизация расходов на маркетинг - базовый процент от выручки, но с ограничением
        # Это предотвращает чрезмерные расходы при высокой выручке
        base_marketing = monthly_revenue * self.marketing_cost_percentage
        max_marketing = self.monthly_operational_cost * 2  # Не более чем в 2 раза больше операционных расходов
        marketing_cost = min(base_marketing, max_marketing)
        
        # Более эффективное распределение пула вознаграждений
        # Процент от выручки, но с динамической корректировкой в зависимости от прибыльности
        base_reward_pool = monthly_revenue * self.reward_pool_allocation
        
        # Если платформа убыточна, немного сокращаем пул вознаграждений
        if self.total_profit < 0:
            adjustment_factor = max(0.7, 1 - abs(self.total_profit) / (self.total_revenue + 1))
            reward_pool = base_reward_pool * adjustment_factor
        else:
            # Если платформа прибыльна, можем немного увеличить пул вознаграждений
            reward_pool = base_reward_pool * min(1.2, 1 + (self.total_profit / (self.total_revenue + 1)))
        
        referral_commissions = self._process_referral_commissions(monthly_revenue)
        
        total_monthly_cost = operational_cost + marketing_cost + reward_pool + referral_commissions
        monthly_profit = monthly_revenue - total_monthly_cost
        
        # Update cumulative financials
        self.total_revenue += monthly_revenue
        self.total_costs += total_monthly_cost
        self.total_profit = self.total_revenue - self.total_costs
        
        return {
            'month': self.current_month,
            'monthly_revenue': monthly_revenue,
            'operational_cost': operational_cost,
            'marketing_cost': marketing_cost,
            'reward_pool': reward_pool,
            'referral_commissions': referral_commissions,
            'total_monthly_cost': total_monthly_cost,
            'monthly_profit': monthly_profit,
            'cumulative_revenue': self.total_revenue,
            'cumulative_costs': self.total_costs,
            'cumulative_profit': self.total_profit
        }
    
    def _calculate_monthly_metrics(self):
        """Calculate various monthly metrics for tracking"""
        active_traders = [t for t in self.traders.values() if t.is_active]
        active_projects = [p for p in self.projects.values() if p.is_active]
        
        # Calculate metrics by trader type
        trader_metrics = {trader_type: {
            'count': 0,
            'volume': 0,
            'rewards': 0
        } for trader_type in TraderType}
        
        for trader in active_traders:
            trader_metrics[trader.type]['count'] += 1
            trader_metrics[trader.type]['volume'] += trader.current_volume
            trader_metrics[trader.type]['rewards'] += trader.total_rewards
        
        # Calculate metrics by project tier
        project_metrics = {tier: {
            'count': 0,
            'subscription_revenue': 0,
            'reward_pool_contribution': 0
        } for tier in SubscriptionTier}
        
        for project in active_projects:
            project_metrics[project.tier]['count'] += 1
            project_metrics[project.tier]['subscription_revenue'] += project.subscription_fee
            project_metrics[project.tier]['reward_pool_contribution'] += project.get_reward_pool_contribution()
        
        # Calculate total trading volume
        total_volume = sum(trader.current_volume for trader in active_traders)
        
        # Calculate average earnings per trader type
        avg_earnings = {}
        for trader_type in TraderType:
            count = trader_metrics[trader_type]['count']
            rewards = trader_metrics[trader_type]['rewards']
            avg_earnings[trader_type] = rewards / count if count > 0 else 0
        
        return {
            'month': self.current_month,
            'active_traders': len(active_traders),
            'active_projects': len(active_projects),
            'total_volume': total_volume,
            'trader_metrics': trader_metrics,
            'project_metrics': project_metrics,
            'avg_earnings': avg_earnings
        }
    
    def _record_monthly_data(self, financials, metrics):
        """Record monthly data for later analysis and visualization"""
        # Flatten the nested dictionaries for easier DataFrame creation
        monthly_data = {**financials}
        
        # Add flattened metrics
        monthly_data['active_traders'] = metrics['active_traders']
        monthly_data['active_projects'] = metrics['active_projects']
        monthly_data['total_volume'] = metrics['total_volume']
        
        # Add trader metrics
        for trader_type in TraderType:
            type_name = trader_type.value
            monthly_data[f'{type_name}_count'] = metrics['trader_metrics'][trader_type]['count']
            monthly_data[f'{type_name}_volume'] = metrics['trader_metrics'][trader_type]['volume']
            monthly_data[f'{type_name}_rewards'] = metrics['trader_metrics'][trader_type]['rewards']
            monthly_data[f'{type_name}_avg_earnings'] = metrics['avg_earnings'][trader_type]
        
        # Add project metrics
        for tier in SubscriptionTier:
            tier_name = tier.value
            monthly_data[f'{tier_name}_count'] = metrics['project_metrics'][tier]['count']
            monthly_data[f'{tier_name}_revenue'] = metrics['project_metrics'][tier]['subscription_revenue']
            monthly_data[f'{tier_name}_reward_pool'] = metrics['project_metrics'][tier]['reward_pool_contribution']
        
        # Append to the monthly metrics DataFrame
        self.monthly_metrics = pd.concat([self.monthly_metrics, pd.DataFrame([monthly_data])], ignore_index=True)
    
    def run_simulation(self):
        """Run the simulation for the specified number of months"""
        for month in range(1, self.max_months + 1):
            self.current_month = month
            
            # Update existing traders and projects
            self._update_traders()
            self._update_projects()
            
            # Generate new traders and projects
            self._generate_new_traders()
            self._generate_new_projects()
            
            # Create new challenges for active projects
            for project_id, project in self.projects.items():
                if project.is_active:
                    # Each project has a chance to create a new challenge each month
                    if random.random() < 0.3:  # 30% chance
                        self._create_challenge(project_id)
            
            # Reset challenge participants for the new month
            for challenge in self.challenges.values():
                challenge.participants = {}
                challenge.total_volume = 0
            
            # Assign traders to challenges
            self._assign_traders_to_challenges()
            
            # Distribute rewards
            self._distribute_challenge_rewards()
            
            # Collect subscription fees
            monthly_revenue = self._collect_monthly_subscription_fees()
            
            # Calculate financials and metrics
            financials = self._calculate_monthly_financials(monthly_revenue)
            metrics = self._calculate_monthly_metrics()
            
            # Record data for analysis
            self._record_monthly_data(financials, metrics)
            
            # Выводим прогресс симуляции
            print(f"Месяц {month}: Выручка=${monthly_revenue:,.2f}, Прибыль=${financials['monthly_profit']:,.2f}, "
                  f"Активных трейдеров={metrics['active_traders']}, Активных проектов={metrics['active_projects']}")
        
        return self.monthly_metrics
    
    def calculate_break_even_point(self):
        """Calculate the month when the platform breaks even"""
        for i, row in self.monthly_metrics.iterrows():
            if row['cumulative_profit'] >= 0:
                month = row['month']
                print(f"\nТочка безубыточности достигнута в месяце {month}")
                return month
        
        print("\nТочка безубыточности не достигнута в течение периода симуляции")
        return None  # Never breaks even within the simulation period
        
    def _print_simulation_summary(self):
        """Print a summary of the simulation results"""
        print("\nСимуляция успешно завершена!")
        
        # Отображаем сравнение типов трейдеров
        print("\nПодробная информация о типах трейдеров:")
        self.display_trader_type_comparison()
    
    def get_revenue_by_tier(self, month):
        """Get revenue breakdown by subscription tier for a specific month"""
        if self.monthly_metrics.empty or month > self.max_months:
            return None
        
        # Получаем данные за указанный месяц
        month_data = self.monthly_metrics[self.monthly_metrics['month'] == month]
        if month_data.empty:
            return None
        
        # Создаем DataFrame с данными о доходах по типам подписки
        revenue_data = []
        for tier in SubscriptionTier:
            tier_name = tier.value
            # Получаем количество проектов данного типа
            count_col = f'{tier_name}_count'
            revenue_col = f'{tier_name}_revenue'
            
            if count_col in month_data.columns:
                count = month_data.iloc[0][count_col]
                # Если revenue_col отсутствует, вычисляем доход на основе количества и цены подписки
                if revenue_col in month_data.columns:
                    revenue = month_data.iloc[0][revenue_col]
                else:
                    # Получаем цену подписки из первого проекта данного типа или используем значения по умолчанию
                    subscription_fees = {
                        SubscriptionTier.STANDARD: 2000,
                        SubscriptionTier.PREMIUM: 5000,
                        SubscriptionTier.ENTERPRISE: 10000
                    }
                    revenue = count * subscription_fees[tier]
                
                revenue_data.append({'tier': tier_name, 'count': count, 'revenue': revenue})
        
        return pd.DataFrame(revenue_data)
    
    def get_rewards_by_type(self, month):
        """Get rewards distribution by reward type for a specific month"""
        if self.monthly_metrics.empty or month > self.max_months:
            return None
        
        # Получаем данные за указанный месяц
        month_data = self.monthly_metrics[self.monthly_metrics['month'] == month]
        if month_data.empty:
            return None
        
        # Получаем общий пул вознаграждений за месяц
        # Если колонка reward_pool отсутствует, вычисляем на основе месячной выручки и процента распределения
        if 'reward_pool' in month_data.columns:
            total_reward_pool = month_data.iloc[0]['reward_pool']
        else:
            # Используем месячную выручку и процент распределения пула вознаграждений
            monthly_revenue = month_data.iloc[0]['monthly_revenue'] if 'monthly_revenue' in month_data.columns else 0
            reward_pool_allocation = self.reward_pool_allocation
            total_reward_pool = monthly_revenue * reward_pool_allocation
        
        # Создаем DataFrame с данными о распределении вознаграждений по типам
        rewards_data = []
        
        # Используем распределение вознаграждений из параметров или значения по умолчанию
        reward_distribution = self.params.get('reward_distribution', {
            RewardType.VOLUME_LEADERBOARD: 0.55,
            RewardType.RANDOM_DRAW: 0.15,
            RewardType.MILESTONE: 0.25,
            RewardType.SPECIAL_EVENT: 0.05
        })
        
        for reward_type, percentage in reward_distribution.items():
            reward_name = reward_type.value
            amount = total_reward_pool * percentage
            rewards_data.append({'reward_type': reward_name, 'percentage': percentage * 100, 'amount': amount})
        
        return pd.DataFrame(rewards_data)
    
    def calculate_lifetime_values(self):
        """Calculate lifetime values for different trader types and project tiers"""
        # Calculate average months active for each trader type
        trader_months_active = {trader_type: [] for trader_type in TraderType}
        for trader in self.traders.values():
            trader_months_active[trader.type].append(trader.months_active)
        
        avg_months_active = {}
        for trader_type, months in trader_months_active.items():
            avg_months_active[trader_type] = sum(months) / len(months) if months else 0
        
        # Calculate average revenue per month for each trader type
        last_month = self.monthly_metrics.iloc[-1] if not self.monthly_metrics.empty else None
        trader_ltv = {}
        
        for trader_type in TraderType:
            type_name = trader_type.value
            if last_month is not None:
                avg_monthly_revenue = last_month[f'{type_name}_volume'] * 0.001  # Simplified revenue model
                trader_ltv[trader_type] = avg_monthly_revenue * avg_months_active[trader_type]
            else:
                trader_ltv[trader_type] = 0
        
        # Calculate project LTV
        project_months_active = {tier: [] for tier in SubscriptionTier}
        for project in self.projects.values():
            project_months_active[project.tier].append(project.months_active)
        
        avg_project_months_active = {}
        for tier, months in project_months_active.items():
            avg_project_months_active[tier] = sum(months) / len(months) if months else 0
        
        project_ltv = {}
        for tier in SubscriptionTier:
            subscription_fee = next((p.subscription_fee for p in self.projects.values() if p.tier == tier), 0)
            project_ltv[tier] = subscription_fee * avg_project_months_active[tier]
        
        return {
            'trader_ltv': trader_ltv,
            'project_ltv': project_ltv
        }
    
    def perform_sensitivity_analysis(self, parameter_ranges):
        """Perform sensitivity analysis by varying key parameters"""
        results = []
        base_params = self.params.copy()
        
        for param_name, param_values in parameter_ranges.items():
            for value in param_values:
                # Create a new set of parameters with just this one changed
                test_params = base_params.copy()
                test_params[param_name] = value
                
                # Run a new simulation with these parameters
                sim = TradeFusionSimulation(test_params)
                sim.run_simulation()
                
                # Get key metrics at month 24 (or the last month if shorter)
                last_month = min(24, len(sim.monthly_metrics))
                if last_month > 0:
                    month_data = sim.monthly_metrics.iloc[last_month-1]
                    
                    results.append({
                        'parameter': param_name,
                        'value': value,
                        'monthly_revenue': month_data['monthly_revenue'],
                        'monthly_profit': month_data['monthly_profit'],
                        'active_traders': month_data['active_traders'],
                        'active_projects': month_data['active_projects'],
                        'break_even_month': sim.calculate_break_even_point()
                    })
        
        return pd.DataFrame(results)
    
    # Visualization Methods
    def visualize_financial_metrics(self, save_path=None):
        """Create line graphs showing monthly revenue, costs, and profit over time"""
        plt.figure(figsize=(12, 8))
        
        # Plot revenue, costs, and profit
        plt.subplot(2, 1, 1)
        plt.plot(self.monthly_metrics['month'], self.monthly_metrics['monthly_revenue'], 'b-', label='Revenue')
        plt.plot(self.monthly_metrics['month'], self.monthly_metrics['total_monthly_cost'], 'r-', label='Costs')
        plt.plot(self.monthly_metrics['month'], self.monthly_metrics['monthly_profit'], 'g-', label='Profit')
        plt.title('Monthly Financial Metrics')
        plt.xlabel('Month')
        plt.ylabel('Amount ($)')
        plt.legend()
        plt.grid(True)
        
        # Plot cumulative profit
        plt.subplot(2, 1, 2)
        plt.plot(self.monthly_metrics['month'], self.monthly_metrics['cumulative_profit'], 'g-')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Cumulative Profit')
        plt.xlabel('Month')
        plt.ylabel('Amount ($)')
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/financial_metrics.png")
        
        plt.show()
    
    def visualize_user_growth(self, save_path=None):
        """Create line graphs showing user growth by trader type"""
        plt.figure(figsize=(12, 8))
        
        # Plot total active traders
        plt.subplot(2, 1, 1)
        plt.plot(self.monthly_metrics['month'], self.monthly_metrics['active_traders'], 'b-')
        plt.title('Total Active Traders')
        plt.xlabel('Month')
        plt.ylabel('Number of Traders')
        plt.grid(True)
        
        # Plot traders by type
        plt.subplot(2, 1, 2)
        for trader_type in TraderType:
            type_name = trader_type.value
            plt.plot(self.monthly_metrics['month'], self.monthly_metrics[f'{type_name}_count'], 
                     label=type_name)
        
        plt.title('Traders by Type')
        plt.xlabel('Month')
        plt.ylabel('Number of Traders')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/user_growth.png")
        
        plt.show()
    
    def visualize_project_growth(self, save_path=None):
        """Create line graphs showing project growth by subscription tier"""
        plt.figure(figsize=(12, 8))
        
        # Plot total active projects
        plt.subplot(2, 1, 1)
        plt.plot(self.monthly_metrics['month'], self.monthly_metrics['active_projects'], 'b-')
        plt.title('Total Active Projects')
        plt.xlabel('Month')
        plt.ylabel('Number of Projects')
        plt.grid(True)
        
        # Plot projects by tier
        plt.subplot(2, 1, 2)
        for tier in SubscriptionTier:
            tier_name = tier.value
            plt.plot(self.monthly_metrics['month'], self.monthly_metrics[f'{tier_name}_count'], 
                     label=tier_name)
        
        plt.title('Projects by Tier')
        plt.xlabel('Month')
        plt.ylabel('Number of Projects')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/project_growth.png")
        
        plt.show()
    
    def visualize_trader_earnings(self, save_path=None):
        """Create line graphs showing average earnings per trader type"""
        plt.figure(figsize=(14, 10))
        
        # График средних доходов по типам трейдеров
        plt.subplot(2, 1, 1)
        for trader_type in TraderType:
            type_name = trader_type.value
            plt.plot(self.monthly_metrics['month'], self.monthly_metrics[f'{type_name}_avg_earnings'], 
                     label=type_name, linewidth=2)
        
        plt.title('Средний доход по типам трейдеров', fontsize=14)
        plt.xlabel('Месяц')
        plt.ylabel('Средний доход ($)')
        plt.legend()
        plt.grid(True)
        
        # График ROI (доход на единицу объема) по типам трейдеров
        plt.subplot(2, 1, 2)
        
        # Рассчитываем ROI для каждого месяца и типа трейдера
        roi_data = {}
        for trader_type in TraderType:
            type_name = trader_type.value
            roi_data[type_name] = []
            
            for _, row in self.monthly_metrics.iterrows():
                volume = row[f'{type_name}_volume']
                earnings = row[f'{type_name}_avg_earnings'] * row[f'{type_name}_count']
                roi = (earnings / volume * 100) if volume > 0 else 0
                roi_data[type_name].append(roi)
        
        # Строим график ROI
        for trader_type in TraderType:
            type_name = trader_type.value
            plt.plot(self.monthly_metrics['month'], roi_data[type_name], 
                     label=f'{type_name} ROI', linewidth=2)
        
        plt.title('ROI по типам трейдеров (доход на единицу объема)', fontsize=14)
        plt.xlabel('Месяц')
        plt.ylabel('ROI (%)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/trader_earnings.png")
        
        plt.show()
        
    def display_trader_type_comparison(self):
        """Отображает подробное сравнение типов трейдеров с их характеристиками"""
        # Создаем таблицу с описанием типов трейдеров
        trader_types_info = {
            'Тип трейдера': [t.value for t in TraderType],
            'Месячный объем': ['$5,000-$10,000', '$50,000-$100,000', '$500,000-$1,000,000', '$5,000,000-$10,000,000'],
            'Описание': [
                'Начинающие трейдеры или трейдеры с частичной занятостью',
                'Регулярные трейдеры со средним объемом',
                'Профессиональные трейдеры, для которых торговля - основной доход',
                'Институциональные игроки с очень высоким объемом'
            ],
            'Участие в челленджах': ['Низкое (1-3)', 'Среднее (3-5)', 'Высокое (5-8)', 'Очень высокое (8-10)'],
            'Средний ROI': ['Средний', 'Выше среднего', 'Высокий', 'Очень высокий']
        }
        
        # Создаем DataFrame
        comparison_df = pd.DataFrame(trader_types_info)
        
        # Если есть данные о доходах, добавляем их
        if not self.monthly_metrics.empty:
            last_month = self.monthly_metrics.iloc[-1]
            avg_earnings = [last_month[f'{t.value}_avg_earnings'] for t in TraderType]
            comparison_df['Средний доход'] = [f'${e:,.2f}' for e in avg_earnings]
        
        # Выводим таблицу
        print("\nСравнение типов трейдеров:\n")
        print(comparison_df.to_string(index=False))
        print("\nРазличия между типами трейдеров:")
        print("- Casual: Начинающие трейдеры с небольшими объемами, часто новички на платформе")
        print("- Active: Регулярные трейдеры со средними объемами, активно участвуют в челленджах")
        print("- Professional: Опытные трейдеры с высокими объемами, для которых торговля - основной источник дохода")
        print("- Whale: Крупнейшие игроки с очень высокими объемами, часто институциональные клиенты")
        
        return comparison_df
    
    def visualize_revenue_breakdown(self, month=None, save_path=None):
        """Create bar chart showing monthly revenue breakdown by subscription tier"""
        if month is None:
            month = self.max_months
        
        if month <= len(self.monthly_metrics):
            month_data = self.monthly_metrics.iloc[month-1]
            
            # Get revenue by tier
            tier_revenues = []
            tier_names = []
            
            for tier in SubscriptionTier:
                tier_name = tier.value
                tier_names.append(tier_name)
                tier_revenues.append(month_data[f'{tier_name}_revenue'])
            
            plt.figure(figsize=(10, 6))
            plt.bar(tier_names, tier_revenues, color=['blue', 'green', 'red'])
            plt.title(f'Revenue Breakdown by Subscription Tier (Month {month})')
            plt.xlabel('Subscription Tier')
            plt.ylabel('Revenue ($)')
            
            # Add revenue values on top of bars
            for i, revenue in enumerate(tier_revenues):
                plt.text(i, revenue + 100, f'${revenue:,.0f}', ha='center')
            
            if save_path:
                plt.savefig(f"{save_path}/revenue_breakdown_month_{month}.png")
            
            plt.show()
    
    def visualize_reward_distribution(self, month=None, save_path=None):
        """Create bar chart showing monthly reward pool distribution"""
        if month is None:
            month = self.max_months
        
        if month <= len(self.monthly_metrics):
            month_data = self.monthly_metrics.iloc[month-1]
            
            # Define reward distribution percentages
            reward_types = ['Volume Leaderboard', 'Random Draw', 'Milestone Achievement', 'Special Event']
            percentages = [0.6, 0.2, 0.15, 0.05]  # As defined in the challenge distribution
            
            # Calculate actual amounts
            total_reward_pool = month_data['reward_pool']
            amounts = [total_reward_pool * pct for pct in percentages]
            
            plt.figure(figsize=(10, 6))
            plt.bar(reward_types, amounts, color=['blue', 'green', 'orange', 'red'])
            plt.title(f'Reward Pool Distribution (Month {month})')
            plt.xlabel('Reward Type')
            plt.ylabel('Amount ($)')
            
            # Add percentage and amount on top of bars
            for i, (amount, pct) in enumerate(zip(amounts, percentages)):
                plt.text(i, amount + 100, f'${amount:,.0f} ({pct*100:.0f}%)', ha='center')
            
            if save_path:
                plt.savefig(f"{save_path}/reward_distribution_month_{month}.png")
            
            plt.show()
    
    def create_monthly_financial_summary(self, save_path=None):
        """Create a table showing monthly financial summary"""
        # Select relevant columns for the financial summary
        financial_cols = [
            'month', 'monthly_revenue', 'operational_cost', 'marketing_cost', 
            'reward_pool', 'referral_commissions', 'total_monthly_cost', 
            'monthly_profit', 'cumulative_profit'
        ]
        
        financial_summary = self.monthly_metrics[financial_cols].copy()
        
        # Format the table for display
        plt.figure(figsize=(14, 8))
        plt.axis('off')
        plt.title('Monthly Financial Summary')
        
        # Create the table
        table = plt.table(
            cellText=financial_summary.values,
            colLabels=financial_summary.columns,
            cellLoc='center',
            loc='center',
            colWidths=[0.1] * len(financial_cols)
        )
        
        # Adjust table appearance
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        if save_path:
            plt.savefig(f"{save_path}/monthly_financial_summary.png", bbox_inches='tight')
        
        plt.show()
        
        return financial_summary
    
    def create_trader_economics_summary(self, save_path=None):
        """Create a table showing user economics by trader type"""
        # Get the last month data for the summary
        if self.monthly_metrics.empty:
            return pd.DataFrame()
            
        last_month = self.monthly_metrics.iloc[-1]
        
        # Calculate lifetime values
        ltv_data = self.calculate_lifetime_values()
        trader_ltv = ltv_data['trader_ltv']
        
        # Расчет дополнительных метрик для каждого типа трейдеров
        trader_details = {}
        for trader_type in TraderType:
            type_traders = [t for t in self.traders.values() if t.is_active and t.type == trader_type]
            
            if type_traders:
                avg_volume = sum(t.current_volume for t in type_traders) / len(type_traders)
                avg_rewards = sum(t.total_rewards for t in type_traders) / len(type_traders)
                avg_months = sum(t.months_active for t in type_traders) / len(type_traders)
                avg_challenges = sum(t.challenges_participated for t in type_traders) / len(type_traders)
                
                # Расчет среднемесячного дохода (вознаграждения / месяцы активности)
                avg_monthly_earnings = avg_rewards / avg_months if avg_months > 0 else 0
                
                # Расчет ROI (доход на единицу объема)
                roi = (avg_rewards / avg_volume) * 100 if avg_volume > 0 else 0
                
                trader_details[trader_type] = {
                    'avg_volume': avg_volume,
                    'avg_rewards': avg_rewards,
                    'avg_months': avg_months,
                    'avg_challenges': avg_challenges,
                    'avg_monthly_earnings': avg_monthly_earnings,
                    'roi': roi
                }
            else:
                trader_details[trader_type] = {
                    'avg_volume': 0, 'avg_rewards': 0, 'avg_months': 0, 
                    'avg_challenges': 0, 'avg_monthly_earnings': 0, 'roi': 0
                }
        
        # Create a DataFrame for the summary with expanded metrics
        trader_economics = pd.DataFrame({
            'Trader Type': [t.value for t in TraderType],
            'Count': [last_month[f'{t.value}_count'] for t in TraderType],
            'Monthly Volume': [last_month[f'{t.value}_volume'] for t in TraderType],
            'Avg Volume Per Trader': [trader_details[t]['avg_volume'] for t in TraderType],
            'Avg Total Rewards': [trader_details[t]['avg_rewards'] for t in TraderType],
            'Avg Monthly Earnings': [trader_details[t]['avg_monthly_earnings'] for t in TraderType],
            'ROI (%)': [trader_details[t]['roi'] for t in TraderType],
            'Avg Challenges': [trader_details[t]['avg_challenges'] for t in TraderType],
            'Lifetime Value': [trader_ltv[t] for t in TraderType]
        })
        
        # Добавляем описание различий между типами трейдеров
        trader_type_descriptions = {
            TraderType.CASUAL: "Трейдеры с небольшим объемом ($5K-$10K в месяц), обычно новички или частичная занятость",
            TraderType.ACTIVE: "Активные трейдеры со средним объемом ($50K-$100K в месяц), регулярно торгуют",
            TraderType.PROFESSIONAL: "Профессиональные трейдеры с высоким объемом ($500K-$1M в месяц), торговля - основной доход",
            TraderType.WHALE: "Крупнейшие трейдеры с очень высоким объемом ($5M-$10M в месяц), институциональные игроки"
        }
        
        # Добавляем колонку с описаниями
        trader_economics['Description'] = [trader_type_descriptions[t] for t in TraderType]
        
        # Format the table for display
        plt.figure(figsize=(12, 6))
        plt.axis('off')
        plt.title('Trader Economics by Type')
        
        # Create the table
        table = plt.table(
            cellText=trader_economics.values,
            colLabels=trader_economics.columns,
            cellLoc='center',
            loc='center',
            colWidths=[0.2, 0.15, 0.2, 0.2, 0.2]
        )
        
        # Adjust table appearance
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        if save_path:
            plt.savefig(f"{save_path}/trader_economics_summary.png", bbox_inches='tight')
        
        plt.show()
        
        return trader_economics
    
    def create_project_economics_summary(self, save_path=None):
        """Create a table showing project economics by subscription tier"""
        # Get the last month data for the summary
        if self.monthly_metrics.empty:
            return pd.DataFrame()
            
        last_month = self.monthly_metrics.iloc[-1]
        
        # Calculate lifetime values
        ltv_data = self.calculate_lifetime_values()
        project_ltv = ltv_data['project_ltv']
        
        # Create a DataFrame for the summary
        project_economics = pd.DataFrame({
            'Subscription Tier': [t.value for t in SubscriptionTier],
            'Count': [last_month[f'{t.value}_count'] for t in SubscriptionTier],
            'Monthly Revenue': [last_month[f'{t.value}_revenue'] for t in SubscriptionTier],
            'Reward Pool Contribution': [last_month[f'{t.value}_reward_pool'] for t in SubscriptionTier],
            'Lifetime Value': [project_ltv[t] for t in SubscriptionTier]
        })
        
        # Format the table for display
        plt.figure(figsize=(12, 6))
        plt.axis('off')
        plt.title('Project Economics by Subscription Tier')
        
        # Create the table
        table = plt.table(
            cellText=project_economics.values,
            colLabels=project_economics.columns,
            cellLoc='center',
            loc='center',
            colWidths=[0.2, 0.15, 0.2, 0.25, 0.2]
        )
        
        # Adjust table appearance
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        if save_path:
            plt.savefig(f"{save_path}/project_economics_summary.png", bbox_inches='tight')
        
        plt.show()
        
        return project_economics