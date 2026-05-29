"""
Модуль с методами штрафных и барьерных функций
"""

import numpy as np
from typing import Callable, List, Dict, Optional, Tuple
from scipy.optimize import minimize

from .functions import (
    penalty_function, 
    barrier_function, 
    combined_penalty_barrier,
    check_constraints,
    find_feasible_point
)


# Импортируем методы безусловной оптимизации из ЛР3
try:
    from optimization import lbfgs, bfgs
except ImportError:
    # Если пакет optimization недоступен, используем простую реализацию
    def lbfgs(f, grad_f, x0, **kwargs):
        result = minimize(f, x0, method='L-BFGS-B', jac=grad_f)
        return {
            'x_star': result.x,
            'f_star': result.fun,
            'iterations': result.nit
        }


def penalty_method(f: Callable, grad_f: Callable, x0: np.ndarray,
                   constraints: List[Callable],
                   r0: float = 1.0, C: float = 10.0,
                   eps: float = 1e-6, max_outer_iter: int = 50,
                   max_inner_iter: int = 100) -> Dict:
    """
    Метод штрафных функций (внешних штрафов).
    
    Args:
        f: Целевая функция
        grad_f: Градиент целевой функции
        x0: Начальная точка
        constraints: Список функций ограничений g_j(x) ≤ 0
        r0: Начальный параметр штрафа
        C: Коэффициент увеличения штрафа (C > 1)
        eps: Точность остановки
        max_outer_iter: Макс. число внешних итераций
        max_inner_iter: Макс. число итераций внутренней оптимизации
    
    Returns:
        Словарь с результатами оптимизации
    """
    history = {
        'x': [x0.copy()],
        'f': [f(x0)],
        'r': [r0],
        'penalty': [penalty_function(x0, constraints, r0)],
        'feasible': [check_constraints(x0, constraints)[0]]
    }
    
    x = x0.copy()
    r = r0
    
    for k in range(max_outer_iter):
        # Вспомогательная функция
        def F(x):
            return f(x) + penalty_function(x, constraints, r)
        
        def grad_F(x):
            return grad_f(x) + penalty_gradient_approx(x, constraints, r)
        
        # Безусловная минимизация F(x)
        result = lbfgs(F, grad_F, x, max_iter=max_inner_iter)
        x_new = result['x_star']
        
        # Проверка окончания
        penalty_val = penalty_function(x_new, constraints, r)
        feasible, _ = check_constraints(x_new, constraints)
        
        history['x'].append(x_new.copy())
        history['f'].append(f(x_new))
        history['r'].append(r)
        history['penalty'].append(penalty_val)
        history['feasible'].append(feasible)
        
        if penalty_val <= eps:
            return {
                'x_star': x_new,
                'f_star': f(x_new),
                'iterations': k + 1,
                'penalty': penalty_val,
                'feasible': feasible,
                'history': history,
                'method': 'penalty'
            }
        
        # Увеличение параметра штрафа
        r = r * C
        x = x_new
    
    # Возвращаем лучшее найденное решение
    return {
        'x_star': x,
        'f_star': f(x),
        'iterations': max_outer_iter,
        'penalty': penalty_function(x, constraints, r),
        'feasible': check_constraints(x, constraints)[0],
        'history': history,
        'method': 'penalty',
        'converged': False
    }


def barrier_method(f: Callable, grad_f: Callable, x0: np.ndarray,
                   constraints: List[Callable],
                   r0: float = 1.0, C: float = 10.0,
                   eps: float = 1e-6, max_outer_iter: int = 50,
                   max_inner_iter: int = 100,
                   barrier_type: str = 'log') -> Dict:
    """
    Метод барьерных функций (внутренних штрафов).
    Согласно методичке: r^{k+1} = r^k / C, C = 10

    Args:
        f: Целевая функция
        grad_f: Градиент целевой функции
        x0: Начальная точка (должна быть внутри допустимой области!)
        constraints: Список функций ограничений g_j(x) ≤ 0
        r0: Начальный параметр барьера
        C: Коэффициент уменьшения барьера (C > 1, обычно C=10)
        eps: Точность остановки
        max_outer_iter: Макс. число внешних итераций
        max_inner_iter: Макс. число итераций внутренней оптимизации
        barrier_type: 'log' или 'inverse'

    Returns:
        Словарь с результатами оптимизации
    """
    # Проверяем, что начальная точка допустима
    feasible, info = check_constraints(x0, constraints)
    if not feasible:
        print(f"Начальная точка недопустима! Нарушения: {info['violations']}")
        print("Попытка найти допустимую точку...")
        x0 = find_feasible_point(x0, constraints)
        feasible, _ = check_constraints(x0, constraints)
        if not feasible:
            raise ValueError("Не удалось найти допустимую начальную точку")

    # Используем точку ближе к границе для лучшей сходимости
    x = x0.copy() * 0.9 + np.array([1.0, 1.0]) * 0.1  # Обновлено для n=2
    
    r = r0
    
    history = {
        'x': [x.copy()],
        'f': [f(x)],
        'r': [r],
        'barrier': [abs(barrier_function(x, constraints, r, barrier_type))],
        'feasible': [check_constraints(x, constraints)[0]]
    }

    for k in range(max_outer_iter):
        # Вспомогательная функция
        def F(x):
            return f(x) + barrier_function(x, constraints, r, barrier_type)
        
        def grad_F(x):
            return grad_f(x) + barrier_gradient_approx(x, constraints, r, barrier_type)
        
        # Безусловная минимизация F(x)
        result = lbfgs(F, grad_F, x, max_iter=max_inner_iter)
        x_new = result['x_star']
        
        # Проверка окончания
        barrier_val = barrier_function(x_new, constraints, r, barrier_type)
        feasible, _ = check_constraints(x_new, constraints)
        
        history['x'].append(x_new.copy())
        history['f'].append(f(x_new))
        history['r'].append(r)
        history['barrier'].append(abs(barrier_val))
        history['feasible'].append(feasible)
        
        if abs(barrier_val) <= eps:
            return {
                'x_star': x_new,
                'f_star': f(x_new),
                'iterations': k + 1,
                'barrier': abs(barrier_val),
                'feasible': feasible,
                'history': history,
                'method': 'barrier'
            }
        
        # Уменьшение параметра барьера
        r = r / C
        x = x_new
    
    return {
        'x_star': x,
        'f_star': f(x),
        'iterations': max_outer_iter,
        'barrier': abs(barrier_function(x, constraints, r, barrier_type)),
        'feasible': check_constraints(x, constraints)[0],
        'history': history,
        'method': 'barrier',
        'converged': False
    }


def combined_penalty_barrier_method(f: Callable, grad_f: Callable, 
                                     x0: np.ndarray,
                                     constraints: List[Callable],
                                     r0: float = 1.0, C: float = 10.0,
                                     eps: float = 1e-6, 
                                     max_outer_iter: int = 50,
                                     max_inner_iter: int = 100,
                                     alpha: float = 0.5) -> Dict:
    """
    Комбинированный метод штрафных и барьерных функций.
    
    Args:
        f: Целевая функция
        grad_f: Градиент целевой функции
        x0: Начальная точка
        constraints: Список функций ограничений
        r0: Начальный параметр
        C: Коэффициент изменения параметра
        eps: Точность остановки
        max_outer_iter: Макс. число внешних итераций
        max_inner_iter: Макс. число итераций внутренней оптимизации
        alpha: Коэффициент комбинации (0 < alpha < 1)
    
    Returns:
        Словарь с результатами оптимизации
    """
    history = {
        'x': [x0.copy()],
        'f': [f(x0)],
        'r': [r0],
        'feasible': [check_constraints(x0, constraints)[0]]
    }
    
    x = x0.copy()
    r = r0
    
    for k in range(max_outer_iter):
        # Вспомогательная функция
        def F(x):
            return f(x) + combined_penalty_barrier(x, constraints, r, alpha)
        
        # Безусловная минимизация
        result = lbfgs(F, grad_f, x, max_iter=max_inner_iter)
        x_new = result['x_star']
        
        # Проверка окончания
        feasible, info = check_constraints(x_new, constraints)
        penalty_val = penalty_function(x_new, constraints, r)
        
        history['x'].append(x_new.copy())
        history['f'].append(f(x_new))
        history['r'].append(r)
        history['feasible'].append(feasible)
        
        if penalty_val <= eps and feasible:
            return {
                'x_star': x_new,
                'f_star': f(x_new),
                'iterations': k + 1,
                'penalty': penalty_val,
                'feasible': feasible,
                'history': history,
                'method': 'combined'
            }
        
        # Изменение параметра
        r = r * C
        x = x_new
    
    return {
        'x_star': x,
        'f_star': f(x),
        'iterations': max_outer_iter,
        'penalty': penalty_function(x, constraints, r),
        'feasible': check_constraints(x, constraints)[0],
        'history': history,
        'method': 'combined',
        'converged': False
    }


# ============================================================================
# Вспомогательные функции для градиентов
# ============================================================================

def penalty_gradient_approx(x: np.ndarray, constraints: List[Callable], 
                            r: float, eps: float = 1e-8) -> np.ndarray:
    """Аппроксимация градиента штрафной функции конечными разностями."""
    n = len(x)
    grad = np.zeros(n)
    
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        
        grad[i] = (penalty_function(x_plus, constraints, r) - 
                   penalty_function(x_minus, constraints, r)) / (2 * eps)
    
    return grad


def barrier_gradient_approx(x: np.ndarray, constraints: List[Callable], 
                            r: float, barrier_type: str = 'log',
                            eps: float = 1e-8) -> np.ndarray:
    """Аппроксимация градиента барьерной функции конечными разностями."""
    n = len(x)
    grad = np.zeros(n)
    
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        
        grad[i] = (barrier_function(x_plus, constraints, r, barrier_type) - 
                   barrier_function(x_minus, constraints, r, barrier_type)) / (2 * eps)
    
    return grad