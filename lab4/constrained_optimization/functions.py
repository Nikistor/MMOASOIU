"""
Модуль с тестовыми функциями, ограничениями и штрафными функциями
Вариант 3: a=80, b=3, f0=110, n=2
"""

import numpy as np
from typing import Callable, Tuple, List, Dict


# ============================================================================
# Параметры варианта 3
# ============================================================================
DEFAULT_A = 80
DEFAULT_B = 3
DEFAULT_F0 = 110
DEFAULT_N = 2


def rosenbrock(x: np.ndarray, a: float = DEFAULT_A, b: float = DEFAULT_B, 
               f0: float = DEFAULT_F0) -> float:
    """
    Функция Розенброка для произвольной размерности n.
    
    f(x) = Σ[i=1 to n-1] [a(x_i² - x_{i+1})² + b(x_i - 1)²] + f0
    """
    n = len(x)
    result = f0
    for i in range(n - 1):
        result += a * (x[i]**2 - x[i+1])**2 + b * (x[i] - 1)**2
    return result


def rosenbrock_gradient(x: np.ndarray, a: float = DEFAULT_A, 
                        b: float = DEFAULT_B) -> np.ndarray:
    """Градиент функции Розенброка."""
    n = len(x)
    grad = np.zeros(n)
    
    for i in range(n - 1):
        grad[i] += 2 * a * (x[i]**2 - x[i+1]) * 2 * x[i] + 2 * b * (x[i] - 1)
        grad[i+1] += -2 * a * (x[i]**2 - x[i+1])
    
    return grad


def rosenbrock_hessian(x: np.ndarray, a: float = DEFAULT_A, 
                       b: float = DEFAULT_B) -> np.ndarray:
    """Матрица Гессе функции Розенброка."""
    n = len(x)
    H = np.zeros((n, n))
    
    for i in range(n - 1):
        H[i, i] += 2 * a * (2 * x[i]**2 + 2 * (x[i]**2 - x[i+1]))
        H[i, i+1] += -4 * a * x[i]
        H[i+1, i] += -4 * a * x[i]
        H[i+1, i+1] += 2 * a
    
    for i in range(n - 1):
        H[i, i] += 2 * b
    
    return H


def create_rosenbrock_function(a: float = DEFAULT_A, b: float = DEFAULT_B, 
                               f0: float = DEFAULT_F0) -> tuple:
    """Создать функцию Розенброка с заданными параметрами."""
    def f(x):
        return rosenbrock(x, a, b, f0)
    
    def grad(x):
        return rosenbrock_gradient(x, a, b)
    
    def hess(x):
        return rosenbrock_hessian(x, a, b)
    
    return f, grad, hess


# ============================================================================
# Ограничения для варианта 3
# ============================================================================

def get_constraints_variant3() -> List[Callable]:
    """
    Получить список функций ограничений для варианта 3.
    
    g1(x) = x1² - x2² - 3 ≤ 0
    g2(x) = -x1 ≤ 0
    g3(x) = -x2 ≤ 0
    """
    def g1(x):
        return x[0]**2 - x[1]**2 - 3
    
    def g2(x):
        return -x[0]
    
    def g3(x):
        return -x[1]
    
    return [g1, g2, g3]


def get_constraints(variant: int = 3) -> List[Callable]:
    """Получить ограничения для указанного варианта."""
    if variant == 3:
        return get_constraints_variant3()
    else:
        raise ValueError(f"Вариант {variant} не реализован")


def check_constraints(x: np.ndarray, constraints: List[Callable], 
                      tol: float = 1e-6) -> Tuple[bool, Dict]:
    """
    Проверить выполнение ограничений в точке x.
    
    Returns:
        feasible: True если все ограничения выполнены
        info: словарь с информацией о нарушениях
    """
    violations = []
    max_violation = 0.0
    
    for i, g in enumerate(constraints):
        g_val = g(x)
        if g_val > tol:
            violations.append((i, g_val))
            max_violation = max(max_violation, g_val)
    
    return len(violations) == 0, {
        'violations': violations,
        'max_violation': max_violation,
        'feasible': len(violations) == 0
    }


# ============================================================================
# Штрафные и барьерные функции
# ============================================================================

def penalty_function(x: np.ndarray, constraints: List[Callable], 
                     r: float = 1.0) -> float:
    """
    Квадратичная штрафная функция для ограничений-неравенств.
    
    P(x, r) = (r/2) * Σ[max(0, g_j(x))]²
    """
    penalty = 0.0
    for g in constraints:
        g_val = g(x)
        if g_val > 0:
            penalty += g_val ** 2
    return (r / 2) * penalty


def penalty_gradient(x: np.ndarray, constraints: List[Callable], 
                     grad_constraints: List[Callable],
                     r: float = 1.0) -> np.ndarray:
    """
    Градиент штрафной функции.
    """
    n = len(x)
    grad = np.zeros(n)
    
    for i, (g, grad_g) in enumerate(zip(constraints, grad_constraints)):
        g_val = g(x)
        if g_val > 0:
            grad += r * g_val * grad_g(x)
    
    return grad


def barrier_function(x: np.ndarray, constraints: List[Callable], 
                     r: float = 1.0, 
                     type: str = 'log') -> float:
    """
    Барьерная функция для ограничений-неравенств.
    
    type='log': P(x, r) = -r * Σ ln[-g_j(x)]
    type='inverse': P(x, r) = -r * Σ 1/g_j(x)
    """
    barrier = 0.0
    
    for g in constraints:
        g_val = g(x)
        if g_val >= 0:
            return np.inf  # Точка вне допустимой области
        
        if type == 'log':
            barrier += np.log(-g_val)
        elif type == 'inverse':
            barrier += 1.0 / g_val
    
    return -r * barrier


def combined_penalty_barrier(x: np.ndarray, constraints: List[Callable],
                             r: float = 1.0, 
                             alpha: float = 0.5) -> float:
    """
    Комбинированная штрафно-барьерная функция.
    
    F(x, r) = f(x) + alpha * P_penalty(x, r) + (1-alpha) * P_barrier(x, r)
    """
    penalty = penalty_function(x, constraints, r)
    barrier = barrier_function(x, constraints, r, type='log')
    
    if np.isinf(barrier):
        return penalty  # Используем только штраф если точка вне области
    
    return alpha * penalty + (1 - alpha) * barrier


# ============================================================================
# Вспомогательные функции
# ============================================================================

def is_feasible(x: np.ndarray, constraints: List[Callable], 
                tol: float = 1e-6) -> bool:
    """Проверить, является ли точка допустимой."""
    feasible, _ = check_constraints(x, constraints, tol)
    return feasible


def find_feasible_point(x0: np.ndarray, constraints: List[Callable], 
                        max_iter: int = 100) -> np.ndarray:
    """
    Найти допустимую точку, начиная с x0.
    Простой метод: проекция на допустимую область.
    """
    x = x0.copy()
    n = len(x)
    
    # Для ограничений g2, g3 (неотрицательность)
    for i in range(n):
        if x[i] < 0:
            x[i] = 0.1  # Небольшое положительное значение
    
    # Проверка и коррекция для g1
    for _ in range(max_iter):
        g1_val = x[0]**2 - x[1]**2 - 3
        if g1_val <= 0:
            break
        # Увеличиваем x1 или уменьшаем x2 чтобы удовлетворить g1
        x[0] = np.sqrt(x[1]**2 + 3) + 0.1
    
    return x