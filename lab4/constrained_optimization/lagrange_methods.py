"""
Модуль с методом модифицированных функций Лагранжа
"""

import numpy as np
from typing import Callable, List, Dict, Optional


def modified_lagrange_method(f: Callable, grad_f: Callable, x0: np.ndarray,
                              constraints: List[Callable],
                              lambda0: Optional[np.ndarray] = None,
                              mu0: Optional[np.ndarray] = None,
                              r0: float = 1.0, C: float = 2.0,
                              eps: float = 1e-6, max_outer_iter: int = 50,
                              max_inner_iter: int = 100) -> Dict:
    """
    Метод модифицированных функций Лагранжа (метод множителей Лагранжа).
    
    Args:
        f: Целевая функция
        grad_f: Градиент целевой функции
        x0: Начальная точка
        constraints: Список функций ограничений g_j(x) ≤ 0
        lambda0: Начальные множители для ограничений-равенств (не используется)
        mu0: Начальные множители для ограничений-неравенств
        r0: Начальный параметр штрафа
        C: Коэффициент увеличения штрафа
        eps: Точность остановки
        max_outer_iter: Макс. число внешних итераций
        max_inner_iter: Макс. число итераций внутренней оптимизации
    
    Returns:
        Словарь с результатами оптимизации
    """
    m = len(constraints)  # Число ограничений-неравенств
    
    # Инициализация множителей Лагранжа
    if mu0 is None:
        mu = np.zeros(m)
    else:
        mu = mu0.copy()
    
    history = {
        'x': [x0.copy()],
        'f': [f(x0)],
        'r': [r0],
        'mu': [mu.copy()],
        'feasible': [all(g(x0) <= 1e-6 for g in constraints)]
    }
    
    x = x0.copy()
    r = r0
    
    for k in range(max_outer_iter):
        # Модифицированная функция Лагранжа
        def L(x):
            val = f(x)
            for j, g in enumerate(constraints):
                g_val = g(x)
                # Для ограничений-неравенств
                combined = mu[j] + r * g_val
                val += (1 / (2 * r)) * (max(0, combined)**2 - mu[j]**2)
            return val
        
        def grad_L(x):
            grad = grad_f(x)
            for j, g in enumerate(constraints):
                g_val = g(x)
                combined = mu[j] + r * g_val
                if combined > 0:
                    # Градиент ограничения (аппроксимация)
                    grad_g = constraint_gradient_approx(x, constraints[j])
                    grad += combined * grad_g
            return grad
        
        # Минимизация модифицированной функции Лагранжа
        try:
            from optimization import lbfgs
            result = lbfgs(L, grad_L, x, max_iter=max_inner_iter)
        except ImportError:
            from scipy.optimize import minimize
            result = minimize(L, x, method='L-BFGS-B', jac=grad_L)
            result = {'x_star': result.x, 'f_star': result.fun, 'iterations': result.nit}
        
        x_new = result['x_star']
        
        # Обновление множителей Лагранжа
        mu_new = np.zeros(m)
        for j, g in enumerate(constraints):
            g_val = g(x_new)
            mu_new[j] = max(0, mu[j] + r * g_val)
        
        # Проверка окончания
        max_violation = max(abs(g(x_new)) for g in constraints)
        feasible, _ = check_constraints_wrapper(x_new, constraints)
        
        history['x'].append(x_new.copy())
        history['f'].append(f(x_new))
        history['r'].append(r)
        history['mu'].append(mu_new.copy())
        history['feasible'].append(feasible)
        
        # Критерий остановки
        mu_change = np.linalg.norm(mu_new - mu)
        if max_violation <= eps and mu_change <= eps:
            return {
                'x_star': x_new,
                'f_star': f(x_new),
                'iterations': k + 1,
                'lagrange_multipliers': mu_new,
                'feasible': feasible,
                'history': history,
                'method': 'modified_lagrange'
            }
        
        # Обновление параметров
        mu = mu_new
        if max_violation > eps / 10:
            r = r * C  # Увеличиваем штраф если нарушения велики
        x = x_new
    
    return {
        'x_star': x,
        'f_star': f(x),
        'iterations': max_outer_iter,
        'lagrange_multipliers': mu,
        'feasible': check_constraints_wrapper(x, constraints)[0],
        'history': history,
        'method': 'modified_lagrange',
        'converged': False
    }


def constraint_gradient_approx(x: np.ndarray, g: Callable, 
                                eps: float = 1e-8) -> np.ndarray:
    """Аппроксимация градиента ограничения конечными разностями."""
    n = len(x)
    grad = np.zeros(n)
    
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        
        grad[i] = (g(x_plus) - g(x_minus)) / (2 * eps)
    
    return grad


def check_constraints_wrapper(x: np.ndarray, constraints: List[Callable],
                               tol: float = 1e-6) -> tuple:
    """Проверка выполнения ограничений."""
    violations = []
    for i, g in enumerate(constraints):
        g_val = g(x)
        if g_val > tol:
            violations.append((i, g_val))
    
    return len(violations) == 0, {
        'violations': violations,
        'max_violation': max([v[1] for v in violations]) if violations else 0.0,
        'feasible': len(violations) == 0
    }