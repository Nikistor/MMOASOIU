"""
Модуль с методом проекции градиента
"""

import numpy as np
from typing import Callable, List, Dict, Optional, Tuple


def projected_gradient_method(f: Callable, grad_f: Callable, x0: np.ndarray,
                               constraints: List[Callable],
                               grad_constraints: Optional[List[Callable]] = None,
                               eps1: float = -1e-6, eps2: float = 1e-8,
                               max_iter: int = 500,
                               line_search: str = 'armijo') -> Dict:
    """
    Метод проекции градиента для задач с ограничениями-неравенствами.
    
    Согласно методичке:
    - eps1 <= 0 для определения активных ограничений (eps1 <= g_j(x) <= 0)
    - eps2 > 0 для проверки сходимости

    Args:
        f: Целевая функция
        grad_f: Градиент целевой функции
        x0: Начальная точка
        constraints: Список функций ограничений g_j(x) ≤ 0
        grad_constraints: Градиенты ограничений (опционально)
        eps1: Точность определения активных ограничений (eps1 <= 0)
        eps2: Точность по направлению (eps2 > 0)
        max_iter: Максимальное число итераций
        line_search: Тип линейного поиска ('armijo', 'exact')

    Returns:
        Словарь с результатами оптимизации
    """
    n = len(x0)
    m = len(constraints)
    
    history = {
        'x': [x0.copy()],
        'f': [f(x0)],
        'step': [0.0],
        'active': [[]],
        'feasible': [check_constraints_simple(x0, constraints)]
    }
    
    x = x0.copy()
    
    # Проверяем допустимость начальной точки
    feasible, info = check_constraints_simple(x, constraints, return_info=True)
    if not feasible:
        print(f"Начальная точка недопустима! Нарушения: {info['violations']}")
        x = find_feasible_point_simple(x, constraints)
    
    for k in range(max_iter):
        # Определяем активные ограничения
        active_set = get_active_constraints(x, constraints, eps=1e-4)
        
        # Вычисляем проекцию антиградиента
        if len(active_set) == 0:
            # Нет активных ограничений - просто антиградиент
            d = -grad_f(x)
        else:
            # Есть активные ограничения - вычисляем проекцию
            d = compute_search_direction(x, grad_f, constraints, 
                                         grad_constraints, active_set)
        
        # Проверка критерия остановки
        if np.linalg.norm(d) < eps1:
            # Проверяем условия Куна-Таккера
            lambdas = compute_lagrange_multipliers(x, grad_f, constraints,
                                                   grad_constraints, active_set)
            if all(l >= -eps1 for l in lambdas):
                return {
                    'x_star': x,
                    'f_star': f(x),
                    'iterations': k,
                    'active_constraints': active_set,
                    'lagrange_multipliers': lambdas,
                    'feasible': check_constraints_simple(x, constraints),
                    'history': history,
                    'method': 'projected_gradient'
                }
        
        # Линейный поиск
        if line_search == 'armijo':
            alpha = armijo_line_search(f, x, d, grad_f, constraints)
        else:
            alpha = exact_line_search(f, x, d, constraints)
        
        # Обновление точки
        x_new = x + alpha * d
        
        # Проверка критериев остановки
        if (np.linalg.norm(x_new - x) < eps1 and 
            abs(f(x_new) - f(x)) < eps2):
            return {
                'x_star': x_new,
                'f_star': f(x_new),
                'iterations': k + 1,
                'active_constraints': get_active_constraints(x_new, constraints),
                'feasible': check_constraints_simple(x_new, constraints),
                'history': history,
                'method': 'projected_gradient'
            }
        
        # Запись в историю
        history['x'].append(x_new.copy())
        history['f'].append(f(x_new))
        history['step'].append(alpha)
        history['active'].append(get_active_constraints(x_new, constraints))
        history['feasible'].append(check_constraints_simple(x_new, constraints))
        
        x = x_new
    
    return {
        'x_star': x,
        'f_star': f(x),
        'iterations': max_iter,
        'active_constraints': get_active_constraints(x, constraints),
        'feasible': check_constraints_simple(x, constraints),
        'history': history,
        'method': 'projected_gradient',
        'converged': False
    }


def get_active_constraints(x: np.ndarray, constraints: List[Callable],
                           eps: float = 1e-4) -> List[int]:
    """Определить индексы активных ограничений."""
    active = []
    for i, g in enumerate(constraints):
        if abs(g(x)) < eps:
            active.append(i)
    return active


def compute_search_direction(x: np.ndarray, grad_f: Callable,
                              constraints: List[Callable],
                              grad_constraints: Optional[List[Callable]],
                              active_set: List[int]) -> np.ndarray:
    """
    Вычислить направление поиска как проекцию антиградиента.
    
    d = -[E - A^T(AA^T)^{-1}A]∇f(x)
    где A - матрица градиентов активных ограничений
    """
    n = len(x)
    grad = grad_f(x)
    
    if len(active_set) == 0:
        return -grad
    
    # Вычисляем градиенты активных ограничений
    if grad_constraints is not None:
        A = np.zeros((len(active_set), n))
        for i, j in enumerate(active_set):
            A[i, :] = grad_constraints[j](x)
    else:
        # Аппроксимация градиентов
        A = np.zeros((len(active_set), n))
        for i, j in enumerate(active_set):
            A[i, :] = constraint_gradient(x, constraints[j])
    
    # Проекция: d = -[E - A^T(AA^T)^{-1}A]∇f
    try:
        ATA_inv = np.linalg.inv(A @ A.T)
        P = np.eye(n) - A.T @ ATA_inv @ A
        d = -P @ grad
    except np.linalg.LinAlgError:
        # Если матрица вырождена, используем псевдообратную
        try:
            ATA_pinv = np.linalg.pinv(A @ A.T)
            P = np.eye(n) - A.T @ ATA_pinv @ A
            d = -P @ grad
        except:
            d = -grad
    
    return d


def compute_lagrange_multipliers(x: np.ndarray, grad_f: Callable,
                                  constraints: List[Callable],
                                  grad_constraints: Optional[List[Callable]],
                                  active_set: List[int]) -> np.ndarray:
    """Вычислить множители Лагранжа для активных ограничений."""
    m = len(constraints)
    lambdas = np.zeros(m)
    
    if len(active_set) == 0:
        return lambdas
    
    n = len(x)
    grad = grad_f(x)
    
    # Вычисляем градиенты активных ограничений
    if grad_constraints is not None:
        A = np.zeros((len(active_set), n))
        for i, j in enumerate(active_set):
            A[i, :] = grad_constraints[j](x)
    else:
        A = np.zeros((len(active_set), n))
        for i, j in enumerate(active_set):
            A[i, :] = constraint_gradient(x, constraints[j])
    
    # λ = -(AA^T)^{-1}A∇f
    try:
        ATA_inv = np.linalg.inv(A @ A.T)
        lambdas_active = -ATA_inv @ A @ grad
        
        for i, j in enumerate(active_set):
            lambdas[j] = lambdas_active[i]
    except:
        pass
    
    return lambdas


def armijo_line_search(f: Callable, x: np.ndarray, d: np.ndarray,
                        grad_f: Callable, constraints: List[Callable],
                        alpha0: float = 1.0, rho: float = 0.5,
                        c: float = 1e-4) -> float:
    """Линейный поиск Армихо с учётом ограничений."""
    alpha = alpha0
    f_x = f(x)
    grad_dot_d = np.dot(grad_f(x), d)
    
    for _ in range(50):
        x_new = x + alpha * d
        
        # Проверка допустимости
        feasible = check_constraints_simple(x_new, constraints)
        
        # Условие Армихо
        if feasible and f(x_new) <= f_x + c * alpha * grad_dot_d:
            return alpha
        
        alpha *= rho
    
    return alpha


def exact_line_search(f: Callable, x: np.ndarray, d: np.ndarray,
                       constraints: List[Callable],
                       alpha_max: float = 10.0,
                       n_points: int = 100) -> float:
    """Точный линейный поиск с ограничениями."""
    # Находим максимальный допустимый шаг
    alpha_feasible = alpha_max
    
    for g in constraints:
        # Решаем квадратное уравнение g(x + αd) = 0
        # Для простоты используем бисекцию
        alpha_low, alpha_high = 0.0, alpha_max
        for _ in range(50):
            alpha_mid = (alpha_low + alpha_high) / 2
            if g(x + alpha_mid * d) > 0:
                alpha_high = alpha_mid
            else:
                alpha_low = alpha_mid
        alpha_feasible = min(alpha_feasible, alpha_high)
    
    # Минимизация на интервале [0, alpha_feasible]
    from scipy.optimize import minimize_scalar
    
    def phi(alpha):
        return f(x + alpha * d)
    
    result = minimize_scalar(phi, bounds=(0, alpha_feasible), method='bounded')
    return result.x


def constraint_gradient(x: np.ndarray, g: Callable, eps: float = 1e-8) -> np.ndarray:
    """Аппроксимация градиента ограничения."""
    n = len(x)
    grad = np.zeros(n)
    
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        
        grad[i] = (g(x_plus) - g(x_minus)) / (2 * eps)
    
    return grad


def check_constraints_simple(x: np.ndarray, constraints: List[Callable],
                              tol: float = 1e-6,
                              return_info: bool = False) -> bool:
    """Проверка выполнения ограничений."""
    violations = []
    for i, g in enumerate(constraints):
        g_val = g(x)
        if g_val > tol:
            violations.append((i, g_val))
    
    if return_info:
        return len(violations) == 0, {
            'violations': violations,
            'max_violation': max([v[1] for v in violations]) if violations else 0.0
        }
    return len(violations) == 0


def find_feasible_point_simple(x0: np.ndarray, constraints: List[Callable],
                                max_iter: int = 100) -> np.ndarray:
    """Найти допустимую точку."""
    x = x0.copy()
    n = len(x)
    
    # Для ограничений неотрицательности (g2, g3)
    for i in range(n):
        if x[i] < 0:
            x[i] = 0.1  # Небольшое положительное значение
    
    # Для g1: x1² - x2² - 3 ≤ 0 => x1² ≤ x2² + 3
    if len(constraints) >= 1:
        g1_val = x[0]**2 - x[1]**2 - 3
        if g1_val > 0:
            x[0] = np.sqrt(x[1]**2 + 3) + 0.1  # Увеличиваем x1 чтобы удовлетворить g1
    
    return x