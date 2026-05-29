"""
Модуль с тестовыми функциями и их градиентами
"""

import numpy as np
from typing import Callable


# Параметры по умолчанию для функции Розенброка
DEFAULT_A = 150
DEFAULT_B = 2
DEFAULT_F0 = 100


def rosenbrock(x: np.ndarray, a: float = DEFAULT_A, b: float = DEFAULT_B, 
               f0: float = DEFAULT_F0) -> float:
    """
    Функция Розенброка для произвольной размерности n.
    
    f(x) = Σ[i=1 to n-1] [a(x_i² - x_{i+1})² + b(x_i - 1)²] + f0
    
    Args:
        x: Вектор аргументов
        a: Параметр a (по умолчанию 150)
        b: Параметр b (по умолчанию 2)
        f0: Свободный член (по умолчанию 100)
    
    Returns:
        Значение функции в точке x
    """
    n = len(x)
    result = f0
    for i in range(n - 1):
        result += a * (x[i]**2 - x[i+1])**2 + b * (x[i] - 1)**2
    return result


def rosenbrock_gradient(x: np.ndarray, a: float = DEFAULT_A, 
                        b: float = DEFAULT_B) -> np.ndarray:
    """
    Градиент функции Розенброка.
    
    Args:
        x: Вектор аргументов
        a: Параметр a
        b: Параметр b
    
    Returns:
        Вектор градиента в точке x
    """
    n = len(x)
    grad = np.zeros(n)
    
    for i in range(n - 1):
        grad[i] += 2 * a * (x[i]**2 - x[i+1]) * 2 * x[i] + 2 * b * (x[i] - 1)
        grad[i+1] += -2 * a * (x[i]**2 - x[i+1])
    
    return grad


def rosenbrock_hessian(x: np.ndarray, a: float = DEFAULT_A, 
                       b: float = DEFAULT_B) -> np.ndarray:
    """
    Матрица Гессе функции Розенброка.
    
    Args:
        x: Вектор аргументов
        a: Параметр a
        b: Параметр b
    
    Returns:
        Матрица Гессе в точке x
    """
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
    """
    Создать функцию Розенброка с заданными параметрами.
    
    Args:
        a: Параметр a
        b: Параметр b
        f0: Свободный член
    
    Returns:
        Кортеж (функция, градиент, гессиан)
    """
    def f(x):
        return rosenbrock(x, a, b, f0)
    
    def grad(x):
        return rosenbrock_gradient(x, a, b)
    
    def hess(x):
        return rosenbrock_hessian(x, a, b)
    
    return f, grad, hess
