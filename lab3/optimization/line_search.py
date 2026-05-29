"""
Модуль с методами одномерного поиска для определения шага
"""

import numpy as np
from typing import Callable


def golden_section_search(f: Callable, a: float, b: float, 
                          tol: float = 1e-8) -> float:
    """
    Метод золотого сечения для одномерной минимизации.
    
    Args:
        f: Функция для минимизации
        a: Левая граница интервала
        b: Правая граница интервала
        tol: Точность
    
    Returns:
        Приближённое значение точки минимума
    """
    phi = (1 + np.sqrt(5)) / 2
    
    c = b - (b - a) / phi
    d = a + (b - a) / phi
    
    while abs(b - a) > tol:
        if f(c) < f(d):
            b = d
            d = c
            c = b - (b - a) / phi
        else:
            a = c
            c = d
            d = a + (b - a) / phi
    
    return (a + b) / 2


def line_search(f: Callable, x: np.ndarray, d: np.ndarray,
                alpha_max: float = 10.0, tol: float = 1e-8) -> float:
    """
    Одномерный поиск шага alpha вдоль направления d.
    
    Args:
        f: Функция для минимизации
        x: Текущая точка
        d: Направление поиска
        alpha_max: Максимальное значение шага
        tol: Точность поиска
    
    Returns:
        Оптимальная величина шага alpha
    """
    def phi(alpha):
        return f(x + alpha * d)
    
    return golden_section_search(phi, 0.0, alpha_max, tol)
