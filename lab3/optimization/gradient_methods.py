"""
Модуль с методами градиентного спуска
"""

import numpy as np
from typing import Callable, Dict
from .line_search import line_search


# Параметры по умолчанию
DEFAULT_EPS1 = 1e-6
DEFAULT_EPS2 = 1e-8
DEFAULT_MAX_ITER = 1000


def gradient_descent(f: Callable, grad_f: Callable, x0: np.ndarray,
                     eps1: float = DEFAULT_EPS1, eps2: float = DEFAULT_EPS2,
                     max_iter: int = DEFAULT_MAX_ITER) -> Dict:
    """
    Метод наискорейшего градиентного спуска.
    
    Args:
        f: Функция для минимизации
        grad_f: Градиент функции
        x0: Начальная точка
        eps1: Точность по градиенту
        eps2: Точность по изменению функции
        max_iter: Максимальное число итераций
    
    Returns:
        Словарь с результатами:
            - x_star: найденная точка минимума
            - f_star: значение функции в точке минимума
            - iterations: число итераций
            - history: история сходимости
    """
    history = {
        'x': [x0.copy()], 
        'f': [f(x0)], 
        'grad_norm': [np.linalg.norm(grad_f(x0))]
    }
    x = x0.copy()
    
    for k in range(max_iter):
        grad = grad_f(x)
        grad_norm = np.linalg.norm(grad)
        
        # Критерий остановки по градиенту
        if grad_norm < eps1:
            break
        
        # Направление спуска
        d = -grad
        
        # Поиск шага
        alpha = line_search(f, x, d)
        
        # Новый пункт
        x_new = x + alpha * d
        
        # Проверка критериев остановки
        if (np.linalg.norm(x_new - x) < eps1 and 
            abs(f(x_new) - f(x)) < eps2):
            x = x_new
            history['x'].append(x.copy())
            history['f'].append(f(x))
            history['grad_norm'].append(np.linalg.norm(grad_f(x)))
            break
        
        x = x_new
        history['x'].append(x.copy())
        history['f'].append(f(x))
        history['grad_norm'].append(np.linalg.norm(grad_f(x)))
    
    return {
        'x_star': x,
        'f_star': f(x),
        'iterations': len(history['f']) - 1,
        'history': history
    }
