"""
Модуль с методами сопряжённых градиентов
"""

import numpy as np
from typing import Callable, Dict
from .line_search import line_search


# Параметры по умолчанию
DEFAULT_EPS1 = 1e-6
DEFAULT_EPS2 = 1e-8
DEFAULT_MAX_ITER = 1000


def fletcher_reeves(f: Callable, grad_f: Callable, x0: np.ndarray,
                    eps1: float = DEFAULT_EPS1, eps2: float = DEFAULT_EPS2,
                    max_iter: int = DEFAULT_MAX_ITER) -> Dict:
    """
    Метод сопряжённых градиентов Флетчера-Ривза.
    
    Args:
        f: Функция для минимизации
        grad_f: Градиент функции
        x0: Начальная точка
        eps1: Точность по градиенту
        eps2: Точность по изменению функции
        max_iter: Максимальное число итераций
    
    Returns:
        Словарь с результатами оптимизации
    """
    history = {
        'x': [x0.copy()], 
        'f': [f(x0)], 
        'grad_norm': [np.linalg.norm(grad_f(x0))]
    }
    x = x0.copy()
    n = len(x0)
    
    grad = grad_f(x)
    d = -grad
    
    for k in range(max_iter):
        grad_norm = np.linalg.norm(grad)
        
        # Критерий остановки
        if grad_norm < eps1:
            break
        
        # Перезапуск каждые n шагов
        if k > 0 and k % n == 0:
            d = -grad
        
        # Поиск шага с ограничением
        alpha = line_search(f, x, d)
        alpha = min(alpha, 1.0)
        
        # Новый пункт
        x_new = x + alpha * d
        
        # Проверка на NaN/Inf
        if np.any(np.isnan(x_new)) or np.any(np.isinf(x_new)):
            alpha = alpha * 0.5
            x_new = x + alpha * d
            if np.any(np.isnan(x_new)) or np.any(np.isinf(x_new)):
                d = -grad
                alpha = 0.01
                x_new = x + alpha * d
        
        grad_new = grad_f(x_new)
        
        # Проверка градиента
        if np.any(np.isnan(grad_new)) or np.any(np.isinf(grad_new)):
            grad_new = grad_f(x_new + 1e-8)
        
        # Проверка критериев остановки
        if (np.linalg.norm(x_new - x) < eps1 and 
            abs(f(x_new) - f(x)) < eps2):
            x = x_new
            history['x'].append(x.copy())
            history['f'].append(f(x))
            history['grad_norm'].append(np.linalg.norm(grad_f(x)))
            break
        
        # Вычисление w по формуле Флетчера-Ривза
        grad_norm_sq = np.linalg.norm(grad)**2
        if grad_norm_sq > 1e-20:
            w = np.linalg.norm(grad_new)**2 / grad_norm_sq
        else:
            w = 0
        
        # Ограничение w
        w = np.clip(w, 0, 1)
        
        # Новое направление
        d = -grad_new + w * d
        
        # Проверка направления спуска
        if np.dot(d, grad_new) > 0:
            d = -grad_new
        
        x = x_new
        grad = grad_new
        history['x'].append(x.copy())
        history['f'].append(f(x))
        history['grad_norm'].append(np.linalg.norm(grad))
    
    return {
        'x_star': x,
        'f_star': f(x),
        'iterations': len(history['f']) - 1,
        'history': history
    }


def polak_ribiere(f: Callable, grad_f: Callable, x0: np.ndarray,
                  eps1: float = DEFAULT_EPS1, eps2: float = DEFAULT_EPS2,
                  max_iter: int = DEFAULT_MAX_ITER) -> Dict:
    """
    Метод сопряжённых градиентов Полака-Рибьера.
    
    Args:
        f: Функция для минимизации
        grad_f: Градиент функции
        x0: Начальная точка
        eps1: Точность по градиенту
        eps2: Точность по изменению функции
        max_iter: Максимальное число итераций
    
    Returns:
        Словарь с результатами оптимизации
    """
    history = {
        'x': [x0.copy()], 
        'f': [f(x0)], 
        'grad_norm': [np.linalg.norm(grad_f(x0))]
    }
    x = x0.copy()
    n = len(x0)
    
    grad = grad_f(x)
    d = -grad
    grad_prev = grad.copy()
    
    for k in range(max_iter):
        grad_norm = np.linalg.norm(grad)
        
        # Критерий остановки
        if grad_norm < eps1:
            break
        
        # Поиск шага
        alpha = line_search(f, x, d)
        
        # Новый пункт
        x_new = x + alpha * d
        grad_new = grad_f(x_new)
        
        # Проверка критериев остановки
        if (np.linalg.norm(x_new - x) < eps1 and 
            abs(f(x_new) - f(x)) < eps2):
            x = x_new
            history['x'].append(x.copy())
            history['f'].append(f(x))
            history['grad_norm'].append(np.linalg.norm(grad_f(x)))
            break
        
        # Вычисление w по формуле Полака-Рибьера
        grad_diff = grad_new - grad_prev
        grad_prev_norm_sq = np.linalg.norm(grad_prev)**2
        
        if grad_prev_norm_sq > 1e-20:
            w = np.dot(grad_new, grad_diff) / grad_prev_norm_sq
        else:
            w = 0
        
        # Перезапуск каждые n шагов
        if (k + 1) % n == 0:
            w = 0
        
        # Ограничение w
        w = np.clip(w, 0, 1)
        
        # Новое направление
        d = -grad_new + w * d
        
        # Проверка направления спуска
        if np.dot(d, grad_new) > 0:
            d = -grad_new
        
        x = x_new
        grad_prev = grad.copy()
        grad = grad_new
        history['x'].append(x.copy())
        history['f'].append(f(x))
        history['grad_norm'].append(np.linalg.norm(grad))
    
    return {
        'x_star': x,
        'f_star': f(x),
        'iterations': len(history['f']) - 1,
        'history': history
    }
