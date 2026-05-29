"""
Модуль с квазиньютоновскими методами оптимизации
"""

import numpy as np
from typing import Callable, Dict, List
from .line_search import line_search


# Параметры по умолчанию
DEFAULT_EPS1 = 1e-6
DEFAULT_EPS2 = 1e-8
DEFAULT_MAX_ITER = 1000


def dfp(f: Callable, grad_f: Callable, x0: np.ndarray,
        eps1: float = DEFAULT_EPS1, eps2: float = DEFAULT_EPS2,
        max_iter: int = DEFAULT_MAX_ITER) -> Dict:
    """
    Квазиньютоновский метод Девидона-Флетчера-Пауэлла (DFP).
    
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
    
    G = np.eye(n)  # Аппроксимация обратной матрицы Гессе
    grad = grad_f(x)
    
    for k in range(max_iter):
        grad_norm = np.linalg.norm(grad)
        
        # Критерий остановки
        if grad_norm < eps1:
            break
        
        # Направление поиска
        d = -G @ grad
        
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
        
        # Обновление матрицы G (DFP формула)
        delta_x = x_new - x
        delta_g = grad_new - grad
        
        dx_dg = np.dot(delta_x, delta_g)
        if dx_dg > 1e-10:
            term1 = np.outer(delta_x, delta_x) / dx_dg
            Gg = G @ delta_g
            term2 = np.outer(Gg, Gg) / np.dot(delta_g, Gg)
            G = G + term1 - term2
        
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


def bfgs(f: Callable, grad_f: Callable, x0: np.ndarray,
         eps1: float = DEFAULT_EPS1, eps2: float = DEFAULT_EPS2,
         max_iter: int = DEFAULT_MAX_ITER) -> Dict:
    """
    Квазиньютоновский метод BFGS (Бройдена-Флетчера-Гольдфарба-Шенно).
    
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
    
    H = np.eye(n)  # Аппроксимация обратной матрицы Гессе
    grad = grad_f(x)
    
    for k in range(max_iter):
        grad_norm = np.linalg.norm(grad)
        
        # Критерий остановки
        if grad_norm < eps1:
            break
        
        # Направление поиска
        d = -H @ grad
        
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
        
        # Обновление матрицы H (BFGS формула)
        s = x_new - x
        y = grad_new - grad
        
        sy = np.dot(s, y)
        if sy > 1e-10:
            rho = 1.0 / sy
            I = np.eye(n)
            V = I - rho * np.outer(s, y)
            H = V @ H @ V.T + rho * np.outer(s, s)
        
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


def lbfgs(f: Callable, grad_f: Callable, x0: np.ndarray,
          eps1: float = DEFAULT_EPS1, eps2: float = DEFAULT_EPS2,
          max_iter: int = DEFAULT_MAX_ITER, m: int = 10) -> Dict:
    """
    Квазиньютоновский метод L-BFGS (Limited-memory BFGS).
    
    Args:
        f: Функция для минимизации
        grad_f: Градиент функции
        x0: Начальная точка
        eps1: Точность по градиенту
        eps2: Точность по изменению функции
        max_iter: Максимальное число итераций
        m: Размер памяти (число сохраняемых пар векторов)
    
    Returns:
        Словарь с результатами оптимизации
    """
    history = {
        'x': [x0.copy()], 
        'f': [f(x0)], 
        'grad_norm': [np.linalg.norm(grad_f(x0))]
    }
    x = x0.copy()
    
    grad = grad_f(x)
    
    # Хранилище для пар векторов (s, y) и коэффициентов rho
    s_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    rho_list: List[float] = []
    
    for k in range(max_iter):
        grad_norm = np.linalg.norm(grad)
        
        # Критерий остановки
        if grad_norm < eps1:
            break
        
        # Двухпетлевая рекурсия для вычисления направления
        q = grad.copy()
        alpha_list: List[float] = []
        
        # Первый цикл (обратный)
        for i in range(len(s_list) - 1, -1, -1):
            s = s_list[i]
            y = y_list[i]
            rho = rho_list[i]
            alpha_i = rho * np.dot(s, q)
            alpha_list.insert(0, alpha_i)
            q = q - alpha_i * y
        
        # Инициализация H_0
        if len(s_list) > 0:
            s_last = s_list[-1]
            y_last = y_list[-1]
            gamma = np.dot(s_last, y_last) / np.dot(y_last, y_last)
        else:
            gamma = 1.0
        
        r = gamma * q
        
        # Второй цикл (прямой)
        for i in range(len(s_list)):
            s = s_list[i]
            y = y_list[i]
            rho = rho_list[i]
            alpha_i = alpha_list[i]
            beta = rho * np.dot(y, r)
            r = r + s * (alpha_i - beta)
        
        d = -r
        
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
        
        # Сохранение новой пары векторов
        s = x_new - x
        y = grad_new - grad
        
        sy = np.dot(s, y)
        if sy > 1e-10:
            if len(s_list) >= m:
                s_list.pop(0)
                y_list.pop(0)
                rho_list.pop(0)
            
            s_list.append(s)
            y_list.append(y)
            rho_list.append(1.0 / sy)
        
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
