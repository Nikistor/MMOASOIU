"""
Модуль с классом Optimizer для удобного запуска оптимизации
"""

import numpy as np
import time
from typing import Callable, Dict, List, Tuple, Optional
from . import (
    gradient_descent,
    fletcher_reeves,
    polak_ribiere,
    dfp,
    bfgs,
    lbfgs
)


class Optimizer:
    """
    Класс для выполнения оптимизации различными методами.
    """
    
    def __init__(self, f: Callable, grad_f: Callable, x0: np.ndarray,
                 eps1: float = 1e-6, eps2: float = 1e-8, max_iter: int = 1000):
        """
        Инициализация оптимизатора.
        
        Args:
            f: Функция для минимизации
            grad_f: Градиент функции
            x0: Начальная точка
            eps1: Точность по градиенту
            eps2: Точность по изменению функции
            max_iter: Максимальное число итераций
        """
        self.f = f
        self.grad_f = grad_f
        self.x0 = x0
        self.eps1 = eps1
        self.eps2 = eps2
        self.max_iter = max_iter
        self.results: Dict[str, Dict] = {}
    
    def run_all_methods(self, verbose: bool = True) -> Dict[str, Dict]:
        """
        Запустить все методы оптимизации.
        
        Args:
            verbose: Выводить ли информацию о процессе
        
        Returns:
            Словарь с результатами всех методов
        """
        methods = [
            ('Градиентный спуск', gradient_descent),
            ('Флетчера-Ривза', fletcher_reeves),
            ('Полака-Рибьера', polak_ribiere),
            ('DFP', dfp),
            ('BFGS', bfgs),
            ('L-BFGS (m=10)', lambda f, g, x0, e1, e2, mi: lbfgs(f, g, x0, e1, e2, mi, m=10))
        ]
        
        if verbose:
            print("\n" + "=" * 60)
            print("ЗАПУСК ВСЕХ МЕТОДОВ ОПТИМИЗАЦИИ")
            print("=" * 60)
        
        for name, method in methods:
            if verbose:
                print(f"\nЗапуск метода: {name}...")
            
            start_time = time.time()
            result = method(self.f, self.grad_f, self.x0, 
                          self.eps1, self.eps2, self.max_iter)
            elapsed_time = time.time() - start_time
            
            self.results[name] = {
                **result,
                'time': elapsed_time
            }
            
            if verbose:
                print(f"  Найдено: x* = {result['x_star']}")
                print(f"  f(x*) = {result['f_star']:.8f}")
                print(f"  Итераций: {result['iterations']}")
                print(f"  Время: {elapsed_time:.4f} сек")
        
        return self.results
    
    def run_method(self, method_name: str, **kwargs) -> Dict:
        """
        Запустить конкретный метод оптимизации.
        
        Args:
            method_name: Название метода
            **kwargs: Дополнительные параметры для метода
        
        Returns:
            Словарь с результатами
        """
        methods = {
            'gradient_descent': gradient_descent,
            'fletcher_reeves': fletcher_reeves,
            'polak_ribiere': polak_ribiere,
            'dfp': dfp,
            'bfgs': bfgs,
            'lbfgs': lbfgs
        }
        
        if method_name not in methods:
            raise ValueError(f"Неизвестный метод: {method_name}. "
                           f"Доступные: {list(methods.keys())}")
        
        method = methods[method_name]
        start_time = time.time()
        result = method(self.f, self.grad_f, self.x0, 
                       self.eps1, self.eps2, self.max_iter, **kwargs)
        elapsed_time = time.time() - start_time
        
        self.results[method_name] = {
            **result,
            'time': elapsed_time
        }
        
        return self.results[method_name]
    
    def get_best_method(self, criterion: str = 'time') -> Tuple[str, Dict]:
        """
        Получить лучший метод по заданному критерию.
        
        Args:
            criterion: Критерий выбора ('time', 'iterations', 'accuracy')
        
        Returns:
            Кортеж (название метода, результаты)
        """
        if not self.results:
            raise ValueError("Сначала запустите методы оптимизации")
        
        if criterion == 'time':
            best_name = min(self.results.keys(), 
                          key=lambda k: self.results[k]['time'])
        elif criterion == 'iterations':
            best_name = min(self.results.keys(), 
                          key=lambda k: self.results[k]['iterations'])
        elif criterion == 'accuracy':
            best_name = min(self.results.keys(), 
                          key=lambda k: np.linalg.norm(
                              self.grad_f(self.results[k]['x_star'])))
        else:
            raise ValueError(f"Неизвестный критерий: {criterion}")
        
        return best_name, self.results[best_name]
    
    def print_comparison_table(self):
        """
        Вывести сравнительную таблицу результатов.
        """
        if not self.results:
            print("Нет результатов для сравнения")
            return
        
        print("\n" + "=" * 80)
        print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА МЕТОДОВ")
        print("=" * 80)
        print(f"{'Метод':<25} {'Итерации':<10} {'f(x*)':<15} "
              f"{'Время (сек)':<12} {'||∇f||':<12}")
        print("-" * 80)
        
        for name, result in self.results.items():
            grad_norm = np.linalg.norm(self.grad_f(result['x_star']))
            print(f"{name:<25} {result['iterations']:<10} "
                  f"{result['f_star']:<15.8f} {result['time']:<12.4f} "
                  f"{grad_norm:<12.2e}")
        
        print("=" * 80)
