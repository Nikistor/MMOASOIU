"""
Класс ConstrainedOptimizer для удобного запуска оптимизации с ограничениями
"""

import numpy as np
import time
from typing import Callable, List, Dict, Optional, Tuple

from .functions import (
    rosenbrock,
    rosenbrock_gradient,
    create_rosenbrock_function,
    get_constraints_variant3,  # Обновлено для третьего варианта
    check_constraints
)

from .penalty_methods import (
    penalty_method,
    barrier_method,
    combined_penalty_barrier_method
)

from .lagrange_methods import (
    modified_lagrange_method
)

from .projection_methods import (
    projected_gradient_method
)


class ConstrainedOptimizer:
    """
    Класс для выполнения оптимизации с ограничениями.
    """
    
    def __init__(self, f: Callable, grad_f: Callable, x0: np.ndarray,
                 constraints: List[Callable],
                 eps: float = 1e-6, max_iter: int = 100):
        """
        Инициализация оптимизатора.
        
        Args:
            f: Целевая функция
            grad_f: Градиент целевой функции
            x0: Начальная точка
            constraints: Список функций ограничений g_j(x) ≤ 0
            eps: Точность остановки
            max_iter: Максимальное число итераций
        """
        self.f = f
        self.grad_f = grad_f
        self.x0 = x0
        self.constraints = constraints
        self.eps = eps
        self.max_iter = max_iter
        self.results: Dict[str, Dict] = {}
    
    def run_all_methods(self, verbose: bool = True) -> Dict[str, Dict]:
        """
        Запустить все методы оптимизации с ограничениями.
        """
        methods = [
            ('Метод штрафных функций', self._run_penalty),
            ('Метод барьерных функций', self._run_barrier),
            ('Комбинированный метод', self._run_combined),
            ('Метод множителей Лагранжа', self._run_lagrange),
            ('Метод проекции градиента', self._run_projection)
        ]
        
        if verbose:
            print("\n" + "=" * 70)
            print("ЗАПУСК ВСЕХ МЕТОДОВ УСЛОВНОЙ ОПТИМИЗАЦИИ")
            print("=" * 70)
        
        for name, method in methods:
            if verbose:
                print(f"\nЗапуск метода: {name}...")
            
            start_time = time.time()
            result = method()
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
                print(f"  Допустимо: {result['feasible']}")
        
        return self.results
    
    def _run_penalty(self) -> Dict:
        """Запустить метод штрафных функций."""
        return penalty_method(
            self.f, self.grad_f, self.x0, self.constraints,
            r0=1.0, C=10.0, eps=self.eps, max_outer_iter=self.max_iter
        )
    
    def _run_barrier(self) -> Dict:
        """Запустить метод барьерных функций."""
        return barrier_method(
            self.f, self.grad_f, self.x0, self.constraints,
            r0=1.0, C=10.0, eps=self.eps, max_outer_iter=self.max_iter
        )
    
    def _run_combined(self) -> Dict:
        """Запустить комбинированный метод."""
        return combined_penalty_barrier_method(
            self.f, self.grad_f, self.x0, self.constraints,
            r0=1.0, C=10.0, eps=self.eps, max_outer_iter=self.max_iter,
            alpha=0.5
        )
    
    def _run_lagrange(self) -> Dict:
        """Запустить метод модифицированных функций Лагранжа."""
        return modified_lagrange_method(
            self.f, self.grad_f, self.x0, self.constraints,
            r0=1.0, C=2.0, eps=self.eps, max_outer_iter=self.max_iter
        )
    
    def _run_projection(self) -> Dict:
        """Запустить метод проекции градиента."""
        return projected_gradient_method(
            self.f, self.grad_f, self.x0, self.constraints,
            eps1=self.eps, max_iter=self.max_iter * 10
        )
    
    def get_best_method(self, criterion: str = 'f_star') -> Tuple[str, Dict]:
        """
        Получить лучший метод по заданному критерию.
        
        Args:
            criterion: 'f_star', 'iterations', 'time', 'accuracy'
        """
        if not self.results:
            raise ValueError("Сначала запустите методы оптимизации")
        
        if criterion == 'f_star':
            best_name = min(self.results.keys(), 
                          key=lambda k: self.results[k]['f_star'])
        elif criterion == 'iterations':
            best_name = min(self.results.keys(), 
                          key=lambda k: self.results[k]['iterations'])
        elif criterion == 'time':
            best_name = min(self.results.keys(), 
                          key=lambda k: self.results[k]['time'])
        elif criterion == 'accuracy':
            def accuracy(name):
                result = self.results[name]
                penalty = sum(max(0, g(result['x_star']))**2 
                             for g in self.constraints)
                return penalty
            best_name = min(self.results.keys(), key=accuracy)
        else:
            raise ValueError(f"Неизвестный критерий: {criterion}")
        
        return best_name, self.results[best_name]
    
    def print_comparison_table(self):
        """Вывести сравнительную таблицу результатов."""
        if not self.results:
            print("Нет результатов для сравнения")
            return
        
        print("\n" + "=" * 90)
        print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА МЕТОДОВ УСЛОВНОЙ ОПТИМИЗАЦИИ")
        print("=" * 90)
        print(f"{'Метод':<35} {'Итер.':<8} {'f(x*)':<15} "
              f"{'Время (с)':<10} {'Допустимо':<10}")
        print("-" * 90)
        
        for name, result in self.results.items():
            feasible_str = "✓" if result['feasible'] else "✗"
            print(f"{name:<35} {result['iterations']:<8} "
                  f"{result['f_star']:<15.8f} {result['time']:<10.4f} "
                  f"{feasible_str:<10}")
        
        print("=" * 90)
        
        # Проверка допустимости решений
        print("\nПРОВЕРКА ДОПУСТИМОСТИ РЕШЕНИЙ:")
        print("-" * 90)
        
        for name, result in self.results.items():
            x = result['x_star']
            feasible, info = check_constraints(x, self.constraints)
            
            print(f"\n{name}:")
            print(f"  x* = {x}")
            print(f"  f(x*) = {result['f_star']:.8f}")
            print(f"  Допустимо: {feasible}")
            
            if info['violations']:
                print(f"  Нарушения:")
                for i, val in info['violations']:
                    print(f"    g{i+1}(x) = {val:.6f} > 0")
            else:
                print(f"  Все ограничения выполнены")
                print(f"  Значения ограничений:")
                for i, g in enumerate(self.constraints):
                    print(f"    g{i+1}(x) = {g(x):.6f} ≤ 0")


# ============================================================================
# Вспомогательная функция для создания оптимизатора для варианта 3
# ============================================================================

def create_optimizer_variant3(x0: Optional[np.ndarray] = None,
                               eps: float = 1e-6, 
                               max_iter: int = 100) -> ConstrainedOptimizer:
    """
    Создать оптимизатор для варианта 3.
    
    Args:
        x0: Начальная точка (по умолчанию [1.5, 1.5])
        eps: Точность
        max_iter: Макс. итераций
    
    Returns:
        ConstrainedOptimizer для варианта 3
    """
    if x0 is None:
        x0 = np.array([1.5, 1.5])  # Новая начальная точка для третьего варианта
    
    f, grad_f, _ = create_rosenbrock_function(a=80, b=3, f0=110)  # Обновленные параметры
    constraints = get_constraints_variant3()  # Обновлено для третьего варианта
    
    return ConstrainedOptimizer(f, grad_f, x0, constraints, eps, max_iter)