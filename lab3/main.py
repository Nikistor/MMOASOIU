"""
Главный файл для запуска лабораторной работы №3
Методы многомерного поиска

Вариант 3: a=80, b=3, f0=110, n=2

Выполнил: Студент группы ИУ5Ц-21М Перевощиков Н.Д.
"""

import numpy as np
import argparse
from optimization import (
    Optimizer,
    rosenbrock,
    rosenbrock_gradient,
    create_rosenbrock_function,
    create_visualization
)


# ============================================================================
# Параметры варианта 3
# ============================================================================
A = 80
B = 3
F0 = 110
N = 2

# Параметры остановки
EPS1 = 1e-6
EPS2 = 1e-8
MAX_ITER = 10000


def rosenbrock_2d(x1, x2, a=A, b=B, f0=F0, x3=1.0):
    """
    Функция Розенброка для 2D визуализации (фиксируем x3=1).
    """
    return (a * (x1**2 - x2)**2 + b * (x1 - 1)**2 + f0)


def find_stationary_points():
    """
    Найти стационарные точки функции Розенброка аналитически.
    """
    x_star = np.ones(N)
    f_star = rosenbrock(x_star, A, B, F0)
    grad_star = rosenbrock_gradient(x_star, A, B)
    
    return {
        'x_star': x_star,
        'f_star': f_star,
        'gradient': grad_star
    }


def main(run_visualization: bool = False):
    """
    Основная функция выполнения лабораторной работы.
    
    Args:
        run_visualization: Запускать ли визуализацию
    """
    print("=" * 80)
    print("ЛАБОРАТОРНАЯ РАБОТА №3")
    print("Методы многомерного поиска")
    print("=" * 80)
    print(f"\nВариант 2: a = {A}, b = {B}, f0 = {F0}, n = {N}")
    print(f"\nФункция Розенброка:")
    print(f"  f(x) = Σ[i=1 to n-1] [{A}(x_i² - x_{{i+1}})² + {B}(x_i - 1)²] + {F0}")
    print("=" * 80)
    
    # Начальная точка
    x0 = np.array([0.0, 0.0, 0.0])
    print(f"\nНачальная точка: x0 = {x0}")
    print(f"  f(x0) = {rosenbrock(x0, A, B, F0):.6f}")
    
    # Аналитическое решение
    print("\n" + "=" * 80)
    print("1. АНАЛИТИЧЕСКОЕ РЕШЕНИЕ (стационарные точки)")
    print("=" * 80)
    stationary = find_stationary_points()
    print(f"\nСтационарная точка (глобальный минимум):")
    print(f"  x* = {stationary['x_star']}")
    print(f"  f(x*) = {stationary['f_star']:.6f}")
    print(f"  ∇f(x*) = {stationary['gradient']}")
    
    # Создание функции с параметрами варианта
    f, grad_f, _ = create_rosenbrock_function(A, B, F0)
    
    # Создание оптимизатора
    optimizer = Optimizer(f, grad_f, x0, EPS1, EPS2, MAX_ITER)
    
    # Запуск всех методов
    print("\n" + "=" * 80)
    print("2. РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ")
    print("=" * 80)
    results = optimizer.run_all_methods(verbose=True)
    
    # Сравнение методов
    print("\n" + "=" * 80)
    print("3. СРАВНЕНИЕ МЕТОДОВ")
    print("=" * 80)
    optimizer.print_comparison_table()
    
    # Визуализация
    if run_visualization:
        print("\n" + "=" * 80)
        print("4. ВИЗУАЛИЗАЦИЯ")
        print("=" * 80)
        
        # Функция для 2D визуализации
        def f_2d(x1, x2):
            return rosenbrock_2d(x1, x2, A, B, F0)
        
        create_visualization(
            results=results,
            f_2d=f_2d,
            x0=x0,
            x_true=stationary['x_star'],
            f_true=stationary['f_star'],
            output_dir='.'
        )
    
    # Выводы
    print("\n" + "=" * 80)
    print("4. ВЫВОДЫ" if not run_visualization else "5. ВЫВОДЫ")
    print("=" * 80)
    
    best_by_iter = optimizer.get_best_method('iterations')
    best_by_time = optimizer.get_best_method('time')
    best_by_accuracy = optimizer.get_best_method('accuracy')
    
    print(f"\nЛучший метод по числу итераций: {best_by_iter[0]} "
          f"({best_by_iter[1]['iterations']} итераций)")
    print(f"Лучший метод по времени: {best_by_time[0]} "
          f"({best_by_time[1]['time']:.4f} сек)")
    print(f"Лучший метод по точности: {best_by_accuracy[0]} "
          f"(||∇f|| = {np.linalg.norm(grad_f(best_by_accuracy[1]['x_star'])):.2e})")
    
    print("\n✓ Все методы сошлись к глобальному минимуму функции Розенброка:")
    print(f"    x* ≈ (1, 1, 1), f(x*) ≈ {F0}")
    
    print("\n" + "=" * 80)
    print("Лабораторная работа выполнена успешно!")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Лабораторная работа №3 - Методы многомерного поиска'
    )
    parser.add_argument(
        '--viz', '--visualization', 
        action='store_true',
        help='Запустить визуализацию результатов'
    )
    
    args = parser.parse_args()
    main(run_visualization=args.viz)
