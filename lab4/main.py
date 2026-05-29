"""
Главный файл для запуска лабораторной работы №4
Методы поиска условного экстремума

Вариант 3: a=80, b=3, f0=110, n=2

Ограничения:
  g1(x) = x1² - x2² - 3 ≤ 0
  g2(x) = -x1 ≤ 0
  g3(x) = -x2 ≤ 0

Выполнил: Студент группы ИУ5Ц-21М Перевощиков Н.Д.
"""

import numpy as np
import argparse
from constrained_optimization import (
    ConstrainedOptimizer,
    create_optimizer_variant3,
    rosenbrock,
    get_constraints_variant3,
    check_constraints,
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
EPS = 1e-6
MAX_ITER = 100


def find_analytical_solution():
    """
    Найти аналитическое решение (если возможно).
    Для задачи с ограничениями это сложно, но проверим точку (1,1).
    """
    x_test = np.array([1.0, 1.0])  # Точка (1,1) для n=2
    constraints = get_constraints_variant3()
    
    # Проверка допустимости
    feasible, info = check_constraints(x_test, constraints)
    
    # Значение функции
    f_val = rosenbrock(x_test, A, B, F0)
    
    return {
        'x': x_test,
        'f': f_val,
        'feasible': feasible,
        'violations': info['violations']
    }


def main(run_visualization: bool = False):
    """
    Основная функция выполнения лабораторной работы.
    """
    print("=" * 80)
    print("ЛАБОРАТОРНАЯ РАБОТА №4")
    print("Методы поиска условного экстремума")
    print("=" * 80)
    print(f"\nВариант 3: a = {A}, b = {B}, f0 = {F0}, n = {N}")
    print(f"\nФункция Розенброка:")
    print(f"  f(x) = Σ[i=1 to n-1] [{A}(x_i² - x_{{i+1}})² + {B}(x_i - 1)²] + {F0}")
    print("=" * 80)
    
    # Ограничения
    print("\nОграничения:")
    print("  g1(x) = x1² - x2² - 3 ≤ 0")
    print("  g2(x) = -x1 ≤ 0  (x1 ≥ 0)")
    print("  g3(x) = -x2 ≤ 0  (x2 ≥ 0)")
    print("=" * 80)
    
    # Начальная точка
    x0 = np.array([1.5, 1.5])  # Допустимая точка для варианта 3
    print(f"\nНачальная точка: x0 = {x0}")
    
    # Проверка допустимости начальной точки
    constraints = get_constraints_variant3()
    feasible, info = check_constraints(x0, constraints)
    print(f"  Допустима: {feasible}")
    if not feasible:
        print(f"  Нарушения: {info['violations']}")
    
    print(f"  f(x0) = {rosenbrock(x0, A, B, F0):.6f}")
    
    # Проверка точки (1,1)
    print("\n" + "=" * 80)
    print("1. ПРОВЕРКА ТОЧКИ (1, 1)")
    print("=" * 80)
    analytical = find_analytical_solution()
    print(f"\nТочка: x = {analytical['x']}")
    print(f"  f(x) = {analytical['f']:.6f}")
    print(f"  Допустима: {analytical['feasible']}")
    if analytical['violations']:
        print(f"  Нарушения:")
        for i, val in analytical['violations']:
            print(f"    g{i+1}(x) = {val:.6f} > 0")
    
    # Создание оптимизатора
    optimizer = create_optimizer_variant3(x0=x0, eps=EPS, max_iter=MAX_ITER)
    
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
    
    # Выводы
    print("\n" + "=" * 80)
    print("4. ВЫВОДЫ")
    print("=" * 80)
    
    best_by_f = optimizer.get_best_method('f_star')
    best_by_iter = optimizer.get_best_method('iterations')
    best_by_time = optimizer.get_best_method('time')
    
    print(f"\nЛучший по значению функции: {best_by_f[0]} "
          f"(f(x*)={best_by_f[1]['f_star']:.8f})")
    print(f"Лучший по числу итераций: {best_by_iter[0]} "
          f"({best_by_iter[1]['iterations']} итераций)")
    print(f"Лучший по времени: {best_by_time[0]} "
          f"({best_by_time[1]['time']:.4f} сек)")
    
    # Проверка допустимости лучшего решения
    best_name, best_result = best_by_f
    x_star = best_result['x_star']
    feasible, info = check_constraints(x_star, constraints)
    
    print(f"\nЛучшее допустимое решение:")
    print(f"    x* = {x_star}")
    print(f"    f(x*) = {best_result['f_star']:.8f}")
    print(f"    Допустимо: {feasible}")
    
    if feasible:
        print(f"    Все ограничения выполнены:")
        for i, g in enumerate(constraints):
            print(f"    g{i+1}(x*) = {g(x_star):.6f} ≤ 0")
    
    # Визуализация
    if run_visualization:
        print("\n" + "=" * 80)
        print("5. ВИЗУАЛИЗАЦИЯ")
        print("=" * 80)
        
        create_visualization(
            results=results,
            constraints=constraints,
            f_opt=110.0,  # Минимальное значение функции
            output_dir='.'
        )
    
    print("\n" + "=" * 80)
    print("Лабораторная работа выполнена успешно!")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Лабораторная работа №4 - Методы условной оптимизации'
    )
    parser.add_argument(
        '--viz', '--visualization', 
        action='store_true',
        help='Запустить визуализацию результатов'
    )
    
    args = parser.parse_args()
    main(run_visualization=args.viz)