"""
Пакет для методов условной оптимизации
Лабораторная работа №4
"""

from .functions import (
    rosenbrock,
    rosenbrock_gradient,
    rosenbrock_hessian,
    create_rosenbrock_function,
    get_constraints,
    get_constraints_variant3,  # Обновлено для третьего варианта
    check_constraints,
    penalty_function,
    barrier_function
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

from .optimizer import (
    ConstrainedOptimizer,
    create_optimizer_variant3  # Обновлено для третьего варианта
)
from .visualization import (
    plot_feasible_region,
    plot_trajectories_3d,
    plot_trajectories_2d_combined,
    plot_trajectories_2d_individual,
    plot_trajectories_per_constraint,
    plot_convergence,
    plot_constraints_check,
    create_visualization,
    plot_all_constraints_grid  # Добавлено
)

__all__ = [
    # Функции
    'rosenbrock',
    'rosenbrock_gradient',
    'rosenbrock_hessian',
    'create_rosenbrock_function',
    'get_constraints',
    'get_constraints_variant3',  # Обновлено для третьего варианта
    'check_constraints',
    'penalty_function',
    'barrier_function',

    # Методы штрафных функций
    'penalty_method',
    'barrier_method',
    'combined_penalty_barrier_method',

    # Метод множителей Лагранжа
    'modified_lagrange_method',

    # Метод проекции градиента
    'projected_gradient_method',

    # Класс оптимизатора
    'ConstrainedOptimizer',
    'create_optimizer_variant3',  # Обновлено для третьего варианта

    # Визуализация
    'plot_feasible_region',
    'plot_trajectories_3d',
    'plot_convergence',
    'plot_constraints_check',
    'create_visualization',
    'plot_all_constraints_grid'  # Добавлено
]