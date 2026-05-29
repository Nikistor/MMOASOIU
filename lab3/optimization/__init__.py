"""
Пакет для методов многомерной оптимизации
Лабораторная работа №3
"""

from .functions import (
    rosenbrock,
    rosenbrock_gradient,
    rosenbrock_hessian,
    create_rosenbrock_function
)

from .line_search import (
    golden_section_search,
    line_search
)

from .gradient_methods import (
    gradient_descent
)

from .conjugate_gradient import (
    fletcher_reeves,
    polak_ribiere
)

from .quasi_newton import (
    dfp,
    bfgs,
    lbfgs
)

from .optimizer import Optimizer

from .visualization import (
    plot_convergence,
    plot_trajectories,
    plot_3d_surface,
    create_visualization
)

__all__ = [
    # Функции
    'rosenbrock',
    'rosenbrock_gradient',
    'rosenbrock_hessian',
    'create_rosenbrock_function',
    
    # Одномерный поиск
    'golden_section_search',
    'line_search',
    
    # Методы оптимизации
    'gradient_descent',
    'fletcher_reeves',
    'polak_ribiere',
    'dfp',
    'bfgs',
    'lbfgs',
    
    # Класс оптимизатора
    'Optimizer',
    
    # Визуализация
    'plot_convergence',
    'plot_trajectories',
    'plot_3d_surface',
    'create_visualization'
]
