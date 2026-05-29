"""
Модуль визуализации результатов оптимизации
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Callable, Optional
import os


def plot_convergence(results: Dict, f_true: float = 100, 
                     save_path: Optional[str] = None):
    """
    Построить графики сходимости методов с разделением на быстрые/медленные.
    
    Args:
        results: Словарь с результатами методов
        f_true: Истинное значение минимума функции
        save_path: Путь для сохранения графика (опционально)
    """
    # Разделяем методы на быстрые (< 100 итераций) и медленные
    fast_methods = {}
    slow_methods = {}
    
    for name, result in results.items():
        if result['iterations'] <= 100:
            fast_methods[name] = result
        else:
            slow_methods[name] = result
    
    # Цвета для методов
    all_names = list(results.keys())
    colors_map = dict(zip(all_names, cm.get_cmap('tab10')(np.linspace(0, 1, len(all_names)))))
    
    # Создаём два отдельных графика
    if fast_methods:
        _plot_convergence_group(fast_methods, f_true, "fast", colors_map, save_path)
    
    if slow_methods:
        _plot_convergence_group(slow_methods, f_true, "slow", colors_map, save_path)
    
    # Также создаём сводный график с нормализацией
    _plot_summary_comparison(results, f_true, save_path)


def _plot_convergence_group(results: Dict, f_true: float, group_name: str,
                           colors_map: dict, save_path: Optional[str] = None):
    """
    Построить графики для группы методов.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    method_names = list(results.keys())
    colors = [colors_map[name] for name in method_names]
    
    # 1. Сходимость по функции
    ax = axes[0, 0]
    for name, result in results.items():
        f_history = np.array(result['history']['f'])
        iterations = range(len(f_history))
        ax.plot(iterations, f_history, label=name, linewidth=2, color=colors_map[name])
    ax.set_xlabel('Итерация')
    ax.set_ylabel('f(x)')
    ax.set_title(f'Сходимость по значению функции ({"быстрые" if group_name == "fast" else "медленные"} методы)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 2. Сходимость по градиенту
    ax = axes[0, 1]
    for name, result in results.items():
        grad_history = np.array(result['history']['grad_norm'])
        iterations = range(len(grad_history))
        ax.semilogy(iterations, grad_history, label=name, linewidth=2, color=colors_map[name])
    ax.set_xlabel('Итерация')
    ax.set_ylabel('||∇f(x)||')
    ax.set_title(f'Сходимость по норме градиента ({"быстрые" if group_name == "fast" else "медленные"} методы)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 3. Время выполнения
    ax = axes[1, 0]
    times = [results[name]['time'] for name in method_names]
    bars = ax.bar(method_names, times, color=[colors_map[name] for name in method_names], alpha=0.8)
    ax.set_xlabel('Метод')
    ax.set_ylabel('Время (сек)')
    ax.set_title('Время выполнения')
    ax.tick_params(axis='x', rotation=45)
    
    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{val:.3f}с', ha='center', va='bottom', fontsize=9)
    
    # 4. Число итераций
    ax = axes[1, 1]
    iterations_list = [results[name]['iterations'] for name in method_names]
    bars = ax.bar(method_names, iterations_list, color=[colors_map[name] for name in method_names], alpha=0.8)
    ax.set_ylabel('Итерации')
    ax.set_title('Число итераций')
    ax.tick_params(axis='x', rotation=45)
    
    for bar, val in zip(bars, iterations_list):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(val), ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        base_path = save_path.rsplit('.', 1)[0]
        new_path = f"{base_path}_{group_name}.png"
        plt.savefig(new_path, dpi=150, bbox_inches='tight')
        print(f"График для {group_name} методов сохранён в '{new_path}'")
    
    plt.show()


def _plot_summary_comparison(results: Dict, f_true: float, save_path: Optional[str] = None):
    """
    Построить сводный сравнительный график (нормализованный).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    method_names = list(results.keys())
    colors = list(cm.get_cmap('tab10')(np.linspace(0, 1, len(method_names))))
    
    # Нормализация
    all_times = [results[name]['time'] for name in method_names]
    all_iters = [results[name]['iterations'] for name in method_names]
    
    time_max = max(all_times)
    iter_max = max(all_iters)
    
    # 1. Время (норм.)
    ax = axes[0]
    times_norm = [t / time_max for t in all_times]
    bars = ax.bar(method_names, times_norm, color=colors, alpha=0.8)
    ax.set_ylabel('Время (норм.)')
    ax.set_title('Время выполнения (нормированное)')
    ax.tick_params(axis='x', rotation=45)
    
    for bar, norm_val, raw_val in zip(bars, times_norm, all_times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{norm_val:.2f}\n({raw_val:.3f}с)', ha='center', va='bottom', fontsize=8)
    
    # 2. Итерации (норм.)
    ax = axes[1]
    iters_norm = [i / iter_max for i in all_iters]
    bars = ax.bar(method_names, iters_norm, color=colors, alpha=0.8)
    ax.set_ylabel('Итерации (норм.)')
    ax.set_title('Число итераций (нормированное)')
    ax.tick_params(axis='x', rotation=45)
    
    for bar, norm_val, raw_val in zip(bars, iters_norm, all_iters):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{norm_val:.2f}\n({raw_val})', ha='center', va='bottom', fontsize=8)
    
    # 3. Точность (норм.) - обратная величина
    ax = axes[2]
    grad_norms = [np.linalg.norm([results[name]['history']['grad_norm'][-1]]) for name in method_names]
    grad_max = max(grad_norms)
    grad_norms_norm = [g / grad_max for g in grad_norms]
    bars = ax.bar(method_names, grad_norms_norm, color=colors, alpha=0.8)
    ax.set_ylabel('||∇f|| (норм.)')
    ax.set_title('Точность (нормированная)')
    ax.tick_params(axis='x', rotation=45)
    
    for bar, norm_val, raw_val in zip(bars, grad_norms_norm, grad_norms):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{norm_val:.2f}\n({raw_val:.1e})', ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    
    if save_path:
        base_path = save_path.rsplit('.', 1)[0]
        new_path = f"{base_path}_summary.png"
        plt.savefig(new_path, dpi=150, bbox_inches='tight')
        print(f"Сводный график сохранён в '{new_path}'")
    
    plt.show()


def plot_trajectories(results: Dict, f_2d: Callable,
                      x0: np.ndarray, x_true: np.ndarray,
                      f_true: float = 100,
                      save_path: Optional[str] = None):
    """
    Построить траектории методов на поверхности функции.

    Args:
        results: Словарь с результатами методов
        f_2d: 2D функция для визуализации
        x0: Начальная точка
        x_true: Истинная точка минимума
        f_true: Истинное значение минимума
        save_path: Путь для сохранения графика (опционально)
    """
    # Создание сетки для 2D проекции
    x1 = np.linspace(-0.5, 1.5, 100)
    x2 = np.linspace(-0.5, 1.5, 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros_like(X1)
    
    for i in range(len(x1)):
        for j in range(len(x2)):
            Z[j, i] = f_2d(x1[i], x2[j])
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Поверхность функции
    ax = axes[0, 0]
    contour = ax.contourf(X1, X2, Z, levels=50, cmap='viridis', alpha=0.8)
    ax.plot(x_true[0], x_true[1], 'r*', markersize=15, label='Глобальный минимум')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Поверхность функции (проекция x1-x2)')
    ax.legend()
    plt.colorbar(contour, ax=ax)
    
    # 2. Траектории всех методов (проецируем 3D на 2D)
    ax = axes[0, 1]
    contour = ax.contourf(X1, X2, Z, levels=50, cmap='viridis', alpha=0.6)
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    for (name, result), color in zip(results.items(), colors):
        x_path = np.array(result['history']['x'])
        # Проецируем 3D траекторию на 2D (первые две координаты)
        ax.plot(x_path[:, 0], x_path[:, 1], 'o-', color=color, 
                label=name, markersize=3, linewidth=1.5, alpha=0.7)
    
    ax.plot(x_true[0], x_true[1], 'r*', markersize=15, label='Минимум')
    ax.plot(x0[0], x0[1], 'ks', markersize=10, label='Старт')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Траектории методов оптимизации')
    ax.legend(fontsize=7, loc='upper right')
    
    # 3. Линии уровня с траекторией лучшего метода
    ax = axes[0, 2]
    contour = ax.contour(X1, X2, Z, levels=30, cmap='viridis')
    
    # Найти лучший метод по итерациям
    best_name = min(results.keys(), key=lambda k: results[k]['iterations'])
    x_path = np.array(results[best_name]['history']['x'])
    ax.plot(x_path[:, 0], x_path[:, 1], 'ro-', markersize=4, linewidth=2, 
            label=f'Лучший: {best_name}')
    ax.plot(x_true[0], x_true[1], 'r*', markersize=15)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title(f'Траектория {best_name}')
    ax.legend()
    
    # 4. Сходимость по функции (лог)
    ax = axes[1, 0]
    for (name, result), color in zip(results.items(), colors):
        f_history = np.array(result['history']['f'])
        f_rel = f_history - f_true + 1e-10
        ax.semilogy(range(len(f_history)), f_rel, label=name, 
                   linewidth=2, color=color)
    ax.set_xlabel('Итерация')
    ax.set_ylabel('f(x) - f* (лог)')
    ax.set_title('Сходимость по значению функции')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 5. Сходимость по градиенту
    ax = axes[1, 1]
    for (name, result), color in zip(results.items(), colors):
        grad_history = np.array(result['history']['grad_norm'])
        ax.semilogy(range(len(grad_history)), grad_history, 
                   label=name, linewidth=2, color=color)
    ax.set_xlabel('Итерация')
    ax.set_ylabel('||∇f(x)|| (лог)')
    ax.set_title('Сходимость по норме градиента')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 6. Гистограмма итераций
    ax = axes[1, 2]
    names = list(results.keys())
    iterations = [results[name]['iterations'] for name in names]
    bars = ax.bar(names, iterations, color=colors[:len(names)], alpha=0.8)
    ax.set_ylabel('Число итераций')
    ax.set_title('Число итераций до сходимости')
    ax.tick_params(axis='x', rotation=45)
    
    for bar, val in zip(bars, iterations):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                str(val), ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Траектории сохранены в '{save_path}'")
    
    plt.show()


def plot_all_trajectories(results: Dict, f_2d: Callable,
                          x0: np.ndarray, x_true: np.ndarray,
                          save_path: Optional[str] = None):
    """
    Построить все траектории методов на одном графике с линиями уровня.
    
    Args:
        results: Словарь с результатами методов
        f_2d: 2D функция для визуализации
        x0: Начальная точка
        x_true: Истинная точка минимума
        save_path: Путь для сохранения графика (опционально)
    """
    # Создание сетки для 2D проекции
    x1 = np.linspace(-0.5, 1.5, 150)
    x2 = np.linspace(-0.5, 1.5, 150)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros_like(X1)
    
    for i in range(len(x1)):
        for j in range(len(x2)):
            Z[j, i] = f_2d(x1[i], x2[j])
    
    fig, axes = plt.subplots(1, 1, figsize=(12, 10))
    
    # Линии уровня
    contour = axes.contourf(X1, X2, Z, levels=50, cmap='viridis', alpha=0.8)
    contour_lines = axes.contour(X1, X2, Z, levels=20, colors='white', 
                                  linewidths=0.5, alpha=0.5)
    
    # Цвета для методов
    method_names = list(results.keys())
    colors = list(cm.get_cmap('tab10')(np.linspace(0, 1, len(method_names))))
    
    # Траектории всех методов
    for (name, result), color in zip(results.items(), colors):
        x_path = np.array(result['history']['x'])
        # Проецируем 3D траекторию на 2D (первые две координаты)
        axes.plot(x_path[:, 0], x_path[:, 1], 'o-', color=color, 
                 label=name, markersize=2, linewidth=1, alpha=0.7)
        
        # Отмечаем начало и конец траектории
        axes.plot(x_path[0, 0], x_path[0, 1], 's', color=color, markersize=8, 
                 alpha=0.5)  # Старт
        axes.plot(x_path[-1, 0], x_path[-1, 1], '*', color=color, markersize=12, 
                 alpha=0.8)  # Финиш
    
    # Глобальный минимум
    axes.plot(x_true[0], x_true[1], 'r*', markersize=20, label='Глобальный минимум (1,1)',
             zorder=10)
    
    # Начальная точка
    axes.plot(x0[0], x0[1], 'ks', markersize=12, label='Старт (0,0)', zorder=10)
    
    axes.set_xlabel('x1', fontsize=12)
    axes.set_ylabel('x2', fontsize=12)
    axes.set_title('Все траектории методов оптимизации\n(проекция на плоскость x1-x2)', 
                  fontsize=14)
    axes.legend(loc='upper right', fontsize=8, bbox_to_anchor=(1.35, 1.05))
    
    # Добавляем цветовую шкалу
    cbar = plt.colorbar(contour, ax=axes)
    cbar.set_label('f(x)', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Все траектории сохранены в '{save_path}'")
    
    plt.show()


def plot_individual_trajectories(results: Dict, f_2d: Callable,
                                  x0: np.ndarray, x_true: np.ndarray,
                                  save_path: Optional[str] = None):
    """
    Построить отдельные графики траекторий для каждого метода.
    
    Args:
        results: Словарь с результатами методов
        f_2d: 2D функция для визуализации
        x0: Начальная точка
        x_true: Истинная точка минимума
        save_path: Путь для сохранения графиков (опционально)
    """
    # Создание сетки для 2D проекции
    x1 = np.linspace(-0.5, 1.5, 150)
    x2 = np.linspace(-0.5, 1.5, 150)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros_like(X1)
    
    for i in range(len(x1)):
        for j in range(len(x2)):
            Z[j, i] = f_2d(x1[i], x2[j])
    
    method_names = list(results.keys())
    colors = list(cm.get_cmap('tab10')(np.linspace(0, 1, len(method_names))))
    
    # Создаём отдельный график для каждого метода
    for (name, result), color in zip(results.items(), colors):
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        x_path = np.array(result['history']['x'])
        
        # Линии уровня
        contour = ax.contourf(X1, X2, Z, levels=50, cmap='viridis', alpha=0.8)
        ax.contour(X1, X2, Z, levels=20, colors='white', linewidths=0.5, alpha=0.5)
        
        # Траектория метода
        ax.plot(x_path[:, 0], x_path[:, 1], 'o-', color=color, 
               markersize=3, linewidth=1.5, alpha=0.8, label=f'{name}')
        
        # Отмечаем начало и конец
        ax.plot(x_path[0, 0], x_path[0, 1], 'gs', markersize=10, 
               label='Старт', zorder=10)
        ax.plot(x_path[-1, 0], x_path[-1, 1], 'r*', markersize=15, 
               label='Финиш', zorder=10)
        
        # Глобальный минимум
        ax.plot(x_true[0], x_true[1], 'r*', markersize=20, 
               label=f'Минимум ({x_true[0]},{x_true[1]})', zorder=10)
        
        ax.set_xlabel('x1', fontsize=12)
        ax.set_ylabel('x2', fontsize=12)
        ax.set_title(f'Траектория: {name}\n'
                    f'Итераций: {result["iterations"]}, '
                    f'f(x*)={result["f_star"]:.8f}, '
                    f'Время: {result["time"]:.4f}с', 
                    fontsize=12)
        ax.legend(loc='upper right', fontsize=9)
        
        # Цветовая шкала
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('f(x)', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            # Сохраняем каждый график отдельно
            base_path = save_path.rsplit('.', 1)[0]
            # Создаём безопасное имя файла
            safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
            new_path = f"{base_path}_{safe_name}.png"
            plt.savefig(new_path, dpi=150, bbox_inches='tight')
            print(f"Траектория '{name}' сохранена в '{new_path}'")
        
        plt.show()


def plot_3d_surface(f_2d: Callable, x_true: np.ndarray,
                    save_path: Optional[str] = None):
    """
    3D визуализация функции.
    
    Args:
        f_2d: 2D функция для визуализации
        x_true: Истинная точка минимума
        save_path: Путь для сохранения графика (опционально)
    """
    fig = plt.figure(figsize=(14, 6))
    
    # 3D поверхность
    ax1 = fig.add_subplot(121, projection='3d')
    
    x1 = np.linspace(-0.5, 1.5, 80)
    x2 = np.linspace(-0.5, 1.5, 80)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros_like(X1)
    
    for i in range(len(x1)):
        for j in range(len(x2)):
            Z[j, i] = f_2d(x1[i], x2[j])
    
    ax1.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.9, edgecolor='none')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_zlabel('f(x)')
    ax1.set_title('3D поверхность функции')
    
    # 3D контур
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.contour3D(X1, X2, Z, 50, cmap='viridis')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_zlabel('f(x)')
    ax2.set_title('3D контуры функции')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"3D визуализация сохранена в '{save_path}'")
    
    plt.show()


def create_visualization(results: Dict, f_2d: Callable,
                        x0: np.ndarray, x_true: np.ndarray,
                        f_true: float = 100,
                        output_dir: str = '.'):
    """
    Создать все визуализации.
    
    Args:
        results: Словарь с результатами методов
        f_2d: 2D функция для визуализации
        x0: Начальная точка
        x_true: Истинная точка минимума
        f_true: Истинное значение минимума
        output_dir: Директория для сохранения графиков
    """
    # Создаем директорию для графиков если не существует
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("ГЕНЕРАЦИЯ ВИЗУАЛИЗАЦИИ")
    print("=" * 60)
    
    # 1. Графики сходимости (разделённые на быстрые/медленные + сводный)
    plot_convergence(
        results, 
        f_true, 
        save_path=os.path.join(plots_dir, 'convergence.png')
    )
    
    # 2. Все траектории на одном графике
    plot_all_trajectories(
        results,
        f_2d,
        x0,
        x_true,
        save_path=os.path.join(plots_dir, 'all_trajectories.png')
    )
    
    # 3. Отдельные траектории для каждого метода (6 графиков)
    plot_individual_trajectories(
        results,
        f_2d,
        x0,
        x_true,
        save_path=os.path.join(plots_dir, 'trajectory.png')
    )
    
    # 4. Детальные траектории (сетка 2x3)
    plot_trajectories(
        results, 
        f_2d, 
        x0, 
        x_true,
        f_true,
        save_path=os.path.join(plots_dir, 'trajectories_grid.png')
    )
    
    # 5. 3D поверхность
    plot_3d_surface(
        f_2d, 
        x_true,
        save_path=os.path.join(plots_dir, 'surface_3d.png')
    )
    
    print(f"\nВсе графики сохранены в директорию '{plots_dir}/'")
