"""
Модуль визуализации для методов условной оптимизации
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from typing import Dict, List, Callable, Optional, Tuple
import os


def plot_feasible_region(constraints: List[Callable],
                          x_bounds: Tuple = (-4.5, 1.5),
                          save_path: Optional[str] = None):
    """
    Построить область допустимых решений для n=2.
    """
    x1 = np.linspace(x_bounds[0], x_bounds[1], 200)
    x2 = np.linspace(x_bounds[0], x_bounds[1], 200)
    X1, X2 = np.meshgrid(x1, x2)
    
    # ИСПРАВЛЕНО: Изменяем на 2x2 сетку
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten() # Преобразуем в одномерный массив для удобной итерации
    
    for i, (g, ax) in enumerate(zip(constraints, axes)):
        # Для 3D переменных фиксируем x3
        Z = np.zeros_like(X1)
        for j in range(len(x1)):
            for k in range(len(x2)):
                # Для варианта 3:
                Z[k, j] = g(np.array([x1[j], x2[k]]))  # Для n=2, без фиксации x3
        
        # Область где g(x) ≤ 0
        feasible = Z <= 0
        
        ax.contourf(X1, X2, feasible, levels=[0.5, 1], colors=['lightgreen'], alpha=0.5)
        ax.contour(X1, X2, Z, levels=[0], colors='red', linewidths=2)
        ax.contourf(X1, X2, Z, levels=[-100, 0], colors=['lightgreen'], alpha=0.3)
        
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title(f'g{i+1}(x) ≤ 0')
        ax.grid(True, alpha=0.3)

    fig.suptitle('Область допустимых решений (индивидуальные ограничения)', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Область допустимых решений сохранена в '{save_path}'")
    
    plt.show()


def plot_all_constraints_grid(constraints: List[Callable],
                               save_path: Optional[str] = None):
    """
    Построить сетку 2x2 с визуализацией каждого ограничения.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    constraint_names = [
        'g1: x1² - x2² - 3 ≤ 0',
        'g2: -x1 ≤ 0 (x1 ≥ 0)',
        'g3: -x2 ≤ 0 (x2 ≥ 0)'
    ]
    
    x1 = np.linspace(-0.5, 1.5, 200)
    x2 = np.linspace(-0.5, 1.5, 200)
    X1, X2 = np.meshgrid(x1, x2)
    x3_fixed = 1.0
    
    for i, (g, name, ax) in enumerate(zip(constraints, constraint_names, axes)):
        # Вычисляем значение ограничения
        Z = np.zeros_like(X1)
        for j in range(len(x1)):
            for k in range(len(x2)):
                Z[k, j] = g(np.array([x1[j], x2[k]]))
        
        # Область где g(x) ≤ 0
        feasible = Z <= 0
        
        ax.contourf(X1, X2, feasible, levels=[0.5, 1], colors=['lightgreen'], alpha=0.5)
        ax.contour(X1, X2, Z, levels=[0], colors='red', linewidths=2)
        
        # Добавим цветовую шкалу
        contour_full = ax.contourf(X1, X2, Z, levels=30, cmap='RdYlGn_r', alpha=0.6)
        cbar = fig.colorbar(contour_full, ax=ax)
        cbar.set_label('g(x)')
        
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title(f'{name}\n(проекция при x3={x3_fixed})')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)

    fig.suptitle('Визуализация каждого ограничения', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Сетка ограничений сохранена в '{save_path}'")
    
    plt.show()


def plot_trajectories_3d(results: Dict, constraints: List[Callable],
                          x_bounds: Tuple = (0, 1.5),
                          save_path: Optional[str] = None):
    """
    Построить траектории методов в 3D.
    """
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # === 1. Визуализация границ ограничений ===
    
    # Сетка для поверхностей
    x_range = np.linspace(x_bounds[0], x_bounds[1], 20)
    y_range = np.linspace(x_bounds[0], x_bounds[1], 20)
    
    # Поверхность g1(x) = 0 => x3 = x1² + x2² - 1
    X1_g1, X2_g1 = np.meshgrid(x_range, y_range)
    X3_g1 = X1_g1**2 + X2_g1**2 - 1
    ax.plot_surface(X1_g1, X2_g1, X3_g1, alpha=0.5, cmap=cm.viridis)

    # Определяем z_bounds на основе поверхности g1
    z_bounds = (np.min(X3_g1), np.max(X1_g1**2 + X2_g1**2))
    z_range = np.linspace(z_bounds[0], z_bounds[1], 20)
    
    # Плоскость g2(x) = 0 => x1 = 0 (плоскость Y-Z)
    Y2_g2, Z2_g2 = np.meshgrid(y_range, z_range)
    X2_g2 = np.zeros_like(Y2_g2)
    ax.plot_surface(X2_g2, Y2_g2, Z2_g2, alpha=0.3, color='red')
    
    # Плоскость g3(x) = 0 => x2 = 0 (плоскость X-Z)
    X3_g3, Z3_g3 = np.meshgrid(x_range, z_range)
    Y3_g3 = np.zeros_like(X3_g3)
    ax.plot_surface(X3_g3, Y3_g3, Z3_g3, alpha=0.3, color='green')
    
    # === 2. Траектории методов ===
    colors = list(cm.get_cmap('tab10')(np.linspace(0, 1, len(results))))
    
    for (name, result), color in zip(results.items(), colors):
        if 'history' in result and 'x' in result['history']:
            x_path = np.array(result['history']['x'])
            ax.plot(x_path[:, 0], x_path[:, 1], x_path[:, 2], 
                   'o-', color=color, label=name, markersize=3, linewidth=1.5, alpha=0.8)
            
            # Начало и конец
            ax.scatter(x_path[0, 0], x_path[0, 1], x_path[0, 2], 
                      color='black', s=80, marker='s', alpha=0.9, label='Старт' if name == list(results.keys())[0] else "")
            ax.scatter(x_path[-1, 0], x_path[-1, 1], x_path[-1, 2], 
                      color=color, s=120, marker='*', alpha=0.9)
    
    # Глобальный минимум (1,1,1)
    ax.scatter(1, 1, color='magenta', s=250, marker='*', label='Минимум (1,1)', edgecolors='black')
    
    ax.set_xlabel('x1', fontweight='bold')
    ax.set_ylabel('x2', fontweight='bold')
    ax.set_zlabel('x3', fontweight='bold')
    ax.set_title('Траектории методов условной оптимизации в 3D', fontsize=16)
    
    # Собираем легенду, избегая дубликатов
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(-0.2, 0.5), fontsize=8)
    
    ax.view_init(elev=20, azim=-60) # Устанавливаем хороший ракурс
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"3D траектории сохранены в '{save_path}'")
    
    plt.show()


def plot_convergence(results: Dict, f_opt: float = 100.0,
                      save_path: Optional[str] = None):
    """
    Построить графики сходимости методов с нормализацией по итерациям.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    method_names = list(results.keys())
    colors = list(cm.get_cmap('tab10')(np.linspace(0, 1, len(method_names))))
    
    # Собираем все данные для нормализации
    all_f = []
    all_violations = []
    max_iter = 0
    for name, result in results.items():
        if 'history' in result and 'f' in result['history']:
            all_f.extend(result['history']['f'])
            max_iter = max(max_iter, len(result['history']['f']))
        if 'history' in result and 'x' in result['history']:
            for x in result['history']['x']:
                g1 = x[0]**2 - x[1]**2 - 3
                g2 = -x[0]
                g3 = -x[1]
                all_violations.append(max(0, g1, g2, g3))
    
    f_min, f_max = min(all_f), max(all_f)
    f_range = f_max - f_min if f_max > f_min else 1
    violation_max = max(all_violations) if all_violations else 1
    
    # 1. Сходимость по функции (НОРМИРОВАННАЯ по X и Y)
    ax = axes[0, 0]
    for (name, result), color in zip(results.items(), colors):
        if 'history' in result and 'f' in result['history']:
            f_history = np.array(result['history']['f'])
            f_norm = (f_history - f_min) / f_range  # Нормализация Y к [0, 1]
            
            # Нормализация X (итерации) к [0, 1]
            n_iter = len(f_history)
            x_norm = np.linspace(0, 1, n_iter)
            
            ax.plot(x_norm, f_norm, label=name, linewidth=2, color=color)
    
    ax.axhline(y=0, color='red', linestyle='--', label=f'Оптимум (норм.)')
    ax.set_xlabel('Итерации (норм.)')
    ax.set_ylabel('f(x) (норм.)')
    ax.set_title('Сходимость по значению функции (нормализованная)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 2. Нарушение ограничений (НОРМИРОВАННОЕ по X и Y)
    ax = axes[0, 1]
    for (name, result), color in zip(results.items(), colors):
        if 'history' in result and 'x' in result['history']:
            violations = []
            for x in result['history']['x']:
                g1 = x[0]**2 - x[1]**2 - 3
                g2 = -x[0]
                g3 = -x[1]
                max_violation = max(0, g1, g2, g3)
                violations.append(max_violation / violation_max if violation_max > 0 else 0)
            
            # Нормализация X (итерации) к [0, 1]
            n_iter = len(violations)
            x_norm = np.linspace(0, 1, n_iter)
            
            ax.plot(x_norm, violations, label=name, linewidth=2, color=color)
    
    ax.set_xlabel('Итерации (норм.)')
    ax.set_ylabel('max violation (норм.)')
    ax.set_title('Нарушение ограничений (нормализованное)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 3. Параметр штрафа r
    ax = axes[1, 0]
    has_r = False
    for (name, result), color in zip(results.items(), colors):
        if 'history' in result and 'r' in result['history']:
            has_r = True
            r_history = result['history']['r']
            # Нормализация X
            n_iter = len(r_history)
            x_norm = np.linspace(0, 1, n_iter)
            ax.semilogy(x_norm, r_history, label=name, linewidth=2, color=color)
    
    if has_r:
        ax.set_xlabel('Итерации (норм.)')
        ax.set_ylabel('r (лог)')
        ax.set_title('Параметр штрафа')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Параметр r не используется', ha='center', va='center')
    
    # 4. Время vs Точность
    ax = axes[1, 1]
    times = [results[name]['time'] for name in method_names]
    f_stars = [results[name]['f_star'] for name in method_names]
    
    bars = ax.bar(method_names, times, color=colors, alpha=0.8)
    ax.set_ylabel('Время (сек)')
    ax.set_title('Время выполнения')
    ax.tick_params(axis='x', rotation=45)
    
    for bar, val, f_val in zip(bars, times, f_stars):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{val:.3f}с\nf={f_val:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Графики сходимости сохранены в '{save_path}'")
    
    plt.show()


def plot_constraints_check(results: Dict, constraints: List[Callable],
                            save_path: Optional[str] = None):
    """
    Построить проверку ограничений для лучших решений.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    method_names = list(results.keys())
    colors = list(cm.get_cmap('tab10')(np.linspace(0, 1, len(method_names))))
    
    n_constraints = len(constraints)
    x = np.arange(n_constraints)
    width = 0.8 / len(method_names)
    
    for i, (name, result) in enumerate(results.items()):
        x_star = result['x_star']
        g_values = [g(x_star) for g in constraints]
        
        bars = ax.bar(x + i*width - 0.4, g_values, width, label=name, color=colors[i], alpha=0.8)
        
        # Добавляем значения над столбцами
        for bar, g_val in zip(bars, g_values):
            height = bar.get_height()
            # Позиция текста зависит от знака значения
            y_pos = height + (0.02 if height >= 0 else -0.03)
            va = 'bottom' if height >= 0 else 'top'
            
            ax.text(bar.get_x() + bar.get_width()/2, y_pos, 
                    f'{g_val:.4f}', ha='center', va=va, fontsize=7, rotation=45)
    
    ax.axhline(y=0, color='red', linestyle='-', linewidth=2, label='g(x)=0')
    ax.set_xlabel('Ограничение')
    ax.set_ylabel('g(x)')
    ax.set_title('Проверка ограничений для найденных решений\n(значения указаны над столбцами)')
    ax.set_xticks(range(n_constraints))
    ax.set_xticklabels([f'g{i+1}' for i in range(n_constraints)])
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Проверка ограничений сохранена в '{save_path}'")
    
    plt.show()


def plot_trajectories_2d_combined(results: Dict, constraints: List[Callable],
                                   save_path: Optional[str] = None):
    """
    Построить все траектории методов на одном 2D графике (проекция x1-x2).
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Область допустимых решений
    x1 = np.linspace(0, 1.5, 200)
    x2 = np.linspace(0, 1.5, 200)
    X1, X2 = np.meshgrid(x1, x2)
    
    # Для 2D проекции фиксируем x3 = 1 (значение в точке минимума)
    x3_fixed = 1.0
    
    # g1: x1² - x2² - 3 ≤ 0
    Z1 = X1**2 - X2**2 - 3

    # g2: -x1 ≤ 0 => x1 ≥ 0
    # g3: -x2 ≤ 0 => x2 ≥ 0
    feasible = (Z1 <= 0) & (X1 >= 0) & (X2 >= 0)
    
    ax.contourf(X1, X2, feasible, levels=[0.5, 1], colors=['lightgreen'], alpha=0.3)
    ax.contour(X1, X2, Z1, levels=[0], colors='blue', linewidths=2, linestyles='--', label='g1(x)=0')
    
    # Границы g2 и g3
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='g2(x)=0 (x1=0)')
    ax.axhline(y=0, color='green', linestyle='--', linewidth=2, label='g3(x)=0 (x2=0)')
    
    # Траектории методов
    colors = list(cm.get_cmap('tab10')(np.linspace(0, 1, len(results))))
    
    for (name, result), color in zip(results.items(), colors):
        if 'history' in result and 'x' in result['history']:
            x_path = np.array(result['history']['x'])
            ax.plot(x_path[:, 0], x_path[:, 1], 'o-', color=color, 
                   label=name, markersize=3, linewidth=1.5, alpha=0.7)
            
            # Начало и конец
            ax.plot(x_path[0, 0], x_path[0, 1], 's', color=color, markersize=10, alpha=0.8)
            ax.plot(x_path[-1, 0], x_path[-1, 1], '*', color=color, markersize=12, alpha=0.8)
    
    # Глобальный минимум
    ax.plot(1, 1, 'r*', markersize=15, label='Минимум (1,1)')
    
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Все траектории методов (проекция x1-x2, x3=1)')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, 2.5)  # Расширяем диапазон для g1
    ax.set_ylim(-0.5, 2.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"2D траектории (общий) сохранены в '{save_path}'")
    
    plt.show()


def plot_trajectories_2d_individual(results: Dict, constraints: List[Callable],
                                     save_path: Optional[str] = None):
    """
    Построить отдельные 2D графики для каждого метода (общие для всех ограничений).
    """
    colors = list(cm.get_cmap('tab10')(np.linspace(0, 1, len(results))))
    
    # Создаём директорию для индивидуальных графиков
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    
    for (name, result), color in zip(results.items(), colors):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        if 'history' not in result or 'x' not in result['history']:
            continue
        
        x_path = np.array(result['history']['x'])
        
        # Область допустимых решений (все ограничения)
        x1 = np.linspace(0, 1.5, 200)
        x2 = np.linspace(0, 1.5, 200)
        X1, X2 = np.meshgrid(x1, x2)
        x3_fixed = 1.0
        
        Z1 = X1**2 - X2**2 - 3
        feasible = (Z1 <= 0) & (X1 >= 0) & (X2 >= 0)
        
        # График 1: Траектория на области допустимых решений
        ax = axes[0]
        ax.contourf(X1, X2, feasible, levels=[0.5, 1], colors=['lightgreen'], alpha=0.3)
        ax.contour(X1, X2, Z1, levels=[0], colors='blue', linewidths=2, linestyles='--')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
        ax.axhline(y=0, color='green', linestyle='--', linewidth=1.5)
        
        ax.plot(x_path[:, 0], x_path[:, 1], 'o-', color=color, markersize=4, linewidth=2, alpha=0.8)
        ax.plot(x_path[0, 0], x_path[0, 1], 'gs', markersize=12, label='Старт', alpha=0.8)
        ax.plot(x_path[-1, 0], x_path[-1, 1], 'r*', markersize=15, label='Финиш', alpha=0.8)
        
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title(f'{name}\nТраектория (все ограничения)')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_xlim(0, 1.5)
        ax.set_ylim(0, 1.5)
        
        # График 2: Сходимость по функции
        ax = axes[1]
        f_history = result['history']['f']
        ax.plot(range(len(f_history)), f_history, color=color, linewidth=2)
        ax.axhline(y=100, color='red', linestyle='--', label='Оптимум (100)')
        ax.set_xlabel('Итерация')
        ax.set_ylabel('f(x)')
        ax.set_title(f'{name}\nСходимость по значению функции')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
            file_path = os.path.join(save_path, f'trajectory_2d_{safe_name}.png')
            plt.savefig(file_path, dpi=150, bbox_inches='tight')
            print(f"Траектория '{name}' сохранена в '{file_path}'")
        
        plt.show()


def plot_trajectories_per_constraint(results: Dict, constraints: List[Callable],
                                      save_path: Optional[str] = None):
    """
    Построить 5 графиков (по одному на метод), каждый с 4 подграфиками ограничений.
    5 методов × 1 график (4 подграфика) = 5 графиков.
    """
    colors = list(cm.get_cmap('tab10')(np.linspace(0, 1, len(results))))
    constraint_names = [
        'g1: x1² - x2² - 3 ≤ 0',
        'g2: -x1 ≤ 0 (x1 ≥ 0)',
        'g3: -x2 ≤ 0 (x2 ≥ 0)'
    ]
    
    # Создаём директорию
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    
    for (method_name, result), method_color in zip(results.items(), colors):
        if 'history' not in result or 'x' not in result['history']:
            continue
        
        x_path = np.array(result['history']['x'])
        f_history = result['history']['f']
        
        # Создаём 4 подграфика - по одному для каждого ограничения
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        # ИСПРАВЛЕНО: Расширяем диапазон, чтобы contour работал на границах x=0, y=0
        x1 = np.linspace(-0.5, 1.5, 200)
        x2 = np.linspace(-0.5, 1.5, 200)
        X1, X2 = np.meshgrid(x1, x2)
        x3_fixed = 1.0
        
        for i, (g, g_name) in enumerate(zip(constraints, constraint_names)):
            ax = axes[i]
            
            # Вычисляем значение ограничения
            Z = np.zeros_like(X1)
            for j in range(len(x1)):
                for k in range(len(x2)):
                    Z[k, j] = g(np.array([x1[j], x2[k]]))
            
            # Градиентная цветовая шкала
            contour_full = ax.contourf(X1, X2, Z, levels=50, cmap='RdYlGn_r', alpha=0.7)
            cbar = plt.colorbar(contour_full, ax=ax)
            cbar.set_label('g(x)')
            
            # Чёрная линия границы g(x) = 0 (универсальный метод)
            ax.contour(X1, X2, Z, levels=[0], colors='black', linewidths=3)

            # Для g4, где нет линии, добавим поясняющий текст
            if i == 3:
                ax.text(0.5, 0.5, 'g4: x3 ≥ 0\n(выполнено при x3=1, нет линии g(x)=0)', 
                       ha='center', va='center', fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            
            # Траектория метода
            ax.plot(x_path[:, 0], x_path[:, 1], 'o-', color=method_color, 
                   markersize=5, linewidth=2.5, alpha=0.9, label='Траектория')
            ax.plot(x_path[0, 0], x_path[0, 1], 'gs', markersize=15, label='Старт', alpha=0.9, zorder=10)
            ax.plot(x_path[-1, 0], x_path[-1, 1], 'r*', markersize=20, label='Финиш', alpha=0.9, zorder=10)
            
            # Точка минимума (1,1)
            ax.plot(1, 1, 'b+', markersize=25, linewidth=3, label='Минимум (1,1)', zorder=10)
            
            ax.set_xlabel('x1', fontsize=12)
            ax.set_ylabel('x2', fontsize=12)
            ax.set_title(f'{g_name}\n(проекция при x3={x3_fixed})', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9, loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            # ИСПРАВЛЕНО: Обновляем пределы, чтобы соответствовать новому диапазону
            ax.set_xlim(-0.5, 1.5)
            ax.set_ylim(-0.5, 1.5)
        
        # Общий заголовок для всех 4 ограничений
        fig.suptitle(f'{method_name}\nТраектория на фоне каждого ограничения\n'
                    f'Финиш: f(x*)={f_history[-1]:.6f}, итераций={len(f_history)}', 
                    fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            safe_method = method_name.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
            file_path = os.path.join(save_path, f'trajectory_all_g_{safe_method}.png')
            plt.savefig(file_path, dpi=150, bbox_inches='tight')
            print(f"Траектория '{method_name} (все g)' сохранена в '{file_path}'")
        
        plt.show()


def create_visualization(results: Dict, constraints: List[Callable],
                          f_opt: float = 100.0,
                          output_dir: str = '.'):
    """
    Создать все визуализации.
    """
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("ГЕНЕРАЦИЯ ВИЗУАЛИЗАЦИИ")
    print("=" * 60)
    
    # 1. Область допустимых решений (все ограничения вместе)
    plot_feasible_region(
        constraints,
        save_path=os.path.join(plots_dir, 'feasible_region_all.png')
    )
    
    # 2. Сетка с визуализацией каждого ограничения
    plot_all_constraints_grid(
        constraints,
        save_path=os.path.join(plots_dir, 'constraints_g_all.png')
    )
    

    
    # 4. 2D траектории - все методы в одном
    plot_trajectories_2d_combined(
        results,
        constraints,
        save_path=os.path.join(plots_dir, 'trajectories_2d_combined.png')
    )
    
    # 5. 2D траектории - отдельные для каждого метода (5 графиков)
    plot_trajectories_2d_individual(
        results,
        constraints,
        save_path=os.path.join(plots_dir, 'trajectories_2d_per_method')
    )
    
    # 6. Траектории методов с каждым ограничением (20 графиков: 5 методов × 4 ограничения)
    plot_trajectories_per_constraint(
        results,
        constraints,
        save_path=os.path.join(plots_dir, 'trajectories_per_constraint')
    )
    
    # 7. Графики сходимости
    plot_convergence(
        results,
        f_opt,
        save_path=os.path.join(plots_dir, 'convergence.png')
    )
    
    # 8. Проверка ограничений
    plot_constraints_check(
        results,
        constraints,
        save_path=os.path.join(plots_dir, 'constraints_check.png')
    )
    
    print(f"\nВсе графики сохранены в директорию '{plots_dir}/'")
