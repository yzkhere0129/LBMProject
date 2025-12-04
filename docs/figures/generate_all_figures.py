#!/usr/bin/env python3
"""
Generate all PPT figures
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Color scheme
COLORS = {
    'primary': '#2C3E50',
    'accent': '#E67E22',
    'success': '#27AE60',
    'warning': '#F1C40F',
    'light': '#ECF0F1',
    'white': '#FFFFFF',
    'blue': '#3498DB',
    'red': '#E74C3C',
}

OUTPUT_DIR = '/home/yzk/LBMProject/docs/figures/'

def create_tradeoff_triangle():
    """Create Speed-Accuracy-Generality trade-off triangle"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    top = (0.5, 0.9)
    left = (0.1, 0.15)
    right = (0.9, 0.15)

    triangle = plt.Polygon([top, left, right], fill=False,
                          edgecolor=COLORS['primary'], linewidth=3)
    ax.add_patch(triangle)

    ax.text(top[0], top[1]+0.05, 'Accuracy', ha='center', va='bottom',
            fontsize=16, fontweight='bold', color=COLORS['primary'])
    ax.text(left[0]-0.05, left[1], 'Speed', ha='right', va='center',
            fontsize=16, fontweight='bold', color=COLORS['primary'])
    ax.text(right[0]+0.05, right[1], 'Generality', ha='left', va='center',
            fontsize=16, fontweight='bold', color=COLORS['primary'])

    methods = {
        'FEM/FVM\n(OpenFOAM)': (0.35, 0.65, '#3498DB', 's'),
        'Commercial\n(Flow-3D)': (0.55, 0.55, '#9B59B6', 'D'),
        'LBM+GPU\n(This Work)': (0.50, 0.38, COLORS['accent'], 'o'),
        'Analytical\n(Rosenthal)': (0.25, 0.25, '#95A5A6', '^'),
    }

    for name, (x, y, color, marker) in methods.items():
        ax.scatter(x, y, c=color, s=300, marker=marker, zorder=5, edgecolors='white', linewidth=2)
        if 'This Work' in name:
            ax.annotate(name, (x, y), xytext=(x+0.12, y-0.02),
                       fontsize=11, fontweight='bold', color=color,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, linewidth=2))
        elif 'OpenFOAM' in name:
            ax.annotate(name, (x, y), xytext=(x-0.15, y+0.02),
                       fontsize=10, ha='right', color=color)
        elif 'Flow-3D' in name:
            ax.annotate(name, (x, y), xytext=(x+0.08, y+0.05),
                       fontsize=10, color=color)
        else:
            ax.annotate(name, (x, y), xytext=(x-0.12, y-0.05),
                       fontsize=10, ha='right', color=color)

    ax.text(0.5, 0.02, 'LBM+GPU: Balanced trade-off for rapid process exploration',
            ha='center', va='bottom', fontsize=12, style='italic', color=COLORS['primary'])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'tradeoff_triangle.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: tradeoff_triangle.png")


def create_architecture_diagram():
    """Create modular platform architecture diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    layers = [
        ('APPLICATION LAYER', 0.75, 0.20, ['LPBF ✓', 'DED', 'EBM', '...'],
         [COLORS['success'], COLORS['warning'], COLORS['warning'], COLORS['light']]),
        ('MULTIPHYSICS SOLVER', 0.55, 0.12, None, None),
        ('PHYSICS MODULES', 0.30, 0.18, ['Thermal', 'Fluid', 'VOF', 'Phase\nChange', 'Marangoni', 'Laser'],
         [COLORS['success']]*6),
        ('LBM CORE (CUDA)', 0.08, 0.12, None, None),
    ]

    for layer_name, y, height, modules, colors in layers:
        if modules:
            box_color = COLORS['light']
        elif 'SOLVER' in layer_name:
            box_color = '#AED6F1'
        else:
            box_color = COLORS['accent']

        rect = FancyBboxPatch((0.05, y), 0.9, height,
                              boxstyle="round,pad=0.02,rounding_size=0.02",
                              facecolor=box_color, edgecolor=COLORS['primary'], linewidth=2)
        ax.add_patch(rect)

        if modules:
            ax.text(0.08, y + height - 0.03, layer_name, fontsize=11,
                   fontweight='bold', color=COLORS['primary'], va='top')
            n_modules = len(modules)
            box_width = 0.12
            spacing = (0.85 - n_modules * box_width) / (n_modules + 1)
            for i, (mod, col) in enumerate(zip(modules, colors)):
                x = 0.075 + spacing * (i + 1) + box_width * i
                mod_rect = FancyBboxPatch((x, y + 0.02), box_width, height - 0.06,
                                         boxstyle="round,pad=0.01,rounding_size=0.01",
                                         facecolor=col, edgecolor=COLORS['primary'], linewidth=1.5)
                ax.add_patch(mod_rect)
                ax.text(x + box_width/2, y + height/2, mod, ha='center', va='center',
                       fontsize=9, fontweight='bold', color='white' if col != COLORS['light'] else COLORS['primary'])
        else:
            ax.text(0.5, y + height/2, layer_name, ha='center', va='center',
                   fontsize=13, fontweight='bold',
                   color='white' if 'CUDA' in layer_name else COLORS['primary'])

    arrow_style = dict(arrowstyle='<->', color=COLORS['primary'], lw=2)
    ax.annotate('', xy=(0.5, 0.75), xytext=(0.5, 0.67), arrowprops=arrow_style)
    ax.annotate('', xy=(0.5, 0.55), xytext=(0.5, 0.48), arrowprops=arrow_style)
    ax.annotate('', xy=(0.5, 0.30), xytext=(0.5, 0.20), arrowprops=arrow_style)

    ax.text(0.52, 0.71, 'Config API', fontsize=9, color=COLORS['primary'], style='italic')
    ax.text(0.52, 0.515, 'Coupling', fontsize=9, color=COLORS['primary'], style='italic')
    ax.text(0.52, 0.25, 'CUDA Kernels', fontsize=9, color=COLORS['primary'], style='italic')

    ax.text(0.5, 0.01, '✓ = Implemented    ○ = Planned    |    Modular: Add new physics without changing core',
            ha='center', va='bottom', fontsize=10, color=COLORS['primary'])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'architecture_diagram.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: architecture_diagram.png")


def create_roadmap():
    """Create timeline roadmap"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    milestones = [
        ('Now', 'LPBF\nValidated', COLORS['success'], True),
        ('Q1 2025', 'Multi-track\nScanning', COLORS['warning'], False),
        ('Q2 2025', 'Other AM\n(DED, EBM)', COLORS['light'], False),
        ('2025+', 'Multi-GPU\nOpen Source', COLORS['light'], False),
    ]

    ax.axhline(y=0.5, xmin=0.08, xmax=0.92, color=COLORS['primary'], linewidth=4, zorder=1)

    for i, (time, desc, color, is_current) in enumerate(milestones):
        x = 0.1 + i * 0.27
        circle = plt.Circle((x, 0.5), 0.04, color=color, ec=COLORS['primary'],
                           linewidth=3 if is_current else 2, zorder=3)
        ax.add_patch(circle)
        if is_current:
            ax.text(x, 0.5, '✓', ha='center', va='center', fontsize=16,
                   fontweight='bold', color='white', zorder=4)
        ax.text(x, 0.62, time, ha='center', va='bottom', fontsize=12,
               fontweight='bold', color=COLORS['primary'])
        ax.text(x, 0.38, desc, ha='center', va='top', fontsize=11,
               color=COLORS['primary'], linespacing=1.2)

    ax.annotate('', xy=(0.1, 0.75), xytext=(0.1, 0.67),
               arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=3))
    ax.text(0.1, 0.78, 'Current', ha='center', va='bottom', fontsize=11,
           fontweight='bold', color=COLORS['accent'])

    legend_items = [(COLORS['success'], 'Completed'), (COLORS['warning'], 'In Progress'), (COLORS['light'], 'Planned')]
    for i, (color, label) in enumerate(legend_items):
        ax.add_patch(plt.Rectangle((0.75 + i*0.08, 0.15), 0.02, 0.02,
                                   facecolor=color, edgecolor=COLORS['primary']))
        ax.text(0.78 + i*0.08, 0.16, label, fontsize=9, va='center', color=COLORS['primary'])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'roadmap_timeline.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: roadmap_timeline.png")


def create_limitations_table():
    """Create visual limitations table"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    limitations = [
        ('Interface diffusion', '3-5 cells (vs 1-2 FVM)', 'LBM inherent'),
        ('Gas phase', 'Void boundary', 'Sufficient for metals'),
        ('Single GPU', '~10M cells max', 'Multi-GPU planned'),
        ('Temperature calibration', 'WIP', 'Absorption tuning'),
    ]

    headers = ['Limitation', 'Current Status', 'Note']
    col_widths = [0.3, 0.35, 0.3]
    y_start = 0.85
    row_height = 0.15

    x = 0.025
    for header, width in zip(headers, col_widths):
        rect = FancyBboxPatch((x, y_start), width, row_height,
                             boxstyle="round,pad=0.01", facecolor=COLORS['primary'],
                             edgecolor=COLORS['primary'])
        ax.add_patch(rect)
        ax.text(x + width/2, y_start + row_height/2, header,
               ha='center', va='center', fontsize=11, fontweight='bold', color='white')
        x += width + 0.025

    for i, (lim, status, note) in enumerate(limitations):
        y = y_start - (i + 1) * (row_height + 0.02)
        x = 0.025
        row_color = COLORS['light'] if i % 2 == 0 else 'white'
        for j, (text, width) in enumerate(zip([lim, status, note], col_widths)):
            rect = FancyBboxPatch((x, y), width, row_height,
                                 boxstyle="round,pad=0.01", facecolor=row_color,
                                 edgecolor=COLORS['primary'], linewidth=0.5)
            ax.add_patch(rect)
            ax.text(x + width/2, y + row_height/2, text,
                   ha='center', va='center', fontsize=10, color=COLORS['primary'])
            x += width + 0.025

    ax.text(0.5, 0.08, 'These are inherent trade-offs of LBM approach, not implementation bugs',
            ha='center', va='center', fontsize=10, style='italic', color=COLORS['primary'])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'limitations_table.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: limitations_table.png")


if __name__ == '__main__':
    print("Generating all PPT figures...")
    create_tradeoff_triangle()
    create_architecture_diagram()
    create_roadmap()
    create_limitations_table()
    print("\nAll figures generated!")
    print(f"Location: {OUTPUT_DIR}")
