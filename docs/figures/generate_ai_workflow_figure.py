#!/usr/bin/env python3
"""
Generate AI Workflow figure for PPT
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# Color scheme
COLORS = {
    'primary': '#2C3E50',
    'accent': '#E67E22',
    'success': '#27AE60',
    'warning': '#F1C40F',
    'light': '#ECF0F1',
    'blue': '#3498DB',
    'red': '#E74C3C',
}

def create_ai_workflow():
    """Create AI workflow comparison figure"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # ========== Top: Previous Workflow ==========
    ax1 = axes[0]
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')

    # Title
    ax1.text(0.5, 0.95, "Previous Workflow", ha='center', va='top',
            fontsize=14, fontweight='bold', color=COLORS['primary'])

    # Boxes for previous workflow
    boxes_prev = [
        (0.05, 'Requirement\nList', COLORS['light']),
        (0.22, 'AI\nPolish', COLORS['blue']),
        (0.39, 'Human\nReview', COLORS['success']),
        (0.56, 'Code\nAgent', COLORS['accent']),
        (0.73, 'Human\nReview', COLORS['success']),
    ]

    box_width = 0.12
    box_height = 0.35
    y_center = 0.5

    for x, label, color in boxes_prev:
        rect = FancyBboxPatch((x, y_center - box_height/2), box_width, box_height,
                             boxstyle="round,pad=0.02,rounding_size=0.02",
                             facecolor=color, edgecolor=COLORS['primary'], linewidth=2)
        ax1.add_patch(rect)
        ax1.text(x + box_width/2, y_center, label, ha='center', va='center',
                fontsize=10, fontweight='bold', color=COLORS['primary'])

    # Arrows
    for i in range(len(boxes_prev) - 1):
        x_start = boxes_prev[i][0] + box_width + 0.01
        x_end = boxes_prev[i+1][0] - 0.01
        ax1.annotate('', xy=(x_end, y_center), xytext=(x_start, y_center),
                    arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2))

    # Loop back arrow
    ax1.annotate('', xy=(0.73 + box_width/2, y_center - box_height/2 - 0.05),
                xytext=(0.56 + box_width/2, y_center - box_height/2 - 0.05),
                arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=2,
                               connectionstyle="arc3,rad=0.3"))

    # Bottleneck annotation
    ax1.text(0.90, 0.5, "⚠️ Bottleneck:\nHard to describe\nsimulation results",
            ha='left', va='center', fontsize=9, color=COLORS['red'],
            bbox=dict(boxstyle='round', facecolor='#FADBD8', edgecolor=COLORS['red']))

    # ========== Bottom: Improved Workflow ==========
    ax2 = axes[1]
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')

    # Title
    ax2.text(0.5, 0.95, "My Improvement: VTK Analysis Agent", ha='center', va='top',
            fontsize=14, fontweight='bold', color=COLORS['accent'])

    # Boxes for improved workflow
    boxes_new = [
        (0.08, 'Run\nSimulation', COLORS['light']),
        (0.28, 'VTK Analysis\nAgent', COLORS['accent']),
        (0.52, 'Human\nGuide', COLORS['success']),
        (0.72, 'Code\nAgent', COLORS['blue']),
    ]

    y_center = 0.55

    for x, label, color in boxes_new:
        rect = FancyBboxPatch((x, y_center - box_height/2), box_width + 0.02, box_height,
                             boxstyle="round,pad=0.02,rounding_size=0.02",
                             facecolor=color, edgecolor=COLORS['primary'], linewidth=2)
        ax2.add_patch(rect)
        ax2.text(x + (box_width + 0.02)/2, y_center, label, ha='center', va='center',
                fontsize=10, fontweight='bold',
                color='white' if color == COLORS['accent'] else COLORS['primary'])

    # Arrows
    for i in range(len(boxes_new) - 1):
        x_start = boxes_new[i][0] + box_width + 0.02 + 0.01
        x_end = boxes_new[i+1][0] - 0.01
        ax2.annotate('', xy=(x_end, y_center), xytext=(x_start, y_center),
                    arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2))

    # VTK Agent description box
    desc_box = FancyBboxPatch((0.25, 0.05), 0.22, 0.25,
                             boxstyle="round,pad=0.02",
                             facecolor='#FEF9E7', edgecolor=COLORS['accent'], linewidth=1.5)
    ax2.add_patch(desc_box)
    ax2.text(0.36, 0.27, "Writes Python scripts\nto extract:", ha='center', va='top',
            fontsize=9, fontweight='bold', color=COLORS['primary'])
    ax2.text(0.36, 0.15, "• T, v ranges\n• Interface shape\n• Phase fractions",
            ha='center', va='center', fontsize=9, color=COLORS['primary'])

    # Arrow from VTK Agent to description
    ax2.annotate('', xy=(0.36, 0.30), xytext=(0.36, y_center - box_height/2 - 0.02),
                arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=1.5))

    # Key benefit annotation
    ax2.text(0.90, 0.5, "✓ Structured\nmetrics for\nCode Agent",
            ha='left', va='center', fontsize=9, color=COLORS['success'],
            bbox=dict(boxstyle='round', facecolor='#D5F5E3', edgecolor=COLORS['success']))

    plt.tight_layout()
    plt.savefig('/home/yzk/LBMProject/docs/figures/ai_workflow_comparison.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: ai_workflow_comparison.png")


def create_my_role_figure():
    """Create 'My Role' summary figure"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Title
    ax.text(0.5, 0.95, '"AI writes code, I make decisions"', ha='center', va='top',
           fontsize=16, fontweight='bold', color=COLORS['accent'], style='italic')

    # Four role boxes
    roles = [
        ('ARCHITECT', ['LBM+GPU approach', 'Modular framework', 'Phase-by-phase strategy'], COLORS['blue']),
        ('PHYSICS JUDGE', ['Identify wrong results', 'Diagnose root cause', 'Select benchmarks'], COLORS['success']),
        ('DEBUG DIRECTOR', ['Spot anomalies', 'Narrow problem scope', 'Guide AI direction'], COLORS['accent']),
        ('QUALITY GATE', ['Review code', 'Reject over-engineering', 'Ensure correctness'], COLORS['primary']),
    ]

    box_width = 0.2
    box_height = 0.35
    y_top = 0.75

    for i, (title, items, color) in enumerate(roles):
        x = 0.05 + i * 0.24

        # Box
        rect = FancyBboxPatch((x, y_top - box_height), box_width, box_height,
                             boxstyle="round,pad=0.02,rounding_size=0.02",
                             facecolor='white', edgecolor=color, linewidth=3)
        ax.add_patch(rect)

        # Title
        ax.text(x + box_width/2, y_top - 0.03, title, ha='center', va='top',
               fontsize=11, fontweight='bold', color=color)

        # Items
        for j, item in enumerate(items):
            ax.text(x + 0.02, y_top - 0.10 - j*0.08, f"• {item}", ha='left', va='top',
                   fontsize=9, color=COLORS['primary'])

    # Bottom summary
    summary_box = FancyBboxPatch((0.15, 0.05), 0.7, 0.18,
                                boxstyle="round,pad=0.02",
                                facecolor=COLORS['accent'], edgecolor=COLORS['primary'], linewidth=2)
    ax.add_patch(summary_box)
    ax.text(0.5, 0.14, "Human expertise essential for:\nArchitecture • Physics judgment • Debug direction • Quality control",
           ha='center', va='center', fontsize=11, fontweight='bold', color='white')

    plt.tight_layout()
    plt.savefig('/home/yzk/LBMProject/docs/figures/my_role_in_ai_dev.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: my_role_in_ai_dev.png")


if __name__ == '__main__':
    print("Generating AI workflow figures...")
    create_ai_workflow()
    create_my_role_figure()
    print("\nAll AI workflow figures generated!")
