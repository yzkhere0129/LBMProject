import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Set up figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Color scheme
fvm_color = '#E74C3C'  # Red
lbm_color = '#27AE60'  # Green
cell_color = '#3498DB'  # Blue
arrow_color = '#7F8C8D'  # Gray

# ============ Left: FVM - Global Pressure Solve ============
ax1 = axes[0]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.set_aspect('equal')
ax1.axis('off')
ax1.set_title('FVM: Pressure Poisson Equation', fontsize=16, fontweight='bold', color=fvm_color, pad=20)

# Draw grid of cells
grid_size = 5
cell_size = 1.2
start_x = 1.5
start_y = 2.5

# Draw cells
for i in range(grid_size):
    for j in range(grid_size):
        x = start_x + i * cell_size
        y = start_y + j * cell_size
        rect = mpatches.FancyBboxPatch((x, y), cell_size*0.8, cell_size*0.8,
                                        boxstyle="round,pad=0.05",
                                        facecolor=cell_color, edgecolor='white', linewidth=2)
        ax1.add_patch(rect)

# Draw arrows connecting ALL cells (global communication)
np.random.seed(42)
for _ in range(25):
    i1, j1 = np.random.randint(0, grid_size, 2)
    i2, j2 = np.random.randint(0, grid_size, 2)
    if (i1, j1) != (i2, j2):
        x1 = start_x + i1 * cell_size + cell_size*0.4
        y1 = start_y + j1 * cell_size + cell_size*0.4
        x2 = start_x + i2 * cell_size + cell_size*0.4
        y2 = start_y + j2 * cell_size + cell_size*0.4
        ax1.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=fvm_color, alpha=0.4, lw=1))

# Add text labels
ax1.text(5, 9.2, 'Global Communication', fontsize=14, ha='center', fontweight='bold', color=fvm_color)
ax1.text(5, 8.5, 'All cells must exchange data', fontsize=11, ha='center', color='#555')
ax1.text(5, 1.3, '→ Hard to parallelize on GPU', fontsize=12, ha='center', fontweight='bold', color=fvm_color)
ax1.text(5, 0.6, 'Poisson solver = global matrix', fontsize=10, ha='center', color='#777')

# ============ Right: LBM - Local Operations ============
ax2 = axes[1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.set_aspect('equal')
ax2.axis('off')
ax2.set_title('LBM: Local Collision + Streaming', fontsize=16, fontweight='bold', color=lbm_color, pad=20)

# Draw grid of cells with LOCAL arrows only
for i in range(grid_size):
    for j in range(grid_size):
        x = start_x + i * cell_size
        y = start_y + j * cell_size
        rect = mpatches.FancyBboxPatch((x, y), cell_size*0.8, cell_size*0.8,
                                        boxstyle="round,pad=0.05",
                                        facecolor=cell_color, edgecolor='white', linewidth=2)
        ax2.add_patch(rect)

        # Draw small circular arrows inside each cell (local operation)
        center_x = x + cell_size*0.4
        center_y = y + cell_size*0.4
        circle = mpatches.Circle((center_x, center_y), 0.25,
                                  fill=False, edgecolor=lbm_color, linewidth=2)
        ax2.add_patch(circle)
        # Small arrow on circle
        ax2.annotate('', xy=(center_x+0.18, center_y+0.18),
                    xytext=(center_x+0.25, center_y),
                    arrowprops=dict(arrowstyle='->', color=lbm_color, lw=1.5))

# Add text labels
ax2.text(5, 9.2, 'Local Operations Only', fontsize=14, ha='center', fontweight='bold', color=lbm_color)
ax2.text(5, 8.5, 'Each cell computes independently', fontsize=11, ha='center', color='#555')
ax2.text(5, 1.3, '→ Perfect for GPU parallelism', fontsize=12, ha='center', fontweight='bold', color=lbm_color)
ax2.text(5, 0.6, 'No global solver needed', fontsize=10, ha='center', color='#777')

# Add bottom comparison box
fig.text(0.5, 0.02, 'LBM: 136× speedup on GPU (Tran 2017)  |  >90% memory bandwidth utilization',
         fontsize=12, ha='center', style='italic',
         bbox=dict(boxstyle='round', facecolor='#E8F6E8', edgecolor=lbm_color))

plt.tight_layout()
plt.subplots_adjust(bottom=0.1)
plt.savefig('/home/yzk/LBMProject/docs/figures/parallelism_comparison.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("Created: parallelism_comparison.png")
