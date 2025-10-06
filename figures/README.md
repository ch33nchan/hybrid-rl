# Report Figures - Hybrid Offline RL for Robotic Pick-and-Place

This directory contains all figures generated for the experiment report.

## Generated Figures

### 1. **phase_distribution.png**
- **Description**: Bar chart showing the distribution of training samples across the 6 task phases
- **Purpose**: Demonstrates the imbalanced nature of the dataset and justifies the need for phase-balanced sampling
- **Key Insight**: Shows which phases are underrepresented (GRASP, FINE) vs overrepresented (APPROACH, MOVE)

### 2. **trajectory_evolution.png**
- **Description**: Multi-panel plot showing state evolution over time for sample trajectories
- **Panels**:
  - End-effector Z-height
  - Cube Z-height (with success threshold line)
  - Gripper state (open/closed)
  - Distance to target (with success radius line)
- **Purpose**: Visualizes the temporal dynamics of successful pick-and-place trajectories
- **Key Insight**: Shows the characteristic pattern of descend → grasp → lift → move → place

### 3. **action_distributions.png**
- **Description**: Histograms of action values for each of the 4 action dimensions
- **Dimensions**: X-axis, Y-axis, Z-axis, Gripper
- **Purpose**: Shows the distribution of actions in the dataset
- **Key Insight**: Reveals action biases and typical movement patterns

### 4. **phase_action_heatmap.png**
- **Description**: Heatmap showing mean absolute action magnitude for each phase-action combination
- **Purpose**: Identifies which actions are most active in each phase
- **Key Insight**: 
  - APPROACH: High XY movement, low Z
  - DESCEND: High Z movement (negative)
  - GRASP: Gripper activation
  - LIFT: High Z movement (positive)
  - MOVE/FINE: Balanced XY for positioning

### 5. **environment_screenshots.png**
- **Description**: Visual snapshots of the MuJoCo environment at different phases
- **Phases shown**: Initial, Approach, Descend, Grasp, Lift
- **Purpose**: Provides visual context for the task and environment
- **Key Insight**: Shows the 7-DoF arm, red cube, and green target placement

### 6. **success_rate_comparison.png**
- **Description**: Bar chart comparing success rates across different methods
- **Methods**:
  - BC Baseline: ~85%
  - **BC + IQL Critic: 96.67%** (highlighted as best)
  - Diffusion (DDIM): 35%
  - Diffusion (Critic-Guided): 30%
- **Purpose**: Demonstrates the superiority of the hybrid BC-IQL approach
- **Key Insight**: Simple, well-structured methods outperform complex generative models in offline RL

### 7. **architecture_diagram.png**
- **Description**: Schematic diagram of the hybrid BC-IQL architecture
- **Components**:
  - State input (9D)
  - Policy trunk (256-dim MLP)
  - Dual heads (Action + Phase)
  - IQL Critic (Twin Q-networks)
  - Critic-guided action selection
- **Purpose**: Visual explanation of the system architecture
- **Key Insight**: Shows how BC policy and IQL critic work together

## Usage in Report

### Recommended Figure Placement

1. **Introduction/Background**: 
   - `environment_screenshots.png` - Show the task setup

2. **Dataset Section**:
   - `phase_distribution.png` - Illustrate dataset composition
   - `trajectory_evolution.png` - Show example trajectories

3. **Methodology Section**:
   - `architecture_diagram.png` - Explain the approach
   - `phase_action_heatmap.png` - Show phase-specific behaviors

4. **Results Section**:
   - `success_rate_comparison.png` - Main results
   - `action_distributions.png` - Additional analysis

### Figure Captions (Suggested)

**Figure 1**: Phase distribution in the offline dataset (N=38,065 samples). The dataset exhibits natural imbalance with overrepresentation of APPROACH and MOVE phases, and underrepresentation of GRASP and FINE phases. This motivated our phase-balanced sampling strategy during training.

**Figure 2**: State evolution across sample trajectories showing the characteristic pick-and-place pattern. The cube is lifted above the success threshold (0.105m) and placed within the target radius (0.055m).

**Figure 3**: Distribution of action values across the dataset for each action dimension. The distributions reveal the control patterns used in human demonstrations.

**Figure 4**: Mean absolute action magnitude by phase. Each phase exhibits distinct action patterns: APPROACH uses primarily XY movement, DESCEND uses Z-axis descent, GRASP activates the gripper, LIFT uses upward Z movement, and MOVE/FINE use balanced XY positioning.

**Figure 5**: MuJoCo environment visualization showing the 7-DoF robotic arm, red cube (object to manipulate), and green target (goal location) at different phases of task execution.

**Figure 6**: Performance comparison across methods. The hybrid BC-IQL approach achieves 96.67% success rate, significantly outperforming both the BC baseline (85%) and generative diffusion policies (30-35%). This demonstrates the effectiveness of critic-guided action selection in offline RL.

**Figure 7**: Architecture of the hybrid BC-IQL system. The multitask policy outputs both actions and phase predictions, while the IQL critic evaluates action candidates. During inference, the critic selects the highest-value action from a phase-adaptive candidate set.

## Regenerating Figures

To regenerate all figures:

```bash
source .venv/bin/activate
python3.11 scripts/generate_report_figures.py \
  --data_root data/raw/mj_pick_place_v5 \
  --output_dir figures \
  --num_trajectories 5 \
  --max_samples 10000
```

## Technical Details

- **Resolution**: 300 DPI (publication quality)
- **Format**: PNG with transparency support
- **Style**: Seaborn darkgrid theme
- **Color scheme**: Colorblind-friendly palette
- **Font**: Bold labels for readability

---

Generated: 2025-10-05
Dataset: mj_pick_place_v5 (38,065 samples, 813 episodes)
