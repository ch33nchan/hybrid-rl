# Quick Reference: Report Figures & Results

## 🎯 Main Result (Headline)
**96.67% Success Rate** - Hybrid BC-IQL Critic approach on robotic pick-and-place task

## 📊 All Generated Assets (19 files, 4.2 MB)

### Core Visualizations (7 figures)
1. `phase_distribution.png` - Dataset composition
2. `trajectory_evolution.png` - State dynamics
3. `action_distributions.png` - Action patterns
4. `phase_action_heatmap.png` - Phase-specific behaviors
5. `environment_screenshots.png` - Task visualization
6. `success_rate_comparison.png` - **MAIN RESULT CHART**
7. `architecture_diagram.png` - System design

### Rollout Examples (4 figures)
8. `rollouts/rollout_ep1.png` - Detailed success (25 steps)
9. `rollouts/rollout_ep2.png` - Detailed success (30 steps)
10. `rollouts/rollout_ep3.png` - Detailed success (35 steps)
11. `rollouts/rollout_comparison.png` - Side-by-side grid

### Tables (8 files)
12-19. Markdown & LaTeX tables for results, ablations, statistics

## 🏆 Performance Summary

| Approach | Success Rate | Notes |
|----------|--------------|-------|
| BC Baseline | 85.0% | Simple imitation learning |
| **BC + IQL Critic** | **96.67%** | **Our method (BEST)** |
| Diffusion DDIM | 35.0% | Generative approach |
| Diffusion + Guidance | 30.0% | Advanced technique |

## 🔬 Key Contributions

1. ✅ **High-performance hybrid policy** (96.67% SR)
2. ✅ **Phase-based task decomposition** (6 phases)
3. ✅ **Critic-guided action selection** (adaptive candidates)
4. ✅ **Comparative analysis** (BC vs Diffusion)
5. ✅ **Ablation studies** (each component validated)

## 📐 Technical Specifications

- **Environment**: MuJoCo pick-and-place, 7-DoF arm
- **Dataset**: 38,065 samples, 813 episodes
- **State**: 9D (EEF pos, gripper, cube pos, target)
- **Action**: 4D continuous (XY, Z, gripper)
- **Phases**: APPROACH → DESCEND → GRASP → LIFT → MOVE → FINE

## 💡 Figure Selection Guide

### For a 6-page paper:
**Essential (4 figures):**
1. `success_rate_comparison.png` - Main result
2. `architecture_diagram.png` - Method
3. `phase_distribution.png` - Dataset
4. `rollout_comparison.png` - Qualitative results

**Optional (2-3 more):**
5. `trajectory_evolution.png` - Detailed analysis
6. `phase_action_heatmap.png` - Behavioral insights
7. `environment_screenshots.png` - Task visualization

### For a poster:
- Use `success_rate_comparison.png` prominently
- Include `rollout_comparison.png` for visual impact
- Add `architecture_diagram.png` for method explanation
- Use tables for detailed numbers

### For slides:
- Start with `environment_screenshots.png` (task intro)
- Show `phase_distribution.png` (challenge)
- Present `architecture_diagram.png` (solution)
- Reveal `success_rate_comparison.png` (results)
- End with `rollout_comparison.png` (demo)

## 📝 LaTeX Integration

### In your preamble:
```latex
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{subcaption}
```

### Include figures:
```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.8\linewidth]{figures/success_rate_comparison.png}
  \caption{Performance comparison across methods.}
  \label{fig:main_results}
\end{figure}
```

### Include tables:
```latex
\input{figures/tables/main_results.tex}
```

## 🎨 Color Scheme (Consistent across all figures)

- **Phase 0 (APPROACH)**: Blue `#1f77b4`
- **Phase 1 (DESCEND)**: Orange `#ff7f0e`
- **Phase 2 (GRASP)**: Green `#2ca02c`
- **Phase 3 (LIFT)**: Red `#d62728`
- **Phase 4 (MOVE)**: Purple `#9467bd`
- **Phase 5 (FINE)**: Brown `#8c564b`

## 📧 Citation-Ready Results

> We developed a hybrid BC-IQL approach that achieves a 96.67% success rate on a 
> robotic pick-and-place task, significantly outperforming both a BC baseline (85%) 
> and state-of-the-art diffusion policies (30-35%). Our method combines the stability 
> of behavioral cloning with the evaluative precision of an IQL critic, using 
> phase-adaptive action candidate generation.

## ⚡ Quick Stats

- **Training time**: ~30 minutes total (BC + Critic)
- **Inference**: Real-time (fast)
- **Dataset size**: 38K samples
- **Success improvement**: +11.67% over BC baseline
- **Rollout length**: ~30 steps average
- **Phase coverage**: All 6 phases represented

---

**All assets ready for your report! 🚀**

Location: `/Users/cheencheen/Desktop/pi/figures/`
