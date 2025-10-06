# Paper Compilation Guide

## 📄 Paper: Pragmatism vs. Power

**Title:** Pragmatism vs. Power: A Comparative Study of Critic-Guided Behavioral Cloning and Diffusion Policies for Offline Robotic Manipulation

**Main Result:** 96.67% Success Rate with BC-IQL Hybrid

---

## ✅ What's Been Updated

### 1. **Proper Figure Integration**
All generated figures are now properly referenced:
- `environment_screenshots.png` - Task visualization
- `phase_distribution.png` - Dataset composition
- `architecture_diagram.png` - System architecture
- `phase_action_heatmap.png` - Phase-specific behaviors
- `success_rate_comparison.png` - Main results chart
- `rollouts/rollout_comparison.png` - Successful episodes
- `trajectory_evolution.png` - State evolution (full-width figure)

### 2. **Accurate Dataset Information**
- Updated to 6 phases (not 4)
- Correct sample count: 38,065 transitions
- Correct state/action dimensions: 9D state, 4D action
- Phase names: APPROACH, DESCEND, GRASP_SETTLE, LIFT, MOVE, FINE

### 3. **Enhanced Methodology**
- Added phase-balanced sampling details
- Explained phase-adaptive candidate generation
- Described twin Q-network usage
- Clarified multitask learning with phase classification

### 4. **Results Tables**
- Integrated LaTeX tables from `figures/tables/`
- Added ablation study table
- Quantified component contributions

### 5. **Improved Context**
- Added specific performance metrics (29/30 episodes, 138 avg steps)
- Quantified improvements (+11.67% over BC, +61.67% over diffusion)
- Enhanced qualitative analysis with rollout visualizations

---

## 🔧 How to Compile

### Option 1: Using the Script (Recommended)
```bash
./compile_paper.sh
```

### Option 2: Manual Compilation
```bash
pdflatex paper.tex
pdflatex paper.tex  # Run twice for references
```

### Option 3: Using Overleaf
1. Upload `paper.tex` to Overleaf
2. Upload the entire `figures/` directory
3. Compile (should work automatically)

---

## 📊 Figures Used in Paper

| Figure | File | Location |
|--------|------|----------|
| Fig. 1 | Environment | `figures/environment_screenshots.png` |
| Fig. 2 | Phase Distribution | `figures/phase_distribution.png` |
| Fig. 3 | Architecture | `figures/architecture_diagram.png` |
| Fig. 4 | Phase-Action Heatmap | `figures/phase_action_heatmap.png` |
| Fig. 5 | Main Results | `figures/success_rate_comparison.png` |
| Fig. 6 | Rollout Comparison | `figures/rollouts/rollout_comparison.png` |
| Fig. 7 | Trajectory Evolution | `figures/trajectory_evolution.png` |
| Table 1 | Main Results | `figures/tables/main_results.tex` |
| Table 2 | Ablation Study | `figures/tables/ablation_study.tex` |

---

## 📝 Key Sections

### Abstract
- Clearly states 96.67% vs 35% comparison
- Highlights pragmatism vs power theme

### Introduction
- Motivates offline RL
- Introduces two approaches
- Lists contributions

### Methodology
- **Section 3.1:** Environment & Dataset (with phase details)
- **Section 3.2:** BC-IQL Hybrid (with phase balancing)
- **Section 3.3:** Diffusion Policy (with critic guidance)

### Results
- **Section 4.2:** Quantitative Results (with tables)
- **Section 4.3:** Ablation Study (component contributions)
- **Section 4.4:** Qualitative Analysis (rollout visualizations)

### Conclusion
- Emphasizes "pragmatism outperforms power"
- Quantifies improvements
- Suggests future directions

---

## 🎯 Presentation Tips

### For Slides
1. **Slide 1:** Title + 96.67% headline
2. **Slide 2:** Environment (Fig. 1)
3. **Slide 3:** Challenge - Phase distribution (Fig. 2)
4. **Slide 4:** Solution - Architecture (Fig. 3)
5. **Slide 5:** Results - Success rates (Fig. 5)
6. **Slide 6:** Demo - Rollouts (Fig. 6)
7. **Slide 7:** Analysis - Ablations (Table 2)
8. **Slide 8:** Conclusion

### Key Talking Points
- **96.67% success rate** - Near-perfect performance
- **Phase-based decomposition** - Natural task structure
- **Critic-guided selection** - Smart action refinement
- **Pragmatism wins** - Simple methods outperform complex ones
- **Each component matters** - Validated by ablations

---

## 🐛 Troubleshooting

### Missing Figures Error
If you get "file not found" errors:
```bash
# Check graphics path
grep graphicspath paper.tex
# Should show: \graphicspath{{figures/}}

# Verify figures exist
ls figures/*.png
ls figures/rollouts/*.png
ls figures/tables/*.tex
```

### Table Compilation Errors
If tables don't compile:
```bash
# Check table files exist
ls figures/tables/main_results.tex
ls figures/tables/ablation_study.tex

# Verify they have proper LaTeX syntax
cat figures/tables/main_results.tex
```

### LaTeX Not Found
Install MacTeX:
```bash
# Using Homebrew
brew install --cask mactex

# Or download from: https://www.tug.org/mactex/
```

---

## 📦 Files Needed for Compilation

### Required Files
- `paper.tex` - Main LaTeX file
- `figures/` directory with all PNG files
- `figures/tables/` directory with TEX files

### Optional Files
- `references.bib` - If using BibTeX (currently using inline bibliography)

---

## ✨ Final Checklist

- [x] All figures properly referenced
- [x] Correct dataset statistics (38,065 samples, 6 phases)
- [x] Accurate results (96.67%, 35%, 30%)
- [x] Tables integrated
- [x] Ablation study included
- [x] Qualitative analysis with visualizations
- [x] Graphics path configured
- [x] All packages included

---

## 🎓 Citation-Ready Abstract

> We present a comparative study of critic-guided behavioral cloning and diffusion policies for offline robotic manipulation. Our hybrid BC-IQL approach achieves 96.67% success rate on a complex pick-and-place task, significantly outperforming both BC baselines (+11.67%) and state-of-the-art diffusion policies (+61.67%). Through systematic ablation studies, we demonstrate that pragmatic combinations of simple, stable methods can outperform complex generative models in data-constrained offline RL scenarios.

---

**Ready for submission! 🚀**

For questions, see `figures/INDEX.md` for complete documentation.
