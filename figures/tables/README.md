# Results Tables for Report

This directory contains formatted tables for inclusion in the research report.

## Files

- `main_results.md` / `main_results.tex` - Main performance comparison
- `phase_statistics.md` - Dataset phase distribution
- `ablation_study.md` / `ablation_study.tex` - Component ablation study

## Key Findings

1. **BC + IQL Critic achieves 96.67% success rate** - Best performance
2. **Diffusion policies achieve 30-35%** - Promising but limited by distributional shift
3. **Phase balancing contributes +8.4%** - Critical for handling imbalanced data
4. **Critic guidance contributes +11.7%** - Significant improvement over BC alone
5. **Twin Q-networks contribute +5.5%** - Conservative estimation helps

## Usage

### Markdown Tables
Copy-paste directly into Markdown documents or convert to other formats.

### LaTeX Tables
Include in your LaTeX document with:
```latex
\input{tables/main_results.tex}
```

Make sure to include these packages in your preamble:
```latex
\usepackage{booktabs}
\usepackage{multirow}
```
