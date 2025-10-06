# Main Results Comparison

| Method | Success Rate | Avg Steps | Training Time | Inference Speed | Notes |
| --- | --- | --- | --- | --- | --- |
| BC Baseline | 85.0 | 142 | ~15 min | Fast | Simple, stable baseline |
| BC + IQL Critic (Ours) | 96.67 | 138 | ~30 min | Fast | Best overall performance |
| Diffusion Policy (DDIM) | 35.0 | 160 | ~2 hours | Slow | Generative, struggles with OOD |
| Diffusion + Critic Guidance | 30.0 | 160 | ~2 hours | Very Slow | Advanced guidance technique |