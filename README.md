# Hybrid Offline Reinforcement Learning for Robotic Manipulation: A Comparative Study of Critic-Guided and Generative Policies

| | |
| :--- | :--- |
| **Author** | ch33nchan |
| **Date** | 2025-10-05 |
| **Status** | Completed |

---

## 1. Abstract

This research investigates and implements a framework for learning complex, multi-stage robotic manipulation tasks from a static, offline dataset. The primary objective was to develop a robust pick-and-place policy for a 7-DoF robotic arm capable of generalizing from a fixed set of human demonstrations. We present a comparative analysis of two distinct offline reinforcement learning paradigms: a hybrid critic-guided behavioral cloning approach and a generative policy based on conditional diffusion models.

Our methodology first established a strong baseline with a multitask behavioral cloning (BC) policy, which leverages a phase-based curriculum to structure the learning process. When augmented with a critic-guided action selection mechanism using a separately trained IQL Q-function, this hybrid policy achieved a **96.67% success rate**. This result demonstrates that a well-engineered, pragmatic approach can yield exceptional performance by combining the stability of BC with the evaluative precision of a critic.

As a central part of our research, we explored the capabilities of Denoising Diffusion Probabilistic Models (DDPMs) for generating entire action sequences. We document the iterative refinement of these models, from simple conditioning to the implementation of advanced inference-time guidance techniques. A DDIM-based sampler achieved a **35% success rate**, validating the model's ability to generate viable long-horizon plans. Furthermore, we implemented a novel **Critic Gradient Guidance** technique, which steers the diffusion denoising process using gradients from the IQL critic. This advanced method achieved a **30% success rate**, confirming its potential but also highlighting the significant challenges of applying generative models in complex, out-of-distribution scenarios.

Our findings conclude that while generative policies represent a promising frontier in robotics, the hybrid BC-Critic model remains the superior solution for this task, offering a near-perfect balance of performance, simplicity, and training stability.

---

## 2. Key Contributions

1.  **High-Performance Hybrid Policy:** Developed a BC-IQL hybrid policy that achieves a 96.67% success rate on a complex pick-and-place task.
2.  **Comparative Analysis:** Provided a detailed comparison between a pragmatic, critic-guided BC policy and a state-of-the-art generative diffusion policy.
3.  **Diffusion Model Investigation:** Systematically documented the process of building, training, and evaluating conditional diffusion models for robotic control, including the implementation of DDIM sampling and advanced critic-guidance techniques.
4.  **Empirical Findings:** Demonstrated that for the given offline dataset, a simpler, well-structured hybrid model significantly outperforms more complex generative models, highlighting the challenges of distributional shift in offline RL.

---

## 3. Main Result: High-Performance Multitask Policy (96.67% SR)

The most effective and reliable policy developed in this project is a **Multitask Behavioral Cloning policy guided by an IQL Critic**. This approach is recommended for practical applications due to its high success rate, training efficiency, and robust performance.

### 3.1. Reproduction Steps

To reproduce the main result, execute the following commands in order.

**Step 1: Train the Multitask BC Policy**
This command trains the base policy using balanced sampling across different task phases.

```bash
python3.11 -m train.train_multitask_bc --data_root data/raw/mj_pick_place_v5 \
  --epochs 6 --sample_balance --loss_balance \
  --out_dir models/ckpts_multitask_balanced_v5
```

**Step 2: Train the IQL Critic**
This command trains the Q-network that will be used to guide the BC policy.

```bash
python3.11 -m train.train_iql_critic --data_root data/raw/mj_pick_place_v5 \
  --epochs 6 --shaped_reward --progress_reward --phase_balance \
  --out_dir models/ckpts_iql_balanced_v4
```

**Step 3: Evaluate the Final Hybrid Policy**
This command runs the BC policy and uses the critic to select the best action from a small set of candidates at each step, achieving the 96.67% success rate.

```bash
python3.11 -m scripts.eval_multitask_with_critic \
  --policy_ckpt models/ckpts_multitask_balanced_v5/multitask_policy_best.pt \
  --qnet_ckpt models/ckpts_iql_balanced_v4/qnet.pt \
  --episodes 30
```

---

## 4. Appendix: Exploratory Research on Generative Diffusion Policies

As a primary research thrust, this project investigated the application of conditional diffusion models for generating action sequences. This section documents the methodology and findings from this exploration.

### 4.1. Training the Diffusion Model

The final and most successful diffusion model (`v4`) utilizes a deep, residual UNet-style architecture, state and phase conditioning, and a cosine learning rate scheduler.

```bash
python3.11 -m train.train_diffusion_policy_v4
```

### 4.2. Evaluation via Advanced Sampling Strategies

Several inference-time strategies were implemented to generate and refine action plans from the trained diffusion model.

**Strategy 1: DDIM Sampling (35% Success Rate)**
A deterministic Denoising Diffusion Implicit Models (DDIM) sampler provided the best performance for the generative policy. By taking larger, more stable steps during the reverse process, it generated coherent action sequences that led to successful task completion in over a third of episodes.

```bash
python3.11 -m scripts.eval_diffusion_policy_v4_ddim \
  --checkpoint models/ckpts_diffusion_cond_v4/diffusion_policy_v4.pt \
  --episodes 20
```

**Strategy 2: Critic Gradient Guidance (30% Success Rate)**
This advanced technique uses gradients from the pre-trained IQL critic to steer the diffusion denoising process at every step. The gradient of the Q-function with respect to the action is used to push the generated plan towards higher-value regions of the state-action space. This method successfully produced viable trajectories, validating a powerful, state-of-the-art guidance technique.

```bash
python3.11 -m scripts.eval_diffusion_gradient_guided \
  --diff_checkpoint models/ckpts_diffusion_cond_v4/diffusion_policy_v4.pt \
  --qnet_ckpt models/ckpts_iql_balanced_v4/qnet.pt \
  --episodes 20 \
  --guidance_scale 0.1
```

### 4.3. Conclusion from Diffusion Experiments

The diffusion policy experiments successfully demonstrated long-horizon planning from a generative model trained on offline data. The 35% success rate is a strong proof-of-concept for this approach. However, the experiments also revealed the significant challenges posed by distributional shift; when the agent enters states not well-represented in the offline dataset, the guidance from the critic becomes unreliable, and the generative model struggles to recover. This exploration highlights that while generative models are a powerful tool, their successful application in offline RL requires careful consideration of state-space coverage and the limits of critic-based guidance.

---

## 5. Final Project Conclusion

This research successfully developed and validated two distinct approaches for learning robotic manipulation from offline data. The primary achievement is a hybrid BC-IQL policy that solves the task with a **96.67% success rate**, offering a robust and efficient solution. The extensive investigation into conditional diffusion models provided valuable insights into the opportunities and challenges of using generative models for robotic control, culminating in a respectable **35% success rate** using modern sampling techniques. We conclude that for many real-world applications, a well-structured hybrid of imitation learning and critic-based guidance remains a superior choice over more complex generative-only approaches.