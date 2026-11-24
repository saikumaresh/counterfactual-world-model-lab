# Counterfactual world models: a 20-second replication

A small, runnable replication of **residual reuse** from *Counterfactual World Models: JEPAs That Reason About "What If"* by Randall Balestriero, Yann LeCun and Kerda Varaku ([arXiv:2611.24778](https://arxiv.org/abs/2611.24778), November 2026).

The setup is the paper's own example: a boat picks a heading while an unseen current pushes it. A world model learns only from what actually happened. Then we ask: what if the boat had picked a different heading, with the same current?

| Answer | How |
|---|---|
| Naive | Ask the world model about the other heading: `mu(z, a')` |
| Residual reuse | Keep what actually happened, add the difference: `z' + mu(z, a') - mu(z, a)` |
| Truth | Rerun the simulator with the same hidden current (only a lab can do this) |

## Results (`results.json`)

R² of the counterfactual position change against the truth.

| Experiment | Naive | Residual reuse |
|---|---|---|
| E1. Boat and current, 16k factual transitions | 53.8% | 99.4% |
| E2. Naive answer with 1k, 4k, 16k, 64k transitions | 53.3%, 54.7%, 52.9%, 54.8% | 97.8% to 99.5% |
| E3a. Worn motor (disturbance scales the command) | 93.3% | 82.7% (worse than naive) |
| E3b. Harbour wall, events that touch the wall | 37.9% | 85.7% |
| E4. Step length 0.25 s to 4 s | 67.5% to 50.8% | 99.1% to 93.6% |

More factual data never fixes the naive answer. Keeping the observed surprise does, as long as the action and the disturbance are separate forces over a short step. When the disturbance multiplies the command, or the world jumps at a contact, the trick weakens or fails, as the paper predicts.

## Run it

```
pip install -r requirements.txt
python lab.py
```

This is a toy. The paper works from pixels with frozen image encoders (DINOv2, MAE and others) and JEPA predictors. The authors' code: https://github.com/galilai-group/counterfactual-WM

MIT licence. Companion article by Sai Kumaresh on Medium.
