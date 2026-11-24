"""Counterfactual world models: a small, runnable replication of "residual reuse".

Paper: Balestriero, LeCun, Varaku. Counterfactual World Models: JEPAs That Reason
About "What If". arXiv:2611.24778 (Nov 2026).

The setup is the paper's own example: a boat picks a heading while an unseen
current pushes it. We learn a world model only from factual transitions, then ask
"what if the boat had picked a different heading, with the same current?"

Three ways to answer:
  naive:          ask the world model about the other heading      mu(z, a')
  residual reuse: keep what actually happened, add the difference   z' + mu(z, a') - mu(z, a)
  truth:          rerun the simulator with the same hidden current  (only a lab can do this)

Run:  python lab.py            (numpy + scikit-learn, about 20 seconds)
"""
import json
import time

import numpy as np
from sklearn.neural_network import MLPRegressor

RNG = np.random.default_rng(7)
H = 0.5          # seconds per transition
SIGMA = 1.5      # spread of the hidden current


def heading(theta):
    return np.stack([np.cos(theta), np.sin(theta)], axis=-1)


# --- worlds -----------------------------------------------------------------
# Every world gets a state z (position, velocity), a heading a, and a hidden
# disturbance d drawn fresh for each event. The model never sees d.

def boat_current(z, a, d, h=H):
    """Thrust and current are separate forces. Water drag grows with speed
    squared, so over a long step the two forces start to interact."""
    p, v = z[:, :2].copy(), z[:, 2:].copy()
    steps = max(1, int(round(h / 0.01)))
    dt = h / steps
    for _ in range(steps):
        speed = np.linalg.norm(v, axis=1, keepdims=True)
        acc = 2.0 * heading(a) + d - 0.8 * speed * v   # thrust + current - drag
        p = p + dt * v
        v = v + dt * acc
    return np.hstack([p, v])


def worn_motor(z, a, d, h=H):
    """The disturbance scales the command: a motor at 40 to 100 percent power."""
    p, v = z[:, :2], z[:, 2:]
    k = 0.7 + 0.3 * np.tanh(d[:, :1])               # hidden efficiency
    acc = 2.0 * k * heading(a) - 0.8 * v
    v2 = v + h * acc
    p2 = p + h * v + 0.5 * h * h * acc
    return np.hstack([p2, v2])


def harbour_wall(z, a, d, h=H):
    """Same boat, but a wall at x = 0.15 stops it dead. A contact: nature leaps."""
    z2 = boat_current(z, a, d, h)
    hit = z2[:, 0] > 0.15
    z2[hit, 0] = 0.15
    z2[hit, 2] = 0.0
    return z2


WORLDS = {"boat_current": boat_current, "worn_motor": worn_motor, "harbour_wall": harbour_wall}


# --- data -------------------------------------------------------------------

def sample(n, world, h=H):
    z = np.hstack([RNG.uniform(-0.05, 0.05, (n, 2)), RNG.normal(0, 0.2, (n, 2))])
    a = RNG.uniform(-np.pi, np.pi, n)
    d = RNG.normal(0, SIGMA, (n, 2))
    return z, a, d, world(z, a, d, h)


def feats(z, a):
    return np.hstack([z, heading(a)])


def fit_world_model(n, world, h=H):
    """Learn mu(z, a) = E[z' | z, a] from factual transitions only."""
    z, a, _, z2 = sample(n, world, h)
    m = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=600, random_state=0,
                     early_stopping=True, n_iter_no_change=20)
    m.fit(feats(z, a), (z2 - z) / h)                # predict the change, scaled
    return lambda zz, aa: zz + h * m.predict(feats(zz, aa))


def r2(pred, truth):
    """Variance explained in the counterfactual displacement (position only)."""
    err = ((pred - truth) ** 2).sum()
    var = ((truth - truth.mean(axis=0)) ** 2).sum()
    return 1 - err / var


def counterfactual_scores(mu, world, n=4000, h=H):
    z, a, d, z_obs = sample(n, world, h)
    a_alt = a + RNG.uniform(np.pi / 4, np.pi, n) * RNG.choice([-1, 1], n)
    truth = world(z, a_alt, d, h)[:, :2] - z[:, :2]
    naive = mu(z, a_alt)[:, :2] - z[:, :2]
    reuse = (z_obs + mu(z, a_alt) - mu(z, a))[:, :2] - z[:, :2]
    out = {"naive": round(float(r2(naive, truth)), 3),
           "residual_reuse": round(float(r2(reuse, truth)), 3)}
    if world is harbour_wall:
        # events where either version of the day touched the wall
        touch = (z_obs[:, 0] >= 0.15) | (truth[:, 0] + z[:, 0] >= 0.15)
        out["contact_events"] = int(touch.sum())
        out["contact_naive"] = round(float(r2(naive[touch], truth[touch])), 3)
        out["contact_residual_reuse"] = round(float(r2(reuse[touch], truth[touch])), 3)
    return out


def main():
    t0 = time.time()
    out = {"h_seconds": H, "current_sigma": SIGMA}

    # E1: the main result, smooth physics
    mu = fit_world_model(16000, boat_current)
    out["E1_boat_current"] = counterfactual_scores(mu, boat_current)

    # E2: more factual data does not fix the naive answer
    out["E2_more_data"] = {}
    for n in (1000, 4000, 16000, 64000):
        m = fit_world_model(n, boat_current)
        out["E2_more_data"][n] = counterfactual_scores(m, boat_current)

    # E3: where it breaks
    for name in ("worn_motor", "harbour_wall"):
        m = fit_world_model(16000, WORLDS[name])
        out["E3_" + name] = counterfactual_scores(m, WORLDS[name])

    # E4: "short" matters. Longer steps let the forces interact more.
    out["E4_step_length"] = {}
    for h in (0.25, 0.5, 1.0, 2.0, 4.0):
        m = fit_world_model(16000, boat_current, h)
        out["E4_step_length"][h] = counterfactual_scores(m, boat_current, h=h)

    out["runtime_s"] = round(time.time() - t0, 1)
    print(json.dumps(out, indent=2))
    with open("results.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
