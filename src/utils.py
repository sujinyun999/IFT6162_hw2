
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
def plot_learning_curves(
    rewards,
    policy_losses,
    value_losses,
    window: int = 10,
    save_path: str | Path = "learning_curves.png",
    title: str = "Learning Curves"
):
    """
    Plot episode rewards, policy loss, and value/critic loss side by side, then save.

    Args:
        rewards:       1D sequence of episode returns.
        policy_losses: 1D sequence of policy losses (same length or similar scale).
        value_losses:  1D sequence of value/critic losses.
        window:        Window size for moving-average smoothing.
        save_path:     Output file path (e.g. 'plots/learning_curves.png').
        title:         Overall figure title.
    """
    def moving_average(x, w):
        x = np.asarray(x, dtype=float)
        if w is None or w <= 1 or len(x) < w:
            return None, None
        cumsum = np.cumsum(np.insert(x, 0, 0.0))
        ma = (cumsum[w:] - cumsum[:-w]) / w
        xs = np.arange(1, len(x) + 1)[w-1:]
        return xs, ma

    rewards = np.asarray(rewards, dtype=float)
    policy_losses = np.asarray(policy_losses, dtype=float)
    value_losses = np.asarray(value_losses, dtype=float)

    ep_r = np.arange(1, len(rewards) + 1)
    ep_p = np.arange(1, len(policy_losses) + 1)
    ep_v = np.arange(1, len(value_losses) + 1)

    r_ma_x, r_ma = moving_average(rewards, window)
    p_ma_x, p_ma = moving_average(policy_losses, window)
    v_ma_x, v_ma = moving_average(value_losses, window)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Left: Rewards
    ax = axes[0]
    ax.plot(ep_r, rewards, alpha=0.4, label="Reward")
    if r_ma is not None:
        ax.plot(r_ma_x, r_ma, linewidth=2, label=f"{window}-ep MA")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title("Episode Reward")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    # Middle: Policy loss
    ax = axes[1]
    ax.plot(ep_p, policy_losses, alpha=0.4, label="Policy Loss")
    if p_ma is not None:
        ax.plot(p_ma_x, p_ma, linewidth=2, label=f"{window}-step MA")
    ax.set_xlabel("Update")
    ax.set_title("Policy Loss")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    # Right: Critic / value loss
    ax = axes[2]
    ax.plot(ep_v, value_losses, alpha=0.4, label="Value Loss")
    if v_ma is not None:
        ax.plot(v_ma_x, v_ma, linewidth=2, label=f"{window}-step MA")
    ax.set_xlabel("Update")
    ax.set_title("Critic Loss")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.0, 1, 0.95])
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Learning curves saved to: {save_path}")



def save_checkpoint(path: Path, agent, episode: int, timestep_counter: int, history: dict, best_avg_reward: float):
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "episode": episode,
        "timestep_counter": timestep_counter,
        "best_avg_reward": best_avg_reward,

        "actor_state_dict": agent.actor.state_dict(),
        "critic_state_dict": agent.critic.state_dict(),
        "optimizer_actor_state_dict": agent.optimizer_actor.state_dict(),
        "optimizer_critic_state_dict": agent.optimizer_critic.state_dict(),

        "history": history,

        # (optional but nice) reproducibility
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
    }
    torch.save(ckpt, path)
    print(f"✓ Saved checkpoint: {path} (episode={episode}, best_avg_reward={best_avg_reward:.4f})")

