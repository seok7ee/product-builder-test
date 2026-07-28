"""PPO configuration (rsl_rl). Section 8.3 of the plan, tuned for one RTX A6000 48 GB.

Two variants because the phases have very different costs:

* ``MasonryUpperBodyPPOCfg`` - P3, pelvis welded, ~17 active DoF, 2048 envs.
* ``MasonryWholeBodyPPOCfg`` - P5/P6, ~30 DoF plus bricks and mortar, 512 envs.
  ``num_steps_per_env`` rises to 48 so the update batch stays ~24k samples
  despite a quarter of the environments.

Network is [512, 256, 128] rather than Isaac Lab's usual [256, 128, 64]: the
observation is ~140-D and the value function has to explain balance outcomes.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class MasonryUpperBodyPPOCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 4000
    save_interval = 200
    experiment_name = "masonry_upper_body"
    empirical_normalization = True

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class MasonryWholeBodyPPOCfg(MasonryUpperBodyPPOCfg):
    num_steps_per_env = 48
    max_iterations = 8000
    experiment_name = "masonry_whole_body"
