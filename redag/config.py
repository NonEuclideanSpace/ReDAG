from dataclasses import dataclass

@dataclass
class ReDAGConfig:
    gate_tau: float = 0.5
    node_threshold: float = 0.7
    min_target_gain: float = 0.05
    max_protected_drop: float = 0.08
    feather_px: int = 8
    alpha_mask: float = 3.0
    lr_sft_planner: float = 0.0001
    lr_sft_controller: float = 0.0001
    lr_sft_editor: float = 5e-05
    protection_lambda: float = 1.5
    step_cost: float = 0.01
    delta_noise: float = 0.015
    reward_scale_s0: float = 0.08
    adv_clip_max: float = 2.0
    group_size_ctrl: int = 6
    clip_eps_ctrl: float = 0.2
    beta_kl_ctrl: float = 0.02
    lr_rl_controller: float = 2e-05
    group_size_edit: int = 6
    num_denoise_steps: int = 50
    sde_eta: float = 0.35
    cfg_scale: float = 3.5
    num_strata: int = 4
    clip_eps_edit: float = 0.2
    gamma_anchor: float = 0.001
    lr_rl_editor: float = 1e-05
    ppo_epochs: int = 2
    target_kl_ctrl: float = 0.02
    target_kl_edit: float = 0.02

    def __post_init__(self):
        import math
        from dataclasses import fields
        if any(not math.isfinite(getattr(self, f.name)) for f in fields(self)):
            raise ValueError('Configuration values must be finite')
        if any(not 0 <= getattr(self, name) <= 1 for name in ('gate_tau', 'node_threshold', 'min_target_gain', 'max_protected_drop')):
            raise ValueError('Probability thresholds must be in [0, 1]')
        if self.group_size_ctrl < 2 or self.group_size_edit < 2 or self.num_strata != 4 or self.num_denoise_steps < 5:
            raise ValueError('Invalid group or replay sizes')
        if self.reward_scale_s0 <= 0 or self.sde_eta <= 0 or self.adv_clip_max <= 0 or self.delta_noise < 0:
            raise ValueError('Invalid reward or diffusion scales')
        if type(self.ppo_epochs) is not int or self.ppo_epochs < 1 or min(self.target_kl_ctrl, self.target_kl_edit) <= 0:
            raise ValueError('Invalid PPO passes or KL limits')
        if self.feather_px < 0 or self.step_cost < 0 or self.protection_lambda < 0:
            raise ValueError('Costs and feather width cannot be negative')
