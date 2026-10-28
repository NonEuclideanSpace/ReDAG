from pathlib import Path

import yaml


def load_config(path):
    with Path(path).open() as stream:
        config = yaml.safe_load(stream)
    return validate_config(config)


def validate_config(config):
    if tuple(config["canvas"]) != (512, 512):
        raise ValueError("A different canvas requires a separate resource profile")
    rl = config["rl"]
    if rl["tasks"] < 1 or min(rl["roots"], rl["suffixes"]) < 2:
        raise ValueError("Hierarchical RLOO requires P >= 1, N >= 2, M >= 2")
    for mode in ("agent", "joint", "renderer"):
        target = rl.get(f"{mode}_effective_updates")
        maximum = rl.get(f"{mode}_max_rounds")
        if target is None and maximum is not None:
            raise ValueError("An RL maximum round override requires an effective update target")
        if target is not None:
            default = rl["agent_rounds" if mode == "agent" else "joint_rounds"]
            maximum = default if maximum is None else maximum
            if type(target) is not int or type(maximum) is not int or target < 1 or maximum < target:
                raise ValueError("RL effective targets and bounded maximum rounds must be positive ordered integers")
    if config["renderer"]["steps"] < config["renderer"]["replay_strata"]:
        raise ValueError("Replay strata cannot exceed diffusion steps")
    if config["renderer"]["replay_strata"] != 4:
        raise ValueError("The main training profile uses four time strata")
    if not 0 < config["renderer"]["eta"] <= 1 or config["renderer"]["cfg_scale"] < 1:
        raise ValueError("The training sampler requires positive stochasticity and CFG >= 1")
    for field in ("max_input_tokens", "max_new_tokens", "head_chunk_size", "max_images"):
        value = config["actor"][field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"Actor {field} must be a positive integer")
    if config["actor"]["max_images"] > 4:
        raise ValueError("The controller supports at most four public images per action")
    for budget in config["episode"].values():
        if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
            raise ValueError("Episode budgets must be positive integers")
    return config
