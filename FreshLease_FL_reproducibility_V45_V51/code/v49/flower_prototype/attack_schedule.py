"""Simulation-only attack schedule; never used as an access-control signal."""


def effective_training_attack(configured_attack: str, server_round: int,
                              attack_start_round: int) -> str:
    if server_round < 1 or attack_start_round < 1:
        raise ValueError("training rounds and attack start must be positive")
    return configured_attack if server_round >= attack_start_round else "none"
