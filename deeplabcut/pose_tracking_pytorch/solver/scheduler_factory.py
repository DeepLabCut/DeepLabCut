""" Scheduler Factory
Hacked together by / Copyright 2020 Ross Wightman
"""
from .cosine_lr import CosineLRScheduler


def create_scheduler(cfg, optimizer, num_epochs=200):
    num_epochs = num_epochs

    lr_min = 0.002 * cfg["base_lr"]
    warmup_lr_init = 0.01 * cfg["base_lr"]

    warmup_t = cfg["warmup_epochs"]
    noise_range = None

    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_epochs,
        lr_min=lr_min,
        t_mul=1.0,
        decay_rate=0.1,
        warmup_lr_init=warmup_lr_init,
        warmup_t=warmup_t,
        cycle_limit=1,
        t_in_epochs=True,
        noise_range_t=noise_range,
        noise_pct=0.67,
        noise_std=1.0,
        noise_seed=42,
    )

    return lr_scheduler
