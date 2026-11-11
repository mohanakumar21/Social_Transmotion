import logging
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

from model_jta_pooled import create_model

device = "cuda:0"


def make_cfg(model_type, seq_len):
    return {
        "DEVICE": device,
        "TRAIN": {"input_track_size": 9, "output_track_size": 12},
        "MODEL": {"type": model_type, "seq_len": seq_len, "token_num": 47, "dim_hidden": 128,
                  "num_heads": 4, "num_layers_local": 6, "num_layers_global": 3,
                  "dim_feedforward": 1024, "output_scale": 1},
    }


logger = logging.getLogger("bench")
variants = [("baseline (435 local tokens)", make_cfg("transmotion", 435)),
            ("pooled (30 local tokens)", make_cfg("transmotion_pooled", 30))]

B, N, in_F, J, K = 4, 11, 9, 47, 4  # N = 11 is about the mean number of agents per scene
x = torch.randn(B, in_F, N * J, K, device=device)
pad = torch.zeros(B, N, device=device)

print(f"{'model':30s} {'params':>10s} {'GFLOPs fwd':>11s} {'infer ms':>9s} {'train ms':>9s} {'peak MB':>8s}")
for name, cfg in variants:
    model = create_model(cfg, logger)
    params = sum(p.numel() for p in model.parameters())

    model.eval()
    with torch.no_grad():
        with FlopCounterMode(display=False) as fc:
            model(x, pad)
        gflops = fc.get_total_flops() / 1e9
        for _ in range(5):
            model(x, pad)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(30):
            model(x, pad)
        torch.cuda.synchronize()
        infer_ms = (time.time() - t) / 30 * 1000

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    torch.cuda.reset_peak_memory_stats()
    for i in range(25):
        if i == 5:
            torch.cuda.synchronize()
            t = time.time()
        opt.zero_grad()
        model(x, pad).sum().backward()
        opt.step()
    torch.cuda.synchronize()
    train_ms = (time.time() - t) / 20 * 1000
    peak_mb = torch.cuda.max_memory_allocated() / 1e6

    print(f"{name:30s} {params:10,d} {gflops:11.2f} {infer_ms:9.1f} {train_ms:9.1f} {peak_mb:8.0f}")
    del model, opt
    torch.cuda.empty_cache()