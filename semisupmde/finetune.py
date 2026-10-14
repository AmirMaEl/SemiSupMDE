"""Domain-adaptation finetuning vKITTI -> KITTI (Sec. 3 of the paper).

Starting from the vKITTI-pretrained model, every step jointly optimises
  - the pairwise source loss on N vKITTI pairs (x_s1, x_s2), and
  - the perturbation consistency loss on N/r unlabelled KITTI images and two RandAugment views of each,
in a single forward pass over 2N + 3N/r images (Eq. 15). The model is evaluated on the KITTI Eigen
validation and test splits after every epoch; best_val.pth is selected on the validation split.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm

from . import config
from .data.kitti import KittiTarget
from .data.vkitti import VKittiPairs
from .evaluate import evaluate, print_results
from .losses import consistency_loss, pairwise_source_loss
from .model import load_model


def infinite(loader):
    while True:
        yield from loader


def split_predictions(preds, sizes):
    """Split the multi-scale predictions of a concatenated batch back into its parts."""
    per_scale = [torch.split(p, sizes, dim=0) for p in preds]
    return [[s[i] for s in per_scale] for i in range(len(sizes))]


def finetune(init='pretrained', out_dir=None, seed=0, **settings):
    """
    Finetune a model (a released model name or a checkpoint path) with the paper's settings
    (config.FINETUNE), any of which can be overridden as keyword arguments.

    Returns:
        The run directory with last.pth, best_val.pth, args.json and metrics.jsonl.
    """
    unknown = set(settings) - set(config.FINETUNE)
    if unknown:
        raise TypeError('unknown finetuning settings: %s' % ', '.join(sorted(unknown)))
    args = argparse.Namespace(init=init, out_dir=out_dir or config.CHECKPOINT_DIR, seed=seed,
                              **{**config.FINETUNE, **settings})
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = os.path.join(args.out_dir, 'finetune_' + time.strftime('%Y%m%d-%H%M%S'))
    os.makedirs(out_dir)
    with open(os.path.join(out_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    source = data.DataLoader(
        VKittiPairs(config.SPLITS['vkitti'], config.VKITTI_ROOT, config.IMG_SIZE, color_jitter=args.color_jitter),
        batch_size=args.n_source, shuffle=True, num_workers=6, drop_last=True, pin_memory=True, persistent_workers=True)
    target = data.DataLoader(
        KittiTarget(config.SPLITS['train'], config.KITTI_ROOT, args.target_camera, args.randaug_n, args.randaug_m,
                    config.IMG_SIZE),
        batch_size=args.n_target, shuffle=True, num_workers=8, drop_last=True, pin_memory=True, persistent_workers=True)
    source, target = infinite(source), infinite(target)

    model = load_model(args.init)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    def log_eval(epoch):
        results = {split: evaluate(model, split)[50] for split in ('val', 'test')}
        for split, m in results.items():
            print_results({50: m}, 'epoch %d %s' % (epoch, split))
        with open(os.path.join(out_dir, 'metrics.jsonl'), 'a') as f:
            f.write(json.dumps({'epoch': epoch, **results}) + '\n')
        return results

    log_eval(0)
    best = float('inf')
    model.train()
    sizes = [args.n_source, args.n_source, args.n_target, args.n_target, args.n_target]
    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(range(args.steps_per_epoch), desc='epoch %d' % epoch, mininterval=10)
        for step in pbar:
            s, t = next(source), next(target)
            batch = torch.cat([s['x1'], s['x2'], t['image'], t['aug1'], t['aug2']]).cuda(non_blocking=True)
            p1, p2, pu, pa1, pa2 = split_predictions(model(batch), sizes)
            loss_source = pairwise_source_loss(p1, p2, s['y1'].cuda(), s['y2'].cuda())
            loss_cons = consistency_loss(pu, pa1) + consistency_loss(pu, pa2)
            loss = args.source_weight * loss_source + loss_cons

            # With accumulate_grads, .grad holds the running sum of all gradients (see config.FINETUNE).
            if not args.accumulate_grads:
                optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            if step % 50 == 0:
                pbar.set_postfix(source=loss_source.item(), cons=loss_cons.item())

        results = log_eval(epoch)
        torch.save(model.state_dict(), os.path.join(out_dir, 'last.pth'))
        score = sum(results['val'][k] for k in ('abs_rel', 'sq_rel', 'rmse', 'rmse_log'))
        if score < best:
            best = score
            torch.save(model.state_dict(), os.path.join(out_dir, 'best_val.pth'))
            print('epoch %d: best on val so far, saved best_val.pth' % epoch)

    print_results(evaluate(load_model(os.path.join(out_dir, 'best_val.pth')), 'test', (50, 80)), 'best_val.pth test')
    print('checkpoints and metrics in %s' % out_dir)
    return out_dir
