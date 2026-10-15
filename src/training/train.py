import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dataset import ChestXrayDataset, DISEASES
from evaluation.metrics import calculate_metrics
from models.densenet import create_model
from training.focal_loss import FocalLoss, calculate_pos_weight


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def choose_device(value):
    if value == "dml":
        import torch_directml
        return torch_directml.device()

    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    device = torch.device(value)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    return device



def train_one(args):
    if args.max_batches is not None and args.max_batches < 1:
        raise ValueError('--max-batches must be >=1')
    set_seed(args.seed)  # reset before every fold and gamma
    device = choose_device(args.device)
    train_ds = ChestXrayDataset(mode='train', fold=args.fold, use_clahe=args.clahe)
    val_ds = ChestXrayDataset(mode='val', fold=args.fold, use_clahe=args.clahe)
    tr_patients = set(train_ds.df['Patient ID'])
    vl_patients = set(val_ds.df['Patient ID'])
    if tr_patients & vl_patients:
        raise RuntimeError('Patient leakage: overlapping train/validation patients')
    gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, generator=gen, pin_memory=device.type == 'cuda')
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=device.type == 'cuda')
    pos_weight = calculate_pos_weight(train_ds)
    model = create_model(pretrained=True).to(device)
    criterion = FocalLoss(gamma=args.gamma, pos_weight=pos_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    outdir = Path(args.output_dir) / f'gamma_{args.gamma:g}' / f'fold_{args.fold}'
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'pos_weight.json').write_text(json.dumps(dict(zip(DISEASES, pos_weight.tolist())), indent=2), encoding='utf-8')
    best_auc, stale, history = -float('inf'), 0, []
    best_path = outdir / 'best.pt'
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sum, seen = 0.0, 0
        for batch_idx, (images, targets) in enumerate(tqdm(train_loader, desc=f'gamma={args.gamma:g} fold={args.fold} epoch={epoch} train')):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_sum += float(loss.item()) * len(images)
            seen += len(images)
        scheduler.step()
        model.eval()
        truths, probs = [], []
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(tqdm(val_loader, desc='validation')):
                if args.max_batches is not None and batch_idx >= args.max_batches:
                    break
                logits = model(images.to(device))
                truths.append(targets.numpy())
                probs.append(torch.sigmoid(logits).cpu().numpy())
        # Tiny smoke tests may not include all 14 positive classes: they check
        # only end-to-end tensor/loss wiring, not performance.
        if args.max_batches is not None:
            metrics = {'mean_auroc': float('nan'), 'mean_auprc': float('nan'), 'ece': float('nan')}
        else:
            metrics = calculate_metrics(np.concatenate(truths), np.concatenate(probs))
        auc = metrics['mean_auroc']
        history.append({'epoch': epoch, 'train_loss': train_sum / seen, **{
            k: metrics[k] for k in ('mean_auroc', 'mean_auprc', 'ece')}, 'lr': optimizer.param_groups[0]['lr']})
        pd.DataFrame(history).to_csv(outdir / 'history.csv', index=False)
        print(f'epoch={epoch} loss={train_sum / seen:.5f} val_auroc={auc:.5f} val_auprc={metrics["mean_auprc"]:.5f} ece={metrics["ece"]:.5f}')
        if args.max_batches is not None:
            continue
        if auc > best_auc + args.min_delta:
            best_auc, stale = auc, 0
            torch.save({'model_state_dict': model.state_dict(), 'fold': args.fold,
                        'gamma': args.gamma, 'epoch': epoch, 'metrics': metrics,
                        'pretrained': True, 'clahe': args.clahe}, best_path)
            (outdir / 'best_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        else:
            stale += 1
            if stale >= args.patience:
                print('Early stopping')
                break
    if args.max_batches is not None:
        print('SMOKE TEST ONLY: no checkpoint or valid evaluation result saved')
        return None
    ckpt = torch.load(best_path, map_location='cpu', weights_only=True)
    m = ckpt['metrics']
    return {'gamma': args.gamma, 'fold': args.fold, 'best_epoch': ckpt['epoch'],
            'mean_auroc': m['mean_auroc'], 'mean_auprc': m['mean_auprc'], 'ece': m['ece'],
            'train_size': len(train_ds), 'val_size': len(val_ds)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--gamma', type=float, default=0)
    p.add_argument('--fold', type=int, choices=range(5), default=0)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--min-delta', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'dml'])
    p.add_argument('--clahe', action='store_true')
    p.add_argument('--max-batches', type=int, default=None, help='Only for quick wiring smoke tests')
    p.add_argument('--output-dir', default='checkpoints')
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        p.error('epochs, batch-size and patience must be positive')
    return args


if __name__ == '__main__':
    args = parse_args()
    result = train_one(args)
    if result is not None:
        print(json.dumps(result, indent=2))
