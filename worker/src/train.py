import os
import csv
import time
import argparse

import torch
from torchvision import datasets, transforms as tv_transforms
from torch.utils.data import DataLoader
from torch import nn

from difflogic import LogicLayer, GroupSum


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--job-id", required=True)
    p.add_argument("--log-csv", default="logs/train.csv")

    p.add_argument("--gate-optimizer", default="sparsemax_noise")
    p.add_argument("--gates", type=int, default=12_000)
    p.add_argument("--network-layers", type=int, default=8)
    p.add_argument("--grouping", type=int, default=10)
    p.add_argument("--group-sum-tau", type=int, default=100)
    p.add_argument("--residual-layers", type=int, default=8)
    p.add_argument("--noise-temp", type=float, default=0.2)

    p.add_argument("--epochs", type=int, default=200)

    return p.parse_args()


def build_model(
    *,
    gates: int,
    gate_optimizer: str,
    noise_temp: float,
    grouping: int,
    group_sum_tau: int,
    residual_layers: int,
    network_layers: int,
    device: torch.device,
):
    layers = [
        nn.Flatten(),
        LogicLayer(
            9216, gates, device="cuda", implementation="cuda",
            gate_function=gate_optimizer,
            noise_temp=noise_temp,
            residual=(residual_layers > 0),
        )
    ]

    for n in range(network_layers - 1):
        use_residual = (n < residual_layers - 1)
        layers.append(
            LogicLayer(
                gates, gates, device="cuda", implementation="cuda",
                gate_function=gate_optimizer,
                noise_temp=noise_temp,
                residual=use_residual,
            )
        )

    layers.append(GroupSum(grouping, group_sum_tau))
    return nn.Sequential(*layers).to(device)


@torch.no_grad()
def eval_accuracy(model, loader, device, mode="eval"):
    orig = model.training
    model.train(mode == "train")

    total, correct = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    model.train(orig)
    return correct / max(1, total)


def main():
    args = parse_args()  # argparse CLI parsing. [web:267]

    device = torch.device("cuda")
    os.makedirs(os.path.dirname(args.log_csv), exist_ok=True)

    tfms = tv_transforms.Compose([
        tv_transforms.ToTensor(),
        tv_transforms.Lambda(lambda x: torch.cat([(x > (i + 1) / 4).float() for i in range(3)], dim=0)),
    ])

    train_ds = datasets.CIFAR10(root="../data", train=True, download=True, transform=tfms)  # [web:384]
    test_ds  = datasets.CIFAR10(root="../data", train=False, download=True, transform=tfms)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    fieldnames = [
        "job_id", "epoch",
        "train_loss", "train_acc",
        "float_eval_acc", "discrete_eval_acc",
        "gate_optimizer", "gates", "network_layers",
        "grouping", "group_sum_tau",
        "residual_layers", "noise_temp",
        "epochs",
        "time",
    ]

    header_needed = (not os.path.exists(args.log_csv)) or (os.path.getsize(args.log_csv) == 0)
    with open(args.log_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if header_needed:
            w.writeheader()

    model = build_model(
        gates=args.gates,
        gate_optimizer=args.gate_optimizer,
        noise_temp=args.noise_temp,
        grouping=args.grouping,
        group_sum_tau=args.group_sum_tau,
        residual_layers=args.residual_layers,
        network_layers=args.network_layers,
        device=device,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print(
        f"JOB_START job_id={args.job_id} "
        f"epochs={args.epochs} layers={args.network_layers} gates={args.gates}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        seen = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            seen += x.size(0)

        train_loss = running_loss / max(1, seen)
        train_acc = correct / max(1, seen)

        disc_eval_acc = eval_accuracy(model, test_loader, device=device, mode="eval")
        float_eval_acc = eval_accuracy(model, test_loader, device=device, mode="train")

        # Per-epoch summary line (easy for executor to parse).
        print(
            f"EPOCH {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"float_eval_acc={float_eval_acc:.4f} discrete_eval_acc={disc_eval_acc:.4f}",
            flush=True,
        )

        with open(args.log_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writerow({
                "job_id": args.job_id,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "float_eval_acc": float_eval_acc,
                "discrete_eval_acc": disc_eval_acc,
                "gate_optimizer": args.gate_optimizer,
                "gates": args.gates,
                "network_layers": args.network_layers,
                "grouping": args.grouping,
                "group_sum_tau": args.group_sum_tau,
                "residual_layers": args.residual_layers,
                "noise_temp": args.noise_temp,
                "epochs": args.epochs,
                "time": time.time(),
            })

    print(f"JOB_END job_id={args.job_id}", flush=True)


if __name__ == "__main__":
    main()
