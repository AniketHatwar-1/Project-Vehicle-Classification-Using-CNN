"""Train/evaluate a custom CNN. Augmentation, scheduler and early stopping included."""
from pathlib import Path
import argparse, csv, random, os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from model import VehicleCNN
from prepare_data import IMAGE_DIR, prepare_data

ROOT = Path(__file__).resolve().parent
MODEL_DIR, RESULTS_DIR = ROOT/"models", ROOT/"results"
IMAGE_SIZE, SEED = 128, 42
CLASSES = ["Bus", "Car", "Motorcycle", "Truck"]

def split_indices_by_class(targets, class_count, seed=SEED):
    rng = random.Random(seed)
    train, val, test = [], [], []
    for cid in range(class_count):
        indices = [i for i, y in enumerate(targets) if y == cid]
        rng.shuffle(indices)
        n = len(indices)
        n_train, n_val = int(n*0.70), int(n*0.15)
        train.extend(indices[:n_train])
        val.extend(indices[n_train:n_train+n_val])
        test.extend(indices[n_train+n_val:])
    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return train, val, test

def transforms_for_data():
    train_t = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.92, 1.08)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])
    eval_t = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])
    return train_t, eval_t

def evaluate(model, loader, criterion, device, nclasses):
    model.eval(); loss_sum = correct = total = 0
    cm = torch.zeros(nclasses, nclasses, dtype=torch.int64)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x); loss = criterion(logits, y)
            pred = logits.argmax(1)
            loss_sum += loss.item()*y.size(0); correct += (pred==y).sum().item(); total += y.size(0)
            for a, p in zip(y.cpu().tolist(), pred.cpu().tolist()): cm[a, p] += 1
    return loss_sum/max(total,1), correct/max(total,1), cm

def save_plots(history, results_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        epochs = [r["epoch"] for r in history]
        fig, ax = plt.subplots()
        ax.plot(epochs, [r["train_accuracy"]*100 for r in history], label="Train accuracy")
        ax.plot(epochs, [r["validation_accuracy"]*100 for r in history], label="Validation accuracy")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)"); ax.legend(); ax.grid(True, alpha=.3)
        fig.tight_layout(); fig.savefig(results_dir/"accuracy_curve.png", dpi=150); plt.close(fig)
        fig, ax = plt.subplots()
        ax.plot(epochs, [r["train_loss"] for r in history], label="Train loss")
        ax.plot(epochs, [r["validation_loss"] for r in history], label="Validation loss")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend(); ax.grid(True, alpha=.3)
        fig.tight_layout(); fig.savefig(results_dir/"loss_curve.png", dpi=150); plt.close(fig)
    except ImportError:
        print("matplotlib not installed; skipping training plots.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=7)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience) < 1: parser.error("arguments must be >= 1")
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    prepare_data()
    MODEL_DIR.mkdir(exist_ok=True); RESULTS_DIR.mkdir(exist_ok=True)
    train_t, eval_t = transforms_for_data()
    train_base = datasets.ImageFolder(str(IMAGE_DIR), transform=train_t)
    eval_base = datasets.ImageFolder(str(IMAGE_DIR), transform=eval_t)
    if train_base.class_to_idx != eval_base.class_to_idx or train_base.classes != CLASSES:
        raise RuntimeError(f"Expected class folders {CLASSES}; found {train_base.classes}.")
    tr_idx, va_idx, te_idx = split_indices_by_class(train_base.targets, len(CLASSES))
    train_loader = DataLoader(Subset(train_base, tr_idx), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(eval_base, va_idx), batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(Subset(eval_base, te_idx), batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VehicleCNN(len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    checkpoint_path = MODEL_DIR/"vehicle_cnn.pth"
    best_val_loss, best_val_acc, stale = float("inf"), -1.0, 0
    history = []
    print("\\nVehicle CNN (trained from scratch)")
    print(f"Device: {device} | Images: {len(tr_idx)} train, {len(va_idx)} validation, {len(te_idx)} test")
    print(f"Training for up to {args.epochs} epochs; early stopping patience={args.patience}.\\n")
    for epoch in range(1, args.epochs+1):
        model.train(); loss_sum = correct = seen = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x); loss = criterion(logits, y)
            loss.backward(); optimizer.step()
            loss_sum += loss.item()*y.size(0); correct += (logits.argmax(1)==y).sum().item(); seen += y.size(0)
        train_loss, train_acc = loss_sum/max(seen,1), correct/max(seen,1)
        val_loss, val_acc, _ = evaluate(model, val_loader, criterion, device, len(CLASSES))
        scheduler.step(val_loss)
        row = {"epoch":epoch,"train_loss":round(train_loss,5),"train_accuracy":round(train_acc,5),
               "validation_loss":round(val_loss,5),"validation_accuracy":round(val_acc,5),
               "learning_rate":optimizer.param_groups[0]["lr"]}
        history.append(row)
        print(f"Epoch {epoch:02d}/{args.epochs} | train loss {train_loss:.3f}, acc {train_acc:.1%} | val loss {val_loss:.3f}, acc {val_acc:.1%}")
        # Select best checkpoint by validation loss; break ties using validation accuracy.
        improved = val_loss < best_val_loss - 0.005 or (abs(val_loss-best_val_loss) <= 0.005 and val_acc > best_val_acc)
        if improved:
            best_val_loss, best_val_acc, stale = val_loss, val_acc, 0
            torch.save({"model_state_dict":model.state_dict(),"class_names":CLASSES,
                        "image_size":IMAGE_SIZE,"architecture":"VehicleCNN_v2"}, checkpoint_path)
        else:
            stale += 1
        if stale >= args.patience:
            print(f"Early stopping: no meaningful validation improvement for {args.patience} epochs.")
            break
    with (RESULTS_DIR/"training_history.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0])); w.writeheader(); w.writerows(history)
    save_plots(history, RESULTS_DIR)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc, cm = evaluate(model, test_loader, criterion, device, len(CLASSES))
    lines = [f"Test accuracy: {test_acc:.1%}", f"Test loss: {test_loss:.4f}",
             f"Best validation loss: {best_val_loss:.4f}", f"Best validation accuracy: {best_val_acc:.1%}", "",
             "Confusion matrix (rows = actual, columns = predicted):",
             "Actual\\\\Predicted | "+" | ".join(CLASSES)]
    for i, name in enumerate(CLASSES):
        lines.append(name+" | "+" | ".join(str(int(v)) for v in cm[i].tolist()))
    lines += ["", "Test set is held out from training. With only 100 source images per class, results can vary."]
    report = "\\n".join(lines)
    (RESULTS_DIR/"test_results.txt").write_text(report, encoding="utf-8")
    print("\\n"+report)
    print("\\nSaved model:", checkpoint_path)
    print("Saved results in:", RESULTS_DIR)

if __name__ == "__main__":
    main()
