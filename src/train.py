import argparse
from src.dataset import HierTextDataModule
from src.model import HierTextModelModule
from src.trainer import HierTextTrainer
from src.callback import EarlyStoppingCallback

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_name", type=str)
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--label_dir", type=str)
    parser.add_argument("--image_size", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--strategy", type=str, help="single_device or ddp")
    parser.add_argument("--devices", nargs="+", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--ckpt_path", type=str)
    args = parser.parse_args()

    dm = HierTextDataModule(
        args.dataset_dir,
        args.label_dir,
        args.image_size,
        args.batch_size,
        args.num_workers,
    )
    mm = HierTextModelModule(args.encoder_name)
    early_stopping_cb = EarlyStoppingCallback(
        patience=5, monitor="val_loss", mode="min"
    )
    trainer = HierTextTrainer(
        strategy=args.strategy,
        devices=args.devices,
        callbacks=[early_stopping_cb],
        epochs=args.epochs,
        amp=args.amp,
        lr=args.lr,
    )
    trainer.fit(mm, dm, args.ckpt_path)
