import argparse
from src.dataset import HierTextDataModule
from src.trainer import HierTextModelModule
from src.callback import EarlyStoppingCallback

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_name", type=str)
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--label_dir", type=str)
    parser.add_argument("--image_size", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    parser = HierTextModelModule.add_model_specific_args(parser)
    parser = HierTextModelModule.add_trainer_specific_args(parser)
    args = parser.parse_args()

    dm = HierTextDataModule(
        args.dataset_dir,
        args.label_dir,
        args.image_size,
        args.batch_size,
        args.num_workers,
    )
    mm = HierTextModelModule(**vars(args))
    early_stopping_cb = EarlyStoppingCallback(
        patience=5, monitor="val_loss", mode="min"
    )
    mm.fit(dm, args.epochs, callbacks_=[early_stopping_cb])
    mm.save_weights("best.pth")
