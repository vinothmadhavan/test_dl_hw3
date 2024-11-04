import argparse
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import ClassificationLoss, load_model, save_model
from .datasets.road_dataset import load_data
# from .utils import load_data

def train(
    exp_dir: str = "logs",
    model_name: str = "linear",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    train_data = load_data("road_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("road_data/val", shuffle=False)

    # create loss function and optimizer
    loss_func = ClassificationLoss()
    # optimizer = ...
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)


    global_step = 0
    metrics = {"train_acc": [], "val_acc": []}


    # Training loop
    for epoch in range(num_epoch):
        # Clear metrics at beginning of epoch
        for key in metrics:
            metrics[key].clear()

        model.train()

        for batch in train_data:
            img = batch['image'].to(device)
            depth = batch['depth'].to(device)
            track = batch['track'].to(device)

            # Forward pass
            logits, pred_depth = model(img)

            # Calculate losses
            seg_loss = nn.CrossEntropyLoss()(logits, track)
            depth_loss = nn.MSELoss()(pred_depth, depth)

            # Combine losses
            loss = seg_loss + depth_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate training accuracy for segmentation
            _, preds = torch.max(logits, 1)
            accuracy = (preds == track).float().mean().item()
            metrics["train_acc"].append(accuracy)

            global_step += 1

        # Disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()

            for batch in val_data:
                img = batch['image'].to(device)
                depth = batch['depth'].to(device)
                track = batch['track'].to(device)

                # Forward pass
                logits, pred_depth = model(img)

                # Calculate validation accuracy for segmentation
                _, preds = torch.max(logits, 1)
                accuracy = (preds == track).float().mean().item()
                metrics["val_acc"].append(accuracy)

        # Log average train and val accuracy to tensorboard
        epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()
        epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean()

        logger.add_scalar('train_accuracy', epoch_train_acc, epoch)
        logger.add_scalar('val_accuracy', epoch_val_acc, epoch)

        # Print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={epoch_train_acc:.4f} "
                f"val_acc={epoch_val_acc:.4f}"
            )

    # # training loop
    # for epoch in range(num_epoch):
    #     # clear metrics at beginning of epoch
    #     for key in metrics:
    #         metrics[key].clear()

    #     model.train()

    #     for img, label in train_data:
    #         img, label = img.to(device), label.to(device)

    #         # Forward pass
    #         logits = model(img)
    #         loss = loss_func(logits, label)

    #         # Backward pass and optimization
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         # Calculate training accuracy
    #         _, preds = torch.max(logits, 1)
    #         accuracy = (preds == label).float().mean().item()
    #         metrics["train_acc"].append(accuracy)

    #         global_step += 1

    #         global_step += 1

    #     # disable gradient computation and switch to evaluation mode
    #     with torch.inference_mode():
    #         model.eval()

    #         for img, label in val_data:
    #             img, label = img.to(device), label.to(device)

    #             # Forward pass
    #             logits = model(img)

    #             # Calculate validation accuracy
    #             _, preds = torch.max(logits, 1)
    #             accuracy = (preds == label).float().mean().item()
    #             metrics["val_acc"].append(accuracy)

    #     # log average train and val accuracy to tensorboard
    #     epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()
    #     epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean()

    #     logger.add_scalar('train_accuracy', epoch_train_acc, epoch)
    #     logger.add_scalar('val_accuracy', epoch_val_acc, epoch)

    #     # print on first, last, every 10th epoch
    #     if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
    #         print(
    #             f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
    #             f"train_acc={epoch_train_acc:.4f} "
    #             f"val_acc={epoch_val_acc:.4f}"
    #         )

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))
