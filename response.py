To address the issues you're facing with the `Detector` model and ensure it passes the test cases, let's make some adjustments to the model architecture and the training script. Here are some key points to consider:

1. **Model Architecture**: Ensure that the model's architecture is capable of learning both segmentation and depth estimation tasks. This involves using skip connections (like in U-Net) to help the model retain spatial information.

2. **Loss Functions**: Use appropriate loss functions for both tasks. Cross-entropy loss for segmentation and mean squared error (MSE) for depth estimation are suitable choices.

3. **Training Script**: Ensure that the training script correctly handles the data and computes the losses.

Let's update the `Detector` class and the training script:

### Updated `Detector` Class

```python
class Detector(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
    ):
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        # Downsampling layers
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Upsampling layers
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        # Segmentation head
        self.segmentation_head = nn.Conv2d(16, num_classes, kernel_size=1)

        # Depth head
        self.depth_head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Downsample
        d1 = self.down1(z)
        d2 = self.down2(d1)

        # Bottleneck
        b = self.bottleneck(d2)

        # Upsample
        u1 = self.up1(b) + d2  # Skip connection
        u2 = self.up2(u1) + d1  # Skip connection

        # Segmentation logits
        logits = self.segmentation_head(u2)

        # Depth estimation
        depth = self.depth_head(u2)

        return logits, depth

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, raw_depth = self(x)
        pred = logits.argmax(dim=1)
        depth = torch.sigmoid(raw_depth)  # Normalize depth to [0, 1]
        return pred, depth
```

### Updated `train_detection.py`

```python
def train(
    exp_dir: str = "logs",
    model_name: str = "detector",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 16,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    train_data = load_data("road_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("road_data/val", shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epoch):
        model.train()
        for batch in train_data:
            img = batch['image'].to(device)
            depth = batch['depth'].to(device)
            track = batch['track'].to(device)

            logits, pred_depth = model(img)

            seg_loss = nn.CrossEntropyLoss()(logits, track)
            depth_loss = nn.MSELoss()(pred_depth, depth)

            loss = seg_loss + depth_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.inference_mode():
            model.eval()
            for batch in val_data:
                img = batch['image'].to(device)
                depth = batch['depth'].to(device)
                track = batch['track'].to(device)

                logits, pred_depth = model(img)

                _, preds = torch.max(logits, 1)
                accuracy = (preds == track).float().mean().item()

        print(f"Epoch {epoch + 1}/{num_epoch}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

    save_model(model)
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")
```

### Key Changes:
- **Skip Connections**: Added skip connections between downsampling and upsampling layers to help retain spatial information.
- **Depth Normalization**: Used `torch.sigmoid` to normalize depth predictions to the range [0, 1].
- **Optimizer**: Switched to `Adam` optimizer for potentially better convergence.

These changes should help improve the model's performance and pass the test cases. Make sure to adjust any other parts of your code as necessary to integrate these updates.
