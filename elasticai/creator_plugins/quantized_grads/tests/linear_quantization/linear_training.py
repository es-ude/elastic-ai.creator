import torch
from torch.nn import Sequential, Flatten, Linear as TorchLinear, ReLU as TorchReLU, Conv2d as TorchConv2d
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from elasticai.creator_plugins.quantized_grads.linear_quantization import LinearAsymQuantizationConfig, \
    ModuleQuantizeLinearAsymForwStochastic, QuantizeParamSTEToLinearAsymQuantizationStochastic
from elasticai.creator_plugins.quantized_grads.linear_quantization.base_modules import Linear
from elasticai.creator_plugins.quantized_grads.linear_quantization.base_modules.conv2d import Conv2d
from elasticai.creator_plugins.quantized_grads.linear_quantization.base_modules.relu import ReLU
from elasticai.creator_plugins.quantized_grads.linear_quantization.module_quantization import QuantizationModule, \
    ModuleQuantizeLinearAsymForwHTE
from elasticai.creator_plugins.quantized_grads.linear_quantization.param_quantization import ParamQuantizationModule, \
    ParamQuantizationSimulatedModule, QuantizeParamSTEToLinearAsymQuantizationHTE, QuantizeParamSTEToIntHTE, \
    QuantizeParamSTEToIntStochastic


def get_linear_model(in_features=28*28, out_features=10):
    return Sequential(Flatten(),
                      TorchLinear(in_features=in_features,
                             out_features=100,
                             bias=True,),
                      TorchReLU(),
                      TorchLinear(in_features=100,
                             out_features=out_features,
                             bias=True,),
                      )

def get_linear_model_quantized(in_features=28*28, out_features=10):
    input_quantization = ModuleQuantizeLinearAsymForwHTE(LinearAsymQuantizationConfig(8))
    weight_quantization = QuantizeParamSTEToLinearAsymQuantizationHTE(LinearAsymQuantizationConfig(8))
    bias_quantization = QuantizeParamSTEToIntHTE(LinearAsymQuantizationConfig(8))
    return Sequential(Flatten(),
                      Linear(in_features=in_features,
                             out_features=100,
                             bias=True,
                             input_quantization=input_quantization,
                             output_quantization=input_quantization,
                             weight_quantization=weight_quantization,
                             bias_quantization=bias_quantization),
                      ReLU(),
                      Linear(in_features=100,
                             out_features=out_features,
                             bias=True,
                             input_quantization=input_quantization,
                             output_quantization=input_quantization,
                             weight_quantization=weight_quantization,
                             bias_quantization=bias_quantization),
                      )

def get_cnn_model(in_features=28*28, out_features=10):
    return Sequential(TorchConv2d(in_channels=1,
                             out_channels=16,
                             kernel_size=3,
                             padding=1,),
                      TorchReLU(),
                      TorchConv2d(in_channels=16,
                             out_channels=32,
                             kernel_size=3,
                             stride=2,
                             padding=1,),
                      TorchReLU(),
                      TorchConv2d(in_channels=32,
                             out_channels=64,
                             kernel_size=3,
                             stride=2,
                             padding=1,),
                      TorchReLU(),
                      Flatten(),
                      TorchLinear(in_features=64 *7 * 7,
                             out_features=128,
                             bias=True),
                      TorchReLU(),
                      TorchLinear(in_features=128,
                             out_features=out_features,
                             bias=True))

def get_cnn_quantized_model(out_features=10):
    num_bits = 32
    input_quantization = ModuleQuantizeLinearAsymForwStochastic(LinearAsymQuantizationConfig(num_bits))
    weight_quantization = QuantizeParamSTEToLinearAsymQuantizationStochastic(LinearAsymQuantizationConfig(num_bits))
    bias_quantization = QuantizeParamSTEToIntStochastic(LinearAsymQuantizationConfig(num_bits))
    return Sequential(Conv2d(in_channels=1,
                             out_channels=16,
                             kernel_size=3,
                             padding=1,
                             input_quantization=input_quantization,
                             output_quantization=input_quantization,
                             weight_quantization=weight_quantization,
                             bias_quantization=bias_quantization),
                      ReLU(),
                      Conv2d(in_channels=16,
                             out_channels=32,
                             kernel_size=3,
                             stride=2,
                             padding=1,
                             input_quantization=input_quantization,
                             output_quantization=input_quantization,
                             weight_quantization=weight_quantization,
                             bias_quantization=bias_quantization),
                      ReLU(),
                      Conv2d(in_channels=32,
                             out_channels=64,
                             kernel_size=3,
                             stride=2,
                             padding=1,
                             input_quantization=input_quantization,
                             output_quantization=input_quantization,
                             weight_quantization=weight_quantization,
                             bias_quantization=bias_quantization),
                      ReLU(),
                      Flatten(),
                      Linear(in_features=64 *7 * 7,
                             out_features=128,
                             bias=True,
                             input_quantization=input_quantization,
                             output_quantization=input_quantization,
                             weight_quantization=weight_quantization,
                             bias_quantization=bias_quantization),
                      ReLU(),
                      Linear(in_features=128,
                             out_features=out_features,
                             bias=True,
                             input_quantization=input_quantization,
                             output_quantization=input_quantization,
                             weight_quantization=weight_quantization,
                             bias_quantization=bias_quantization),)


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def main():
    # Hyperparameter
    batch_size = 64
    lr = 1e-3
    num_epochs = 5

    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Transforms: nur Normalisierung für MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Datasets & Dataloaders
    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # Modell, Loss, Optimizer
    #model = get_model()
    #model = get_linear_model()
    #model = get_linear_model_quantized()
    #model = get_cnn_model()
    model = get_cnn_quantized_model()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training Loop
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )

        print(
            f"Epoch [{epoch}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
            f"| Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
        )


if __name__ == "__main__":
    main()