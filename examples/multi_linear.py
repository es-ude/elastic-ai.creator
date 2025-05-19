import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.utils.data import random_split
from elasticai.creator.vhdl.code_generation.code_abstractions import to_vhdl_binary_string
from elasticai.creator.nn.fixed_point._math_operations import MathOperations
from elasticai.creator.file_generation.on_disk_path import OnDiskPath
from elasticai.creator.nn.fixed_point import Linear, Tanh, HardTanh, ReLU
from elasticai.creator.nn.sequential import Sequential
from elasticai.creator.nn.fixed_point._two_complement_fixed_point_config import FixedPointConfig
from cable_length_dataset import CableLengthDataset

total_bits = 8
frac_bits = 5
learning_rate = 0.001
num_epochs = 5000

config = FixedPointConfig(total_bits=total_bits, frac_bits=frac_bits)
ops = MathOperations(config=config)

dataset = CableLengthDataset("fixed_current_training_data.csv")
train, valid, test = random_split(dataset, lengths=[0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(0))
train_dl = DataLoader(train, batch_size=1000, shuffle=True)
valid_dl = DataLoader(valid, batch_size=1000, shuffle=True)
test_dl = DataLoader(test, batch_size=1000, shuffle=True)

multi_linear = Sequential(
    Linear(in_features=19, out_features=10, total_bits=total_bits, frac_bits=frac_bits, parallel=False),
    Tanh(total_bits=total_bits, frac_bits=frac_bits, num_steps=2**8, sampling_intervall=(-4, 3.96875)), # num_steps auf 2**8 und sampling_intervall = (-4, 3.96875)
    Linear(in_features=10, out_features=4, total_bits=total_bits, frac_bits=frac_bits, parallel=False),
)

# Training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(multi_linear.parameters(), lr=learning_rate)
best_valid_loss = float("inf")

for epoch in range(num_epochs):
    multi_linear.train()
    train_loss = 0.0
    valid_loss = 0.0
    train_correct_predictions = 0
    train_total_samples = 0
    valid_correct_predictions = 0
    valid_total_samples = 0

    for batch_features, batch_targets in train_dl:
        # Forward pass
        outputs = multi_linear(batch_features)
        loss = criterion(outputs, batch_targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss
        train_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)  # Get predicted labels
        train_correct_predictions += (predicted == batch_targets).sum().item()
        train_total_samples += batch_targets.size(0)

    for batch_features, batch_targets in valid_dl:
        with torch.no_grad():
            outputs = multi_linear(batch_features)
            loss = criterion(outputs, batch_targets)

            # Accumulate loss
            valid_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)  # Get predicted labels
            valid_correct_predictions += (predicted == batch_targets).sum().item()
            valid_total_samples += batch_targets.size(0)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(multi_linear.state_dict(), 'build_dir/leds/best_model_3bit.pth')

    train_accuracy = train_correct_predictions / train_total_samples
    valid_accuracy = valid_correct_predictions / valid_total_samples

    # Print epoch summary
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/1000:.6f}, Valid Loss: {valid_loss/1000:.6f}")
    print(f"Accuracy: Train: {train_accuracy:.2%}, Valid: {valid_accuracy:.2%}")

multi_linear.load_state_dict(torch.load('build_dir/leds/best_model_3bit.pth'))
multi_linear.eval()

with torch.no_grad():
    multi_linear[0].weight.data = ops.quantize(multi_linear[0].weight)
    multi_linear[0].bias.data = ops.quantize(multi_linear[0].bias)
    multi_linear[2].weight.data = ops.quantize(multi_linear[2].weight)
    multi_linear[2].bias.data = ops.quantize(multi_linear[2].bias)

# Testing
test_loss = 0.0
test_correct_predictions = 0
test_total_samples = 0
for batch_features, batch_targets in test_dl:
    with torch.no_grad():
        outputs = multi_linear(batch_features)
        loss = criterion(outputs, batch_targets)

        # Accumulate loss
        test_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)  # Get predicted labels
        test_correct_predictions += (predicted == batch_targets).sum().item()
        test_total_samples += batch_targets.size(0)
test_accuracy = test_correct_predictions / test_total_samples
print(f"Test Loss: {test_loss/1000:.6f}, Test Accuracy: {test_accuracy:.2%}")

# Save Model
destination = OnDiskPath("build_dir/leds/3bit")
design = multi_linear.create_design("led_model")
design.save_to(destination)
torch.save(multi_linear.state_dict(), 'build_dir/leds/eval_model_3bit.pth')