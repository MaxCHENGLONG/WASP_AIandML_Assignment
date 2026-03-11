import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Loading Dataset and DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # Normalization for MNIST dataset
])


train_dataset = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=1000, shuffle=False)
# Model Definition
class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),               # 28×28 → 784
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),         # 10 classes for MNIST
            nn.Softmax(dim=1)           # → p(k | x) = f_k(x)
        )

    def forward(self, x):
        return self.net(x)
    

# Loss Function and Optimizer
# Pytoch's CrossEntropyLoss combines LogSoftmax and NLLLoss, so we should remove the Softmax layer from the model.
# CrossEntropyLoss = NLLLoss(LogSoftmax(x))
model = MNISTClassifier()
critierion = nn.CrossEntropyLoss()  # This will compute the loss based on the raw output (logits) from the model
optimizer = optim.Adam(model.parameters(), lr=0.001)

model_iteration_history = {
    'iteration': [],
    'train_loss': [],
    'train_acc':  [],
    'test_loss':  [],
    'test_acc':   []
}

model_epoch_history = {
    'epoch': [],
    'train_loss': [],
    'train_acc':  [],
    'test_loss':  [],
    'test_acc':   []
}

# Training Function
def train_one_epoch(model, loader, optimizer, criterion, start_iteration=0):
    model.train()
    total_loss, correct = 0, 0
    iteration_records = []
    global_iteration = start_iteration

    for images, labels in loader:
        optimizer.zero_grad()
        probs = model(images)
        loss  = criterion(torch.log(probs), labels)
        loss.backward()
        optimizer.step()

        batch_size = len(images)
        batch_loss = loss.item()
        batch_correct = probs.argmax(1).eq(labels).sum().item()
        batch_acc = batch_correct / batch_size

        global_iteration += 1
        iteration_records.append({
            'iteration': global_iteration,
            'train_loss': batch_loss,
            'train_acc': batch_acc
        })

        total_loss += batch_loss * batch_size
        correct    += batch_correct
    n = len(loader.dataset)
    return total_loss / n, correct / n, iteration_records, global_iteration  # Return average loss and accuracy(<1)

# Evaluation Function
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            probs = model(images)
            loss  = criterion(torch.log(probs), labels)
            total_loss += loss.item() * len(images)
            correct    += probs.argmax(1).eq(labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n  # Return average loss and accuracy(<1)

# Main Training Loop
num_epochs = 5
global_iteration = 0

for epoch in range(num_epochs):
    train_loss, train_acc, iteration_records, global_iteration = train_one_epoch(
        model,
        train_loader,
        optimizer,
        critierion,
        start_iteration=global_iteration
    )
    test_loss,  test_acc  = evaluate(model, test_loader, critierion)

    for record in iteration_records:
        model_iteration_history['iteration'].append(record['iteration'])
        model_iteration_history['train_loss'].append(record['train_loss'])
        model_iteration_history['train_acc'].append(record['train_acc'])

    model_epoch_history['epoch'].append(epoch + 1)
    model_epoch_history['train_loss'].append(train_loss)
    model_epoch_history['train_acc'].append(train_acc)
    model_epoch_history['test_loss'].append(test_loss)
    model_epoch_history['test_acc'].append(test_acc)

    print(f'Epoch {epoch+1}/{num_epochs} - '
          f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - '
          f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

import json

with open('/Users/max/WASP_AIandML_Assignment/nesy_semanticloss/code/Q1_history/model_epoch_history.json', 'w') as f:
    json.dump(model_epoch_history, f)
print("Model epoch history saved to 'model_epoch_history.json'.")

with open('/Users/max/WASP_AIandML_Assignment/nesy_semanticloss/code/Q1_history/model_iteration_history.json', 'w') as f:
    json.dump(model_iteration_history, f)
print("Model iteration history saved to 'model_iteration_history.json'.")

if __name__ == "__main__":
    # The training loop is already executed above, so we can just pass here.
    pass