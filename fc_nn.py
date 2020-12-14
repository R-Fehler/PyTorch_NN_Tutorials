import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Create FCNN


class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('running on {}'.format(device))
# Hyperparameters
in_size = 28*28
num_of_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 16

# Load Data
train_dataset = datasets.MNIST(
    root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(
    root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size, shuffle=True)

# initialize Network

model = NN(input_size=in_size, num_classes=num_of_classes).to(device)
print('Number of Parameters: {}'.format(model.count_parameters()))
#Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Define the Accuracy Check Function

def check_accuracy(loader, model: NN):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            # 64 img x 10
            # the index of the max value is relevant not the max value ( a probability)
            _, predictions = scores.max(1)
            # of type tensor so we convert them for printing in floats
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        acc = float(num_correct)/float(num_samples)*100
        print(f'Got {num_correct} / {num_samples} with accuracy {acc:.2f}')

    # model.train() # if you want to get the accuracy while training, we dont have a training method implemented yet, but trained beforehand
    return acc

# Train Network


for epoch in range(num_epochs):
    for batch_index, (data, targets) in enumerate(train_loader):
        # get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)
        # get to correct input shape (input layer is linear with size 758 or smth)
        data = data.reshape(data.shape[0], -1)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()
    print('epoch No: {}'.format(epoch))
    check_accuracy(train_loader, model)
    check_accuracy(test_loader, model)


# Check accuracy on training & test to see how good the model is


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
