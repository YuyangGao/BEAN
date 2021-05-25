# ---------------------------------------------------------------------------- #
# An sample code of Resnet + BEAN regularization on CIFAR-10 dataset           #
# ---------------------------------------------------------------------------- #

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from resnet import ResNet18

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Model Hyper parameters
num_epochs = 50
num_classes = 10
batch_size = 100
learning_rate = 0.0005

# ---------------------------------------------------------------------------- #
# Hyper parameters for BEAN
# ---------------------------------------------------------------------------- #
# p = 1 -> BEAN-1
# P = 2 -> BEAN-2
p = 1
# Regularization term factor, set to 0 to disable BEAN, could be tuned via validation set
alpha = 3
# set to 1 as default
gamma = 1

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

model = ResNet18().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        h, outputs = model(images)
        cross_entropy_loss = criterion(outputs, labels)
        # add 4 cycle on layer fc
        loss_BEAN = 0

        for fc_weights in model.linear.parameters():
            break

        # FC
        w_hat_out = torch.abs(torch.tanh(gamma * fc_weights.t()))
        if p == 2:
            # original way
            w_corr_out = (w_hat_out @ w_hat_out.t()) * (w_hat_out @ w_hat_out.t())

        elif p == 1:
            # original way
            w_corr_out = w_hat_out @ w_hat_out.t()

            # cosine
            # w_hat_out = w_hat_out.repeat(w_hat_out.size(0), 1, 1)
            # w_corr_out = cos(w_hat_out, w_hat_out)

        xx = h.t().unsqueeze(1).expand(h.size(1), h.size(1), h.size(0))
        yy = h.t().unsqueeze(0).expand(h.size(1), h.size(1), h.size(0))
        # square difference
        dist = torch.pow(xx - yy, 2).mean(2)
        # absolute difference
        # dist = torch.abs(xx - yy).mean(2)
        loss_BEAN = loss_BEAN + alpha * torch.mean(w_corr_out * dist)

        # compute final loss
        loss = cross_entropy_loss + loss_BEAN

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute accuracy
        _, argmax = torch.max(outputs, 1)
        accuracy = (labels == argmax.squeeze()).float().mean()

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc: {:.2f}'
                   .format(epoch+1, num_epochs, i+1, total_step, cross_entropy_loss.item(), accuracy.item()))
            print ('BEAN loss: {:.4f}'
                   .format(loss_BEAN))

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        _, outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), './models/model.ckpt')


