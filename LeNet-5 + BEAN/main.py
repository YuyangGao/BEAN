# ---------------------------------------------------------------------------- #
# An sample code of LeNet-5 + BEAN regularization on MNIST dataset             #
# ---------------------------------------------------------------------------- #

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

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


# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='data/',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='data/',
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

# Convolutional neural network (LeNet 5)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        # Convolution (In LeNet-5, 32x32 images are given as input. Hence padding of 2 is done below)
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        # Max-pooling
        self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=2)
        # Convolution
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
        # Max-pooling
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=2)
        # Fully connected layer
        self.fc1 = torch.nn.Linear(16*5*5, 120)   # convert matrix with 16*5*5 (= 400) features to a matrix of 120 features (columns)
        self.fc2 = torch.nn.Linear(120, 84)       # convert matrix with 120 features to a matrix of 84 features (columns)
        self.fc3 = torch.nn.Linear(84, 10)        # convert matrix with 84 features to a matrix of 10 features (columns)
        
    def forward(self, x):
        # convolve, then perform ReLU non-linearity
        h = torch.nn.functional.relu(self.conv1(x))
        # max-pooling with 2x2 grid
        h = self.max_pool_1(h)
        # convolve, then perform ReLU non-linearity
        h = torch.nn.functional.relu(self.conv2(h))
        # max-pooling with 2x2 grid
        h = self.max_pool_2(h)
        # first flatten 'max_pool_2_out' to contain 16*5*5 columns
        h1 = h.view(-1, 16*5*5)
        # FC-1, then perform ReLU non-linearity
        h2 = torch.nn.functional.relu(self.fc1(h1))
        # FC-2, then perform ReLU non-linearity
        h3 = torch.nn.functional.relu(self.fc2(h2))
        # FC-3
        out = self.fc3(h3)
        return h1, h2, h3, out

model = ConvNet(num_classes).to(device)

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
        h1, h2, h3, outputs = model(images)
        cross_entropy_loss = criterion(outputs, labels)

        # add 4 cycle on all fc layers
        loss_BEAN = 0
        # fc1
        for fc1_weights in model.fc1.parameters():
            w_hat = torch.abs(torch.tanh(gamma * fc1_weights.t()))

            if p == 2:
                w_corr = (w_hat @ w_hat.t()) * (w_hat @ w_hat.t())
            elif p == 1:
                w_corr = w_hat @ w_hat.t()

            xx = h1.t().unsqueeze(1).expand(h1.size(1), h1.size(1), h1.size(0))
            yy = h1.t().unsqueeze(0).expand(h1.size(1), h1.size(1), h1.size(0))
            # square difference
            dist = torch.pow(xx - yy, 2).mean(2)
            # absolute difference
            # dist = torch.abs(xx - yy).mean(2)

            loss_BEAN = loss_BEAN + alpha * torch.mean(w_corr * dist)

            break
        # fc2
        for fc2_weights in model.fc2.parameters():
            w_hat = torch.abs(torch.tanh(gamma * fc2_weights.t()))

            if p == 2:
                w_corr = (w_hat @ w_hat.t()) * (w_hat @ w_hat.t())
            elif p == 1:
                w_corr = w_hat @ w_hat.t()

            xx = h2.t().unsqueeze(1).expand(h2.size(1), h2.size(1), h2.size(0))
            yy = h2.t().unsqueeze(0).expand(h2.size(1), h2.size(1), h2.size(0))
            # square difference
            dist = torch.pow(xx - yy, 2).mean(2)
            # absolute difference
            # dist = torch.abs(xx - yy).mean(2)

            loss_BEAN = loss_BEAN + alpha * torch.mean(gamma * w_corr * dist)
            break
        # fc3
        for fc3_weights in model.fc3.parameters():
            w_hat = torch.abs(torch.tanh(fc3_weights.t()))

            if p == 2:
                w_corr = (w_hat @ w_hat.t()) * (w_hat @ w_hat.t())
            elif p == 1:
                w_corr = w_hat @ w_hat.t()

            xx = h3.t().unsqueeze(1).expand(h3.size(1), h3.size(1), h3.size(0))
            yy = h3.t().unsqueeze(0).expand(h3.size(1), h3.size(1), h3.size(0))
            # square difference
            dist = torch.pow(xx - yy, 2).mean(2)
            # absolute difference
            # dist = torch.abs(xx - yy).mean(2)

            loss_BEAN = loss_BEAN + alpha * torch.mean(w_corr * dist)
            break

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
        _, _, _, outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), './models/model.ckpt')


