# ---------------------------------------------------------------------------- #
# An sample code of LeNet-5 + BEAN regularization on few-shot learning from    #
# scratch task on MNIST dataset.                                               #
# ---------------------------------------------------------------------------- #

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epoch to run')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of classes')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='learning rate')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--n_shot', type=int, default=10,
                        help='number of samples for each class for few shot learning task')
    parser.add_argument('--l1', type=float, default=0,
                        help='scale factor for l1 regularization, 0 to disable the regularization')
    parser.add_argument('--dropout', action='store_true', default=False,
                        help='Use dropout regularization')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='scale factor for weight_decay, 0 to disable the regularization')
    parser.add_argument('--BEAN', type=int, default=1,
                        help='BEAN-n, currently implemented 1 and 2')
    parser.add_argument('--alpha', type=float, default=1,
                        help='scale factor for BEAN regularization, 0 to disable the regularization')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = get_args()

    # Hyper parameters
    num_epochs = args.num_epochs
    num_classes = args.num_classes
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    n = args.n_shot

    torch.manual_seed(args.seed)

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ---------------------------------------------------------------------------- #
    # Hyper parameters for conventional regularization
    # ---------------------------------------------------------------------------- #
    # L1
    l1 = args.l1
    # Dropout
    drop_out = args.dropout
    # weight decay
    weight_decay = args.weight_decay

    # ---------------------------------------------------------------------------- #
    # Hyper parameters for BEAN
    # ---------------------------------------------------------------------------- #
    # p = 1 -> BEAN-1
    # P = 2 -> BEAN-2
    p = args.BEAN
    # Regularization term factor, set to 0 to disable BEAN, could be tuned via validation set
    alpha = args.alpha
    # Typically set to 1, could also be tuned via validation set for better performance
    gamma = 1

    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='data/',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='data/',
                                              train=False,
                                              transform=transforms.ToTensor())


    # construct few-shot learning indices
    labels = train_dataset.train_labels[:50000]
    idx = torch.zeros(num_classes * n, dtype=torch.int32)
    for c in range(num_classes):
        # samples' index for class c
        idxs_c = torch.squeeze(torch.nonzero(labels == c))
        # randomly select n samples for each class
        perm = torch.randperm(idxs_c.size(0))
        idx_perm = perm[:n]
        idx[c * n:c * n + n] = idxs_c[idx_perm]

    train_sampler = SubsetRandomSampler(idx)

    # construct random validation set (10000) from training dataset
    perm = torch.randperm(train_dataset.train_data.size(0))
    idx = perm[:10000]
    vali_sampler = SubsetRandomSampler(idx)

    # Data loader
    vali_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               # shuffle=True,
                                               sampler=vali_sampler)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               # shuffle=True,
                                               sampler=train_sampler)

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

            if drop_out:
                h2 = torch.nn.functional.dropout(h2)

            # FC-2, then perform ReLU non-linearity
            h3 = torch.nn.functional.relu(self.fc2(h2))

            if drop_out:
                h3 = torch.nn.functional.dropout(h3)
            # FC-3
            out = self.fc3(h3)
            return h1, h2, h3, out

    model = ConvNet(num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_vali_acc = 0
    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            h1, h2, h3, outputs = model(images)
            cross_entropy_loss = criterion(outputs, labels)

            all_fc1_params = torch.cat([x.view(-1) for x in model.fc1.parameters()])
            all_fc2_params = torch.cat([x.view(-1) for x in model.fc2.parameters()])
            all_fc3_params = torch.cat([x.view(-1) for x in model.fc3.parameters()])
            l1_loss = l1 * (torch.norm(all_fc1_params, 1)+ torch.norm(all_fc2_params, 1)+ torch.norm(all_fc3_params, 1))

            # add 4 cycle on layer fc
            loss_BEAN = 0

            for fc3_weights in model.fc3.parameters():
                break
            for fc2_weights in model.fc2.parameters():
                break
            for fc1_weights in model.fc1.parameters():
                break

            # FC3
            w_hat_out = torch.abs(torch.tanh(gamma * fc3_weights.t()))
            if p == 2:
                w_corr3_out = (w_hat_out @ w_hat_out.t()) * (w_hat_out @ w_hat_out.t())
            elif p == 1:
                w_corr3_out = w_hat_out @ w_hat_out.t()

            xx = h3.t().unsqueeze(1).expand(h3.size(1), h3.size(1), h3.size(0))
            yy = h3.t().unsqueeze(0).expand(h3.size(1), h3.size(1), h3.size(0))
            # square difference
            dist = torch.pow(xx - yy, 2).mean(2)
            # absolute difference
            # dist = torch.abs(xx - yy).mean(2)
            loss_BEAN = loss_BEAN + alpha * torch.mean(w_corr3_out * dist)

            # FC2
            w_hat_out = torch.abs(torch.tanh(gamma*fc2_weights.t()))
            if p == 2:
                w_corr2_out = (w_hat_out @ w_hat_out.t()) * (w_hat_out @ w_hat_out.t())
            elif p == 1:
                w_corr2_out = w_hat_out @ w_hat_out.t()

            xx = h2.t().unsqueeze(1).expand(h2.size(1), h2.size(1), h2.size(0))
            yy = h2.t().unsqueeze(0).expand(h2.size(1), h2.size(1), h2.size(0))
            # square difference
            dist = torch.pow(xx - yy, 2).mean(2)
            # absolute difference
            # dist = torch.abs(xx - yy).mean(2)
            loss_BEAN = loss_BEAN + alpha * torch.mean(w_corr2_out * dist)

            # FC1
            w_hat = torch.abs(torch.tanh(gamma*fc1_weights.t()))
            if p == 2:
                w_corr1 = (w_hat @ w_hat.t()) * (w_hat @ w_hat.t())
            elif p == 1:
                w_corr1 = w_hat @ w_hat.t()

            xx = h1.t().unsqueeze(1).expand(h1.size(1), h1.size(1), h1.size(0))
            yy = h1.t().unsqueeze(0).expand(h1.size(1), h1.size(1), h1.size(0))
            # square difference
            dist = torch.pow(xx - yy, 2).mean(2)
            # absolute difference
            # dist = torch.abs(xx - yy).mean(2)
            loss_BEAN = loss_BEAN + alpha * torch.mean(w_corr1 * dist)

            # compute final loss
            loss = cross_entropy_loss + l1_loss + loss_BEAN

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # compute vali acc
        correct = 0
        total = 0
        for t_images, t_labels in vali_loader:
            t_images = t_images.to(device)
            t_labels = t_labels.to(device)

            _, _, _, outputs = model(t_images)

            _, predicted = torch.max(outputs.data, 1)
            total += t_labels.size(0)
            correct += (predicted == t_labels).sum().item()

        vali_acc = correct / total

        print ('Epoch [{}/{}], Step [{}/{}], Cross_entropy_loss: {:.4f}, Vali_acc: {:.2f}'
               .format(epoch+1, num_epochs, i+1, total_step, cross_entropy_loss.item(), vali_acc))
        print ('BEAN loss: {:.4f}'
               .format(loss_BEAN))
        # save the model if it achieves best acc
        if vali_acc > best_vali_acc:
            best_vali_acc = vali_acc
            # Save the model checkpoint
            torch.save(model.state_dict(),
                       './models/model_{}shot_seed_{}_BEAN{}_a_{}_l1_{}_l2_{}_drop_{}.ckpt'.format(n, args.seed,
                                                                                                   args.BEAN,
                                                                                                   args.alpha, args.l1,
                                                                                                   args.weight_decay,
                                                                                                   args.dropout))
            print ('Model saved!')

    # Test the model
    PATH = './models/model_{}shot_seed_{}_BEAN{}_a_{}_l1_{}_l2_{}_drop_{}.ckpt'.format(n, args.seed,args.BEAN,args.alpha, args.l1, args.weight_decay, args.dropout)
    model.load_state_dict(torch.load(PATH))

    model.eval()  # eval mode
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

        print('Best validation Accuracy of the model: {} %'.format(100 * best_vali_acc))
        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))


