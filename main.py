# This the Python script to train GAN

import torch
import numpy as np
from model import Generator_1
from model import Discriminator_1
import argparse
from tqdm import tqdm
import time
import scipy.io


def load_mnist(d_dir):
    mnist = scipy.io.loadmat(d_dir + '/MNIST/mnist_all.mat')
    mnist_train = np.concatenate((mnist['train0'], mnist['train1'], mnist['train2'],
                                  mnist['train3'], mnist['train4'], mnist['train5'], mnist['train6'],
                                  mnist['train7'], mnist['train8'], mnist['train9']), axis=0) / 256
    mnist_test = np.concatenate((mnist['test0'], mnist['test1'], mnist['test2'],
                                 mnist['test3'], mnist['test4'], mnist['test5'], mnist['test6'],
                                 mnist['test7'], mnist['test8'], mnist['test9']), axis=0) / 256
    return mnist_train, mnist_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo for Training original GAN")
    parser.add_argument("--dataset", default='mnist', dest='dataset',
                        choices=('mnist', 'cifar10'),
                        help="Dataset to train the GAN")
    parser.add_argument("--data_dir", dest='data_dir', default="../../dataset",
                        help="The directory of your dataset")
    parser.add_argument("--epochs", dest='num_epochs', default=2, type=int,
                        help="Total number of epochs")
    parser.add_argument("--batch_size", dest="batch_size", default=100, type=int,
                        help="The batch size")
    parser.add_argument("--k", dest="k", default=1, type=int,
                        help="Train discriminator k times and then generator")
    parser.add_argument("--logD_times", dest="logD_times", default=50, type=int,
                        help="Use the -log(D) trick for how many times")
    parser.add_argument("--lr_G", dest="lr_G", default=2e-4, type=float,
                        help="Learning rate for the generator")
    parser.add_argument("--lr_D", dest="lr_D", default=1e-4, type=float,
                        help="Learning rate for the discriminator")
    parser.add_argument('--device', dest='device', default=0, type=int,
                        help='Index of device')
    parser.add_argument("--save_dir", dest='save_dir', default="./saved_models",
                        help="The directory to save your trained model")
    parser.add_argument("--Nz", dest='Nz', default=20, type=int,
                        help="Nz (dimension of the latent code)")
    args = parser.parse_args()

    # load data
    if args.dataset == 'mnist':
        Nz = args.Nz
        batch_size = args.batch_size
        dim = 28 * 28
        train_num = 60000
        lr_G = args.lr_G
        lr_D = args.lr_D
        training_set, test_set = load_mnist(args.data_dir)
    else:
        raise Exception("I didn't get a MLP-based GAN model for cifar10, please try DCGAN")

    print(training_set.shape)
    print(test_set.shape)
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size,
                                               shuffle=True, num_workers=2)
    # define the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net_G = Generator_1(Nz, dim)
    net_D = Discriminator_1(dim)
    net_G.to(device)
    net_D.to(device)
    optimizer_G = torch.optim.RMSprop(net_G.parameters(), lr=lr_G)
    optimizer_D = torch.optim.RMSprop(net_D.parameters(), lr=lr_D)

    # train the model
    start = time.time()
    for epoch in range(args.num_epochs):

        # iterations
        running_loss_D = 0.0
        running_loss_G = 0.0
        with tqdm(total=len(train_loader.dataset)) as progress_bar:
            for i, data in enumerate(train_loader, 0):
                real = data.to(device).float()
                optimizer_D.zero_grad()

                # sample minibatch from noise prior
                noise_sample = torch.randn(batch_size, Nz).to(device)
                fake = net_G(noise_sample).detach()

                # loss for the discriminator
                loss_D = -(torch.log(net_D(real)) + torch.log(1 - net_D(fake))).mean()

                # train the discriminator
                loss_D.backward()
                optimizer_D.step()
                running_loss_D += loss_D.item() * batch_size

                if (i + 1) % args.k == 0:
                    optimizer_G.zero_grad()

                    # sample minibatch from noise prior
                    noise_sample = torch.randn(batch_size, Nz).to(device)
                    fake = net_G(noise_sample)

                    # loss for the generator
                    if epoch < args.logD_times:
                        # the -log(D) trick
                        loss_G = -torch.log(net_D(fake)).mean()
                    else:
                        loss_G = torch.log(1 - net_D(fake)).mean()

                    # train the generator
                    loss_G.backward()
                    optimizer_G.step()
                    running_loss_G += loss_G.item() * batch_size

                # progress bar
                progress_bar.set_postfix(loss_D_G=(loss_D.item(), loss_G.item()))
                progress_bar.update(data.size(0))

        print('[%d] loss_D: %.3f' % (epoch + 1, running_loss_D / train_num))
        print('[%d] loss_G: %.3f' % (epoch + 1, running_loss_G / train_num))

    print('Finished Training, time cost', time.time() - start)

    PATH = args.save_dir + '/Nz_' + str(args.Nz) + '_dataset_' + args.dataset
    torch.save(net_D.state_dict(), PATH + "_Discriminator.pth")
    torch.save(net_G.state_dict(), PATH + "_Generator.pth")
