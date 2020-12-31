# After running main.py to train the model, this script can let
# you generate images using one of your saved model


import torch
import torchvision
from model import Generator_1
import argparse


def generate(net_g, z, dataset):
    if dataset == 'cifar10':
        fake = net_g(z).cpu().detach().view(1, 3, 32, 32)
    else:
        fake = net_g(z).cpu().detach().view(1, 1, 28, 28)
    return fake


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Handwritten Digits")
    parser.add_argument("--dataset", default='mnist', dest='dataset',
                        choices=('mnist', 'cifar10'),
                        help="Dataset used to train the GAN")
    parser.add_argument("--model_dir", dest='model_dir',
                        default="./saved_models",
                        help="The directory and name of your model")
    parser.add_argument("--Nz", default=20, type=int,
                        help="Nz (dimension of the latent code)")
    parser.add_argument("--num_images", dest='n_images', default=1, type=int,
                        help="The number of images you want")
    args = parser.parse_args()

    if args.dataset == 'mnist':
        Nz = args.Nz
        dim = 28 * 28
    else:
        raise Exception("I didn't get a MLP-based GAN model for cifar10, please try DCGAN")

    PATH = args.model_dir + '/Nz_' + str(args.Nz) + '_dataset_' + args.dataset
    filename = PATH + "_Generator.pth"
    net_G = Generator_1(Nz, dim)
    net_G.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
    net_G.eval()

    for i in range(args.n_images):
        z1 = torch.randn(1, args.Nz)
        img = generate(net_G, z1, args.dataset)
        torchvision.utils.save_image(img, './generated_images/'+str(args.dataset) + str(i) + '.png',
                                     normalize=True)

