import argparse
import torch, torchvision
from torchvision import transforms
import GAN
import torch.nn as nn
from matplotlib import pyplot as plt
from torchvision.utils import save_image


def train(G,D, trainloader, optimizer_G, optimizer_D, epochs, loss_type, device, dataset, sample_size, train_per_epoch=2):
    G.train()  # set to training mode
    D.train()
    if loss_type == 'standard':
        print(f'Loss type is: {loss_type}. The generator maximizes E[log(D(G(z))]')
    else:
        print(f'Loss type is: {loss_type}. The generator minimizes E[log(1 - D(G(z))]')

    disc_losses = []
    gen_losses = []
    disc_loss_total = []
    gen_loss_total = []
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        for batch_idx, (data, _) in enumerate(trainloader):
            G.zero_grad()
            noisy_input = torch.randn(trainloader.batch_size, G.latent_dim).to(device)
            generator = G(noisy_input)
            discriminator = D(generator)
            if loss_type == 'standard':
                label = torch.ones(trainloader.batch_size, 1).to(device)
                gen_loss = criterion(discriminator, label)
            else:
                label = torch.zeros(trainloader.batch_size, 1).to(device)
                gen_loss = - criterion(discriminator, label)

            gen_loss.backward()
            optimizer_G.step()
            gen_losses.append(gen_loss.item())

            for idx in range(train_per_epoch):
                D.zero_grad()
                noisy_input = torch.randn(trainloader.batch_size, G.latent_dim).to(device)
                ### training on fake (generated) images:
                fake_img = G(noisy_input)
                label_fake = torch.zeros(trainloader.batch_size, 1).to(device)
                D_fake = D(fake_img)
                D_fake_loss = criterion(D_fake, label_fake)
                real_img = data.view(-1, 784).to(device)
                label_real = torch.ones(real_img.size(0), 1).to(device)
                ### train on real images:
                D_real = D(real_img)
                D_real_loss = criterion(D_real, label_real)
                D_total_loss = D_real_loss + D_fake_loss
                D_total_loss.backward()
                optimizer_D.step()
            disc_losses.append(D_total_loss.item())
        gen_loss_total.append(torch.mean(torch.FloatTensor(gen_losses)))
        disc_loss_total.append(torch.mean(torch.FloatTensor(disc_losses)))

        print(f'Epoch: {epoch + 1}, discriminator loss: {disc_loss_total[-1]}, generator loss: {gen_loss_total[-1]}')
        if (epoch + 1) % 10 == 0:
            sample(G, sample_size, device, epoch+1, dataset, loss_type)
    return gen_loss_total, disc_loss_total


def sample(G, sample_size, device, epoch, dataset, loss_type):
    G.eval()  # set to inference mode
    with torch.no_grad():
        #sampling and generate
        noisy_input = torch.randn(sample_size, G.latent_dim).to(device)
        output = G(noisy_input)
        output = (output + 1) / 2
        save_image(output.view(output.size(0), 1, 28, 28), './samples/sample_' + f'{dataset}_{loss_type}_epoch_{epoch}.png')

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5), std=(0.5)),
    ])

    if args.dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data/MNIST',
                                              train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=args.batch_size, shuffle=True, num_workers=0)
        testset = torchvision.datasets.MNIST(root='./data/MNIST',
                                             train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=args.batch_size, shuffle=False, num_workers=0)
    elif args.dataset == 'fashion-mnist':
        trainset = torchvision.datasets.FashionMNIST(root='~/torch/data/FashionMNIST',
                                                     train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=args.batch_size, shuffle=True, num_workers=0)
        testset = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
                                                    train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=args.batch_size, shuffle=False, num_workers=0)
    else:
        raise ValueError('Dataset not implemented')

    G = GAN.Generator(latent_dim=args.latent_dim,
                      batch_size=args.batch_size, device=device).to(device)
    D = GAN.Discriminator().to(device)

    optimizer_G = torch.optim.Adam(
        G.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(
        D.parameters(), lr=args.lr)
    # Train function
    gen_loss, disc_loss = train(G, D, trainloader, optimizer_G, optimizer_D, args.epochs, args.loss_type,
                                device, args.dataset, args.sample_size, train_per_epoch=2)

    with torch.no_grad():
        fig, ax = plt.subplots()
        ax.plot(gen_loss)
        ax.plot(disc_loss)
        ax.set_title("Generator and Discriminator Losses")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        plt.legend(['Generator loss', 'Discriminator loss'])
        plt.savefig("./loss/" + f'{args.dataset}_{args.loss_type}_loss.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset',
                        help='dataset to be modeled.',
                        type=str,
                        default='mnist')
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=128)
    parser.add_argument('--epochs',
                        help='maximum number of iterations.',
                        type=int,
                        default=100)
    parser.add_argument('--sample_size',
                        help='number of images to generate.',
                        type=int,
                        default=64)

    parser.add_argument('--latent-dim',
                        help='.',
                        type=int,
                        default=100)
    parser.add_argument('--lr',
                        help='initial learning rate.',
                        type=float,
                        default=2e-4)
    parser.add_argument('--loss_type',
                        help='either maximize generator loss with \'standard\' or minimize with \'modified\'',
                        type=str,
                        default='standard')
    # default='original')

    args = parser.parse_args()
    main(args)
