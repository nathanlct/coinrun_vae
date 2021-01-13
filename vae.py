import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()

        # dimensions
        self.input_dim = 3  # RGB 
        self.hidden_dims = [32, 64, 128, 256, 512]
        self.latent_dim = latent_dim
        self.dims = [self.input_dim] + self.hidden_dims + [self.latent_dim]

        # encoder
        self.encoder = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(in_channels=self.dims[i], out_channels=self.dims[i+1],
                          kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(self.dims[i+1]),
                nn.LeakyReLU()
            ) for i in range(len(self.hidden_dims))]
        )
        self.fc_mu = nn.Linear(4 * self.hidden_dims[-1], self.latent_dim)
        self.fc_var = nn.Linear(4 * self.hidden_dims[-1], self.latent_dim)

        # decoder
        self.fc_decoder = nn.Linear(self.latent_dim, 4 * self.hidden_dims[-1])
        self.decoder = nn.Sequential(
            *[nn.Sequential(
                nn.ConvTranspose2d(in_channels=self.hidden_dims[i+1], out_channels=self.hidden_dims[i],
                                   kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(self.hidden_dims[i]),
                nn.LeakyReLU()              
            ) for i in reversed(range(len(self.hidden_dims) - 1))],

            nn.ConvTranspose2d(in_channels=self.hidden_dims[0], out_channels=self.hidden_dims[0],
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(self.hidden_dims[0]),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=self.hidden_dims[0], out_channels=self.input_dim,
                      kernel_size=3, padding=1),
            nn.Tanh()
        )

    def loss(self, input, reconstruction, mu, log_var, epoch, total_epochs):
        """
        Loss = reconstruction loss (MSE) + regularisation loss (KL divergence)
        """
        reconstruction_loss = F.mse_loss(reconstruction, input)

        regularisation_loss = - 0.5 * (1 + log_var - torch.exp(log_var) - torch.square(mu))
        regularisation_loss = torch.mean(regularisation_loss) / 10

        beta = 0.5 + (30 - 0.5) * epoch / total_epochs

        loss = reconstruction_loss + beta * regularisation_loss

        return {'loss': loss, 
                'reconstruction_loss': reconstruction_loss,
                'regularisation_loss': regularisation_loss,
                'regularisation_loss_beta': beta * regularisation_loss,
                'beta': beta}

    def encode(self, input):
        """
        Encode input of shape (batch_size, n_channels, height, width)
        into mean and logvar components, both of shape (batch_size, latent_dim)
        """
        out = self.encoder(input)
        out = torch.flatten(out, start_dim=1)
        mu = self.fc_mu(out)
        log_var = self.fc_var(out)
        return [mu, log_var]

    def decode(self, z):
        """
        Decode latent input of shape (batch_size, latent_dim) into image of
        shape (batch_size, n_channels, height, width), scaled between -1 and 1
        """
        out = self.fc_decoder(z)
        out = out.view(-1, 512, 2, 2)
        out = self.decoder(out)
        return out

    def reparameterize(self, mu, logvar):
        """
        reparameterization trick to sample from N(mu, var) using N(0, 1)
        input: mu and logvar each of dim (batch_size, latent_dim)
        output: latent of dim (batch_size, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        """
        forward pass
        takes an input of dim (batch_size, n_channels, height, width)
        returns a reconstruction of shape (batch_size, n_channels, height, width)
        as well as mu and logvar each of shape (batch_size, latent_dim)
        """
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def sample(self, num_samples, current_device):
        """
        samples num_samples from the latent space and returns the corresponding
        reconstructions, of shape (num_samples, height, width)
        also takes the current device (cpu/gpu)
        """
        z = torch.randn(num_samples, self.latent_dim).to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x):
        """
        takes an input of dim (batch_size, n_channels, height, width)
        returns a reconstruction of shape (batch_size, n_channels, height, width)
        """
        return self.forward(x)[0]
