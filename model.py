import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

# custom layer for convolutions with batch normalization and lrelu activation
class ConvolutionLayer2d(nn.Module):
    def __init__(self, channels_in, channels_out, kernel, stride, padding):
        super(ConvolutionLayer2d, self).__init__()
        self.conv = nn.Conv2d(channels_in, channels_out, kernel, stride, padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(channels_out)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace = False)
    
    def forward(self,x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.leaky_relu(x)
        return x

# custom layer for transposed convolutions with batch normalization and relu activation
class ConvolutionTransposeLayer2d(nn.Module):
    def __init__(self, channels_in, channels_out, kernel, stride, padding):
        super(ConvolutionTransposeLayer2d, self).__init__()
        self.conv = nn.ConvTranspose2d(channels_in, channels_out, kernel, stride, padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(channels_out)
        self.relu = nn.ReLU(inplace = False)
    
    def forward(self, x, normalize=True, activation=None):
        x = self.conv(x)
        if normalize:
            x = self.batch_norm(x)
        if activation != None:
            x = activation(x)
        return x

# linear layer with batch normalization and relu activation for generator
class LinearLayer(nn.Module):
    def __init__(self, linear_in, linear_out):
        super(LinearLayer, self).__init__()
        self.linear = nn.Linear(linear_in, linear_out)
        self.batch_norm = nn.BatchNorm1d(linear_out)
        self.relu =  nn.ReLU()
    
    def forward(self, x):
        x = self.linear(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x

# generator
class Generator(nn.Module):
    def __init__(self, nz=100, pitches=128):
        super(Generator,self).__init__()
        self.nz = nz
        self.pitches = pitches
        self.filters_number = 256
        self.filters_number_conditioner = 256
        self.transpose_filters_number = 512
        
        self.linear0 = LinearLayer(nz, 1024)
        self.linear1 = LinearLayer(1024,512)
        
        # Generator transposed convolutions layers
        self.hidden0 = ConvolutionTransposeLayer2d(channels_in=self.transpose_filters_number, channels_out=self.filters_number, kernel=(2,1), stride=2, padding=0)
        self.hidden1 = ConvolutionTransposeLayer2d(channels_in=self.transpose_filters_number, channels_out=self.filters_number, kernel=(2,1), stride=2, padding=0)
        self.hidden2 = ConvolutionTransposeLayer2d(channels_in=self.transpose_filters_number, channels_out=self.filters_number, kernel=(2,1), stride=2, padding=0)
        self.hidden3 = ConvolutionTransposeLayer2d(channels_in=self.transpose_filters_number, channels_out=1, kernel=(1,pitches), stride=(1,2), padding=0)
        
        # Conditioner convolution layers
        self.condition_hidden0 = ConvolutionLayer2d(channels_in=1, channels_out=self.filters_number_conditioner, kernel=(1,128), stride=(1,2), padding=0)
        self.condition_hidden1 = ConvolutionLayer2d(channels_in=self.filters_number_conditioner, channels_out=self.filters_number_conditioner, kernel=(2,1), stride=2, padding=0)
        self.condition_hidden2 = ConvolutionLayer2d(channels_in=self.filters_number_conditioner, channels_out=self.filters_number_conditioner, kernel=(2,1), stride=2, padding=0)
        self.condition_hidden3 = ConvolutionLayer2d(channels_in=self.filters_number_conditioner, channels_out=self.filters_number_conditioner, kernel=(2,1), stride=2, padding=0)
        
        self.sigmoid = nn.Sigmoid()
            
    def forward(self, z, prev_x):
        batch_size = prev_x.shape[0]
        
        condition_hidden0 = self.condition_hidden0(prev_x) 
        condition_hidden1 = self.condition_hidden1(condition_hidden0)   
        condition_hidden2 = self.condition_hidden2(condition_hidden1)
        condition_hidden3 = self.condition_hidden2(condition_hidden2)
        
        z = z.view(batch_size,-1)
        
        h0 = self.linear0(z)
        h1 = self.linear1(h0)
        
        h1 = h1.view(batch_size, 256, 2, 1)
        h1 = conv_prev_concat(h1, condition_hidden3)

        h2 = self.hidden0(h1)
        h2 = conv_prev_concat(h2, condition_hidden2)
        
        h3 = self.hidden1(h2)
        h3 = conv_prev_concat(h3, condition_hidden1)
        
        h4 = self.hidden2(h3)
        h4 = conv_prev_concat(h4, condition_hidden0)
        
        x = self.hidden3(h4, normalize=False, activation=nn.Sigmoid())
                
        return x
        
# discriminator
class Discriminator(nn.Module):
    def __init__(self, pitches=128):
        super(Discriminator,self).__init__()
        self.pitches = pitches
        self.linear_in = 231 #77
        
        self.hidden0 = ConvolutionLayer2d(channels_in=1, channels_out=27, kernel=(2,128), stride=2, padding=0)
        #self.hidden2 = ConvolutionLayer2d(channels_in=256, channels_out=256, kernel=(3,1), stride=2, padding=0)
        self.hidden1 = ConvolutionLayer2d(channels_in=27, channels_out=77, kernel=(4,1), stride=2, padding=0)
        self.linear = nn.Linear(self.linear_in, 1024)
        self.linear2 = nn.Linear(1024,1)
        self.sigmoid = nn.Sigmoid()
        self.lrelu = nn.LeakyReLU()
        
    def forward(self, x):
        batch_size = x.shape[0]
        h0 = self.hidden0(x)
        fm = h0
        #h1 = self.hidden2(h0)
        h1 = self.hidden1(h0)
        h1 = h1.view(batch_size,-1)
        l = self.linear(h1)
        l = self.lrelu(l)
        out = self.linear2(l)
        out_sigmoid = self.sigmoid(out)
        return out_sigmoid, out, fm                                                                
    
            
                           
    
def train(netD, netG, optimizerG, optimizerD, data_loader, epochs, criterion, nz = 100, n_g_train=2, lamda1=0.01, lamda2=0.1, device="cpu"):
    netG.train()
    netD.train()

    netG.to(device)
    netD.to(device)
        
    G_losses = []
    D_losses = []

    for epoch in range(epochs):
        for i, (X, X_prev) in enumerate(data_loader, 0):        
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            batch_size = X.size(0)
            output_real, logits_real, fm = netD(X)
            real_label = torch.ones_like(logits_real) * 0.9 
            errD_real = reduce_mean(criterion(logits_real,real_label))
            D_x = output_real.mean().item()
            
            ## Train with all-fake batch
            noise = torch.rand(batch_size, nz, device=device)
            fake = netG(noise,X_prev)
            output_fake, logits_fake, fm_ = netD(fake.detach())
            fake_label = torch.zeros_like(logits_fake)
            errD_fake = reduce_mean(criterion(logits_fake,fake_label))
            D_G_z1 = output_fake.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            errD.backward(retain_graph=True)
            # Update D
            optimizerD.step()
        
            ############################
            # (2) Update G network: 
            # maximize log(D(G(z))) + lamda1 * l2_loss(fake_image,real_image) + lamda2 * l2_loss(fm_fake,fm_real)
            # Update G n_g_train times to make Discriminator weaker
            ###########################
            for _ in range(0,n_g_train):
                _,_,fm_r = netD(X)
                optimizerG.zero_grad()
                fake = netG(noise,X_prev)
                output, logits, fm_ = netD(fake)
                D_G_z2 = output.mean().item()
                real_label = torch.ones_like(logits)
                g_loss_fake = reduce_mean(criterion(logits, real_label))
                # Feature matching 
                mean_fake_image = torch.mean(fake,0)
                mean_real_image = torch.mean(X, 0)
                g_loss_image = l2_loss(mean_fake_image, mean_real_image)
                g_loss_image = torch.mul(g_loss_image, lamda1)
                # Feature matching based on first convolution output
                mean_fm_real = torch.mean(fm_r,0)
                mean_fm_fake = torch.mean(fm_,0)
                g_loss_fm = l2_loss(mean_fm_fake, mean_fm_real)
                g_loss_fm = torch.mul(g_loss_fm, lamda2)
                # Compute error of G as sum of criterion loss and feature matching loss
                errG = g_loss_fake + g_loss_image + g_loss_fm
                errG.backward(retain_graph=True)
                # Update G
                optimizerG.step()
            
            
            if i % 15 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, epochs, i, len(data_loader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
    
    print("Training finished.")
    # save training losses image to disk
    save_losses_image(G_losses, D_losses)
    # save models to disk
    torch.save(netD.state_dict(), "./checkpoint/discriminator.pth")
    torch.save(netG.state_dict(), "./checkpoint/generator.pth")