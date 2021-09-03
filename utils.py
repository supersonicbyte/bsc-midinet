import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# concat condition vector to feature map axis
def conv_prev_concat(x, y):
        x_shapes = x.shape
        y_shapes = y.shape
        if x_shapes[2:] == y_shapes[2:]:
            y2 = y.expand(x_shapes[0],y_shapes[1],x_shapes[2],x_shapes[3])
            return torch.cat((x, y2),1)
        else:
            print("Error")
            print(x_shapes)
            print(y_shapes)

def reduce_mean(x):
    output = torch.mean(x,0, keepdim = False)
    output = torch.mean(output,-1, keepdim = False)
    return output

def l2_loss(x,y):
    return nn.MSELoss(reduction='sum')(x, y) / 2

def save_losses_image(G_losses, D_losses):
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('losses.png')