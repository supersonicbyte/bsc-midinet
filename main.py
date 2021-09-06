import argparse
from model import *
from utils import *
from configuration import *
from preprocess_data import *
from dataloader import *
from torch.utils.data import DataLoader
from random import randrange


parser = argparse.ArgumentParser(description="Select mode: 0 - for data preparation, 1 - training, 2 - generating samples")
parser.add_argument("mode",type=int)
args = parser.parse_args()



def main():
    mode = args.mode
    if mode == 0:
        get_data()
    elif mode == 1:
        train_model()
    else:
        sample()


def get_data():
    data, data_prev = parse_midi_to_pianoroll(dir=DATAPATH, 
                                             beat_resolution=BEAT_RESOLUTION,
                                            measure_resolution=MEASURE_RESOLUTION,
                                            number_of_measures=NUMBER_OF_MEASURES,
                                            start_offset=START_OFFSET)
    data = process_pianoroll(data)
    data_prev = process_pianoroll(data_prev)
    save_data("./" + PROCESSED_DATA_PREFIX + "processed_x.npy", data)
    save_data("./" + PROCESSED_DATA_PREFIX + "processed_x_prev.npy", data_prev)
    print("Data processed and saved to disk.")

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    netG = Generator(nz=NZ)
    netD = Discriminator()

    optimizerG = torch.optim.Adam(netG.parameters(), lr=LR, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=LR, betas=(0.5, 0.999))

    criterion = nn.BCEWithLogitsLoss()

    x = load_data("./" + PROCESSED_DATA_PREFIX + "processed_x.npy")
    x_prev = load_data("./" + PROCESSED_DATA_PREFIX + "processed_x_prev.npy")

    dataset = BarDataset(data=x, data_prev=x_prev, device=device)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    train(netD=netD,
          netG=netG,
          optimizerD=optimizerD,
          optimizerG=optimizerG,
          epochs=EPOCHS,
          criterion=criterion,
          nz=NZ,
          n_g_train=N_G_TRAIN,
          device=device,
          data_loader=data_loader)


def sample():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    netG = Generator(nz=NZ)
    netG.load_state_dict(torch.load("./checkpoint/generator.pth"))
    x_prev = load_data("./" + PROCESSED_DATA_PREFIX + "processed_x_prev.npy")
    random_sample_index = randrange(x_prev.shape[0] // BATCH_SIZE)
    priming_melody = torch.from_numpy(x_prev[random_sample_index:random_sample_index + NUMBER_OF_PRIMING_BARS])
    priming_melody = priming_melody.type(torch.FloatTensor)
    netG.to(device)
    priming_melody = priming_melody.to(device)
    outputs = []
    for i in range(0,10):
        noise = torch.randn(NUMBER_OF_PRIMING_BARS, NZ).to(device)
        if i == 0:
            outputs.append(netG(noise,priming_melody))
        else:
            outputs.append(netG(noise,outputs[i - 1]))

    outputs = np.array(outputs)
    melody = [outputs[i].detach().cpu().numpy() for i in range(0,10)]
    melody = np.stack(melody)
    melody = np.concatenate(melody, axis=0)
    m = melody[0]
    for i in range(1,melody.shape[0]):
        m = np.concatenate((m,melody[i]),axis=1)
    m = m[0,:,0:127]
    m[m > 0.6] = 1
    m = m == 1
    melody_track = pypianoroll.BinaryTrack(pianoroll = m)
    multi_track = pypianoroll.Multitrack(resolution=24, tracks=[melody_track])
    pypianoroll.write('./sample.mid', multi_track)
    print("Generated midi file {0}".format("sample.mid"))
    

if __name__ == "__main__" :
    main()