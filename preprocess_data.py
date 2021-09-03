import numpy as np
import pypianoroll
from numpy import asarray
from numpy import save
from numpy import load
import os


def parse_midi_to_pianoroll(dir, beat_resolution, measure_resolution, number_of_measures, start_offset=0, singletrack=True, track_names=[]):
    song_paths = os.listdir(dir)
    data_unprocessed = []
    data_prev_unprocessed = []
    for path in song_paths:
        try:
            multitrack = pypianoroll.read(dir + path)
        except:
            print("Could't read track " + path)
            continue
        try:
            multitrack.set_resolution(beat_resolution)
            multitrack.binarize()
            if singletrack:
                multitrack.tracks = [multitrack.tracks[0]]
            else:
                track_appended = False
                selected_track = []
                for track in multitrack.tracks:
                    if len(list(filter(lambda x: x.lower() in track.name, track_names))) > 0 and not track_appended:
                            selected_track.append(track)
                            track_appended = True
                    if len(selected_track) == 0:
                        continue
                    multitrack.tracks = selected_track
            pianoroll = multitrack.stack()
            m = pianoroll[:,start_offset:number_of_measures * measure_resolution + start_offset]
            bars = np.hsplit(m,number_of_measures)
            # shift bars for 1
            bars_prev = bars[-1:] + bars[:-1]
            bars_prev[0] = np.zeros(bars_prev[0].shape)
            for x in bars:
                data_unprocessed.append(x)
            for x_prev in bars_prev:
                data_prev_unprocessed.append(x_prev)
            print("File {0} processed".format(path))
        except Exception as e:
            print(e)
            print("Error processing file at path " + path)
            continue
    print("Finished processing data.")
    print("Length of x: {0}".format(len(data_unprocessed)))
    print("Length of x_prev: {0}".format(len(data_prev_unprocessed)))
    return data_unprocessed, data_prev_unprocessed


def process_pianoroll(data):
    # check bar shapes in pianoroll
    data = [bar for bar in data if bar.shape == (1, 16, 128)]
    # convert list to np array
    data = np.stack(data)
    return data


def save_data(path, data):
    p = asarray(data)
    save(path, p)


def load_data(path):
    data = load(path)
    return data
