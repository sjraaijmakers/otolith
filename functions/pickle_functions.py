import pickle
import os
import cv2


def save_pickle(data, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def open_pickle(filename):
    with open(filename, 'rb') as handle:
        return pickle.load(handle)
