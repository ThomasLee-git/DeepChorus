from pathlib import Path

from preprocess.extract_spectrogram import extract_features
from network.DeepChorus import create_model
from evaluator import scale_result, median_filter
from constant import SHAPE, CHUNK_SIZE

import tensorflow as tf
import numpy as np
import tgre


def bin2interval(bin_result: np.ndarray) -> list:
    result = list()
    last_val = None
    count = 0
    for v in bin_result:
        if last_val is None:
            last_val = v
            count = 1
        else:
            if last_val == v:
                count += 1
            else:
                # add new interval
                result.append((last_val, count))
                last_val = v
                count = 1
    # add last interval
    result.append((v, count))
    return result


def result2textgrid(interval_result: list) -> tgre.TextGrid:
    tmp_time = [0] + [x[1] for x in interval_result]
    cumsum_time = np.cumsum(tmp_time)
    xmin = cumsum_time[0]
    xmax = cumsum_time[-1]

    # add tiers
    interval_list = list()
    for idx, (label, count) in enumerate(interval_result):
        tmp_interval = tgre.Interval(
            xmin=cumsum_time[idx], xmax=cumsum_time[idx + 1], text=str(label)
        )
        interval_list.append(tmp_interval)
    tier_list = list()
    tmp_tier = tgre.IntervalTier(
        name="chorus", xmin=xmin, xmax=xmax, items=interval_list
    )
    tier_list.append(tmp_tier)
    textgrid = tgre.TextGrid(xmin=xmin, xmax=xmax, tiers=tier_list)
    return textgrid


def result2texgridfile(interval_result, path: str) -> None:
    textgrid = result2textgrid(interval_result)
    textgrid.to_praat(path)
    return


def init_model():
    model_file = "model/Deepchorus_2021.h5"
    model = create_model(input_shape=SHAPE, chunk_size=CHUNK_SIZE)
    model.load_weights(model_file)
    model.summary()
    return model


def get_features(audio_path: str):
    features = extract_features(audio_path)
    num_frames = features.shape[1]
    num_frames_to_pad = (-num_frames) % 9
    features_to_pad = np.zeros((128, num_frames_to_pad, 1))
    features = np.concatenate((features, features_to_pad), axis=1)
    features = np.expand_dims(features, axis=0)
    return features


def evaluate():
    model = init_model()

    @tf.function(experimental_relax_shapes=True)
    def predict(t):
        return model(t)

    audio_path = Path("86a7fc02850ee03a5e622eee709dd03b.wav")
    features = get_features(audio_path.as_posix())
    print(features.shape)
    # eval result, return the first sample
    result = predict(features)[0]
    result = scale_result(result)
    result = median_filter(result, 9)
    # binarize
    bin_result = np.logical_not(result < 0.5)
    interval_result = bin2interval(bin_result)
    result2texgridfile(interval_result, audio_path.stem + "_deepchorus.textgrid")
    print("done")


if __name__ == "__main__":
    evaluate()
