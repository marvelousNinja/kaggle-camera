import os
from invoke import task
import cv2
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed
from camera.shared.data import get_all_labels, get_image_paths
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

def process_patch(image, i, j, size, threshold):
    patch = image[i * size:(i + 1) * size, j * size:(j + 1) * size]
    residual_dimension = 2 * threshold + 1
    patterns = np.zeros((residual_dimension, residual_dimension, residual_dimension), dtype=np.int)
    # TODO AS: Performance eater
    np.add.at(patterns, [patch[::2, ::2], patch[::2, 1::2], patch[1::2, 1::2]], 1)
    patterns = patterns / (patch.shape[0] * patch.shape[1] / 4)
    return patterns.reshape(-1)

def process_red_green_patch(image, i, j, size, threshold):
    patch = image[i * size:(i + 1) * size, j * size:(j + 1) * size]
    residual_dimension = 2 * threshold + 1
    patterns = np.zeros((residual_dimension, residual_dimension, residual_dimension), dtype=np.int)
    # TODO AS: Performance eater
    np.add.at(patterns, [patch[::2, ::2, 0], patch[::2, 1::2, 0], patch[0::2, 1::2, 1]], 1)
    np.add.at(patterns, [patch[1::2, 1::2, 0], patch[::2, 1::2, 0], patch[0::2, 1::2, 1]], 1)
    patterns = patterns / (patch.shape[0] * patch.shape[1] / 4)
    return patterns.reshape(-1)

@task
def extract_red(ctx):
    image_id = 0
    size = 512
    threshold = 3
    quantization = 2
    delayed_process_patch = delayed(process_patch)

    with Parallel(n_jobs=-1, backend='threading') as parallel:
        with open('./extracted_features.csv', 'a') as output_file:
            for label in tqdm(get_all_labels()):
                for image_path in tqdm(get_image_paths(label)):
                    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
                    # try:
                    if (image.shape == ()):
                        # TODO AS: Some images have second thumbnail frame
                        continue

                    # 1. Interpolate
                    red_image = image[:, :, 0].astype(np.int16)
                    interpolated_image = np.array(red_image)
                    true_red = red_image[1::2, 0::2]
                    interpolated_image[0::2, 0::2] = true_red
                    interpolated_image[1::2, 1::2] = true_red
                    interpolated_image[0::2, 1::2] = true_red

                    # 2. Calculate difference, quantize and clip
                    diff = red_image - interpolated_image
                    diff //= quantization
                    # TODO AS: Performance eater
                    np.clip(diff, a_min=-threshold, a_max=threshold, out=diff)
                    diff += threshold

                    # 3. For each patch, calculate co-occurrence matrix
                    processed_patches = np.array(parallel([delayed_process_patch(diff, i, j, size, threshold) for i in range(diff.shape[0] // size) for j in range(diff.shape[1] // size)]))

                    # 4. Append information to the csv
                    df = np.c_[
                        processed_patches,
                        np.full(processed_patches.shape[0], image_id),
                        np.full(processed_patches.shape[0], label)]

                    np.savetxt(output_file, df, delimiter=',', fmt='%s')
                    # except Exception as err:
                    #     tqdm.write('Error on image {} with label {}: {}'.format(image_id, label, err))
                    image_id += 1

@task
def extract_red_green(ctx):
    image_id = 0
    size = 512
    threshold = 3
    quantization = 2
    delayed_process_red_green_patch = delayed(process_red_green_patch)

    with Parallel(n_jobs=-1, backend='threading') as parallel:
        with open('./red.green.extracted_features.csv', 'a') as output_file:
            for label in tqdm(get_all_labels()):
                for image_path in tqdm(get_image_paths(label)):
                    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
                    # try:
                    if (image.shape == ()):
                        # TODO AS: Some images have second thumbnail frame
                        continue

                    # 1. Interpolate
                    red_green_image = image[:, :, :2].astype(np.int32)
                    interpolated_image = np.array(red_green_image)

                    true_red = red_green_image[1::2, 0::2, 0]
                    interpolated_image[0::2, 0::2, 0] = true_red
                    interpolated_image[1::2, 1::2, 0] = true_red
                    interpolated_image[0::2, 1::2, 0] = true_red

                    true_green_1 = red_green_image[0::2, 0::2, 1]
                    true_green_2 = red_green_image[1::2, 1::2, 1]
                    interpolated_image[0::2, 1::2, 1] = true_green_1
                    interpolated_image[1::2, 0::2, 1] = true_green_2

                    # 2. Calculate difference, quantize and clip
                    diff = red_green_image - interpolated_image
                    diff //= quantization
                    # TODO AS: Performance eater
                    np.clip(diff, a_min=-threshold, a_max=threshold, out=diff)
                    diff += threshold

                    # 3. For each patch, calculate co-occurrence matrix
                    processed_patches = np.array(parallel([delayed_process_red_green_patch(diff, i, j, size, threshold) for i in range(diff.shape[0] // size) for j in range(diff.shape[1] // size)]))

                    # 4. Append information to the csv
                    df = np.c_[
                        processed_patches,
                        np.full(processed_patches.shape[0], image_id),
                        np.full(processed_patches.shape[0], label)]

                    np.savetxt(output_file, df, delimiter=',', fmt='%s')
                    # # except Exception as err:
                    # #     tqdm.write('Error on image {} with label {}: {}'.format(image_id, label, err))
                    image_id += 1


@task
def predict(ctx):
    intra_channel_features = pd.read_csv('./red.green.extracted_features.csv')
    inter_channel_features = pd.read_csv('./extracted_features.csv')

    image_id_column = inter_channel_features.columns[-2]
    target_column = inter_channel_features.columns[-1]

    image_id = inter_channel_features[image_id_column]
    target = inter_channel_features[target_column]

    intra_channel_features.drop([image_id_column, target_column], axis=1, inplace=True)
    inter_channel_features.drop([image_id_column, target_column], axis=1, inplace=True)

    full_data = pd.concat([intra_channel_features, inter_channel_features], axis=1)

    full_data['zero_count'] = (full_data == 0).astype(int).sum(axis=1)
    full_data['zero_count'] = (full_data == 0).astype(int).sum(axis=1)

    train_mask = image_id % 10 != 0

    X_train, y_train = full_data[train_mask], target[train_mask]
    X_test, y_test = full_data[~train_mask], target[~train_mask]

    params = {
        'n_estimators': 100
    }

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))

@task
def download(ctx):
    competition = os.environ['KAGGLE_COMPETITION']
    username = os.environ['KAGGLE_USERNAME']
    password = os.environ['KAGGLE_PASSWORD']
    data_dir = os.environ['DATA_DIR']
    ctx.run(f'kg download -c {competition} -u {username} -p {password}', pty=True)
    ctx.run(f'unzip test.zip -d {data_dir}')
    ctx.run(f'unzip train.zip -d {data_dir}')
    ctx.run('rm -f test.zip train.zip')
