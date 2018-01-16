import os
from invoke import task
import cv2
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from dotenv import load_dotenv, find_dotenv
from camera.data import list_dirs_in, list_images_in

load_dotenv(find_dotenv())

@task
def extract_intra_features(_):
    image_id = 0
    threshold = 3
    quantization = 2
    data_dir = os.environ['DATA_DIR']
    output_dir = os.path.join(data_dir, 'intra_channel_features')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_dir + '/intra_channel_features.csv', 'a') as output_file:
        for image_path in tqdm(list_images_in(data_dir + '/transformed_patches')):
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

            # 0. Extract information from filename
            name = os.path.splitext(os.path.basename(image_path))[0]
            label, image_id, transform, _ = name.split('_')

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

            # 3. Calculate co-occurrence matrix
            residual_dimension = 2 * threshold + 1
            patterns = np.zeros((residual_dimension, residual_dimension, residual_dimension), dtype=np.int)
            # TODO AS: Performance eater
            np.add.at(patterns, [diff[::2, ::2], diff[::2, 1::2], diff[1::2, 1::2]], 1)
            patterns = patterns / (diff.shape[0] * diff.shape[1] / 4)

            # 4. Append information to the csv
            features = np.append(patterns.reshape(-1), [transform != 'unalt', image_id, label])
            np.savetxt(output_file, [features], delimiter=',', fmt='%s')

@task
def extract_inter_features(_):
    threshold = 3
    quantization = 2
    data_dir = os.environ['DATA_DIR']
    output_dir = os.path.join(data_dir, 'inter_channel_features')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_dir + '/inter_channel_features.csv', 'a') as output_file:
        for image_path in tqdm(list_images_in(data_dir + '/transformed_patches')):
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

            # 0. Extract information from filename
            name = os.path.splitext(os.path.basename(image_path))[0]
            label, image_id, transform, _ = name.split('_')

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

            # 3. Calculate co-occurrence matrix
            residual_dimension = 2 * threshold + 1
            patterns = np.zeros((residual_dimension, residual_dimension, residual_dimension), dtype=np.int)
            # TODO AS: Performance eater
            np.add.at(patterns, [diff[::2, ::2, 0], diff[::2, 1::2, 0], diff[0::2, 1::2, 1]], 1)
            np.add.at(patterns, [diff[1::2, 1::2, 0], diff[::2, 1::2, 0], diff[0::2, 1::2, 1]], 1)
            patterns = patterns / (diff.shape[0] * diff.shape[1] / 4)

            # 4. Append information to the csv
            features = np.append(patterns.reshape(-1), [transform != 'unalt', image_id, label])
            np.savetxt(output_file, [features], delimiter=',', fmt='%s')

@task
def predict(_):
    data_dir = os.environ['DATA_DIR']

    intra_channel_features = pd.read_csv(data_dir + '/intra_channel_features/intra_channel_features.csv')
    inter_channel_features = pd.read_csv(data_dir + '/inter_channel_features/inter_channel_features.csv')

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
        'n_estimators': 10
    }

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))

from camera.transforms import adjust_gamma
from camera.transforms import flip_horizontally
from camera.transforms import jpeg_compress
from camera.transforms import resize

@task
def transform_and_crop(_):
    crop_size = 512
    data_dir = os.environ['DATA_DIR']
    output_dir = os.path.join(data_dir, 'transformed_patches')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    transforms = {
        'unalt': lambda image: image,
        'flip': flip_horizontally,
        # 'gamma80': lambda image: adjust_gamma(image, 0.8),
        # 'gamma120': lambda image: adjust_gamma(image, 1.2),
        # 'jpeg70': lambda image: jpeg_compress(image, 70),
        # 'jpeg90': lambda image: jpeg_compress(image, 90),
        # 'resize50': lambda image: resize(image, 0.5),
        # 'resize80': lambda image: resize(image, 0.8),
        # 'resize150': lambda image: resize(image, 1.5),
        # 'resize200': lambda image: resize(image, 2.0)
    }

    for label in tqdm(list_dirs_in(data_dir + '/train')):
        image_id = 0
        for image_path in tqdm(list_images_in(data_dir + '/' + label)):
            image = cv2.imread(image_path)
            for transform_name, transform in transforms.items():
                transformed_image = transform(image)
                # TODO AS: Alternative cropping strategies
                center_x = image.shape[0] // 2 - 1
                center_y = image.shape[1] // 2 - 1
                top_x, top_y = center_x - crop_size // 2, center_y - crop_size // 2
                patch_id = 0
                patch = transformed_image[top_x:top_x + crop_size, top_y:top_y + crop_size]
                filename = f'{label}_{image_id}_{transform_name}_{patch_id}.png'
                cv2.imwrite(os.path.join(output_dir, filename), patch)
            image_id += 1

@task
def download(ctx):
    competition = os.environ['KAGGLE_COMPETITION']
    username = os.environ['KAGGLE_USERNAME']
    password = os.environ['KAGGLE_PASSWORD']
    data_dir = os.environ['DATA_DIR']

    with ctx.cd(data_dir):
        ctx.run(f'kg download -c {competition} -u {username} -p {password}', pty=True)
        ctx.run(f'unzip test.zip -d {data_dir}')
        ctx.run(f'unzip train.zip -d {data_dir}')
        ctx.run(f'unzip sample_submission.csv.zip -d {data_dir}')
        ctx.run('rm -f test.zip train.zip sample_submission.csv.zip')
