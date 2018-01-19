import os
from invoke import task
import cv2
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from dotenv import load_dotenv, find_dotenv
from camera.data import list_dirs_in, list_images_in
from camera.generators import train_test_holdout_split
from camera.generators import list_all_samples_in
from camera.generators import image_generator, parallel_image_generator
from camera.generators import center_crop
from camera.generators import sequential_crop
from camera.generators import random_crop
from camera.feature_extraction import intra_channel
from itertools import islice
from multiprocessing import Pool

load_dotenv(find_dotenv())

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

@task
def generate_test(_):
    data_dir = os.environ['DATA_DIR']
    labels_and_image_paths = list_all_samples_in(data_dir)
    _, test, _ = train_test_holdout_split(labels_and_image_paths)
    crop_size = 512

    test_data = list()
    for _, label, transform_name, image in tqdm(islice(parallel_image_generator(Pool(), test, crop_generator=center_crop, crop_size=crop_size), len(test)), total=len(test)):
        features = np.concatenate([
            intra_channel(image),
            [int(transform_name != 'unalt'), label]
        ])

        test_data.append(features)

    full_test = np.array(test_data)
    np.savetxt(data_dir + '/test.csv', full_test, fmt='%s', delimiter=',')

@task
def learn(_):
    data_dir = os.environ['DATA_DIR']
    labels_and_image_paths = list_all_samples_in(data_dir)
    train, test, holdout = train_test_holdout_split(labels_and_image_paths)
    crop_size = 512

    full_test = np.loadtxt(data_dir + '/test.csv', delimiter=',', dtype=np.object)
    X_test, y_test = full_test[:, :-1], full_test[:, -1]

    i = 0
    train_data = list()
    for _, label, transform_name, image in tqdm(parallel_image_generator(Pool(), train, crop_generator=sequential_crop, crop_size=crop_size, crop_limit=15)):
        features = np.concatenate([
            intra_channel(image),
            [int(transform_name != 'unalt'), label]
        ])

        train_data.append(features)

        i += 1
        if i % 500 == 0:
            model = RandomForestClassifier(n_estimators=50, n_jobs=-1)
            full_train = np.array(train_data)
            X_train, y_train = full_train[:, :-1], full_train[:, -1]
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            tqdm.write(classification_report(y_test, predictions))
