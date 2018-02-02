import os
from invoke import task
from tqdm import tqdm
import numpy as np
from dotenv import load_dotenv, find_dotenv
from camera.experiments.residual_cnn import conduct as conduct_residual_cnn

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

@task()
def experiment_residual_cnn(
        ctx,
        learning_rate=0.0001,
        optimizer='adam',
        callbacks='reduce_lr',
        crop_size=32,
        n_epochs=200,
        batch_size=16,
        crop_strategy='crop_random',
        transform_strategy='random',
        residual_filter_strategy='spam_11_5',
        overfit_run=False,
        network='mobile_net',
        verbose=0
    ):
    data_dir = os.environ['DATA_DIR']
    conduct_residual_cnn(
        data_dir,
        learning_rate,
        optimizer,
        callbacks.split(','),
        crop_size,
        n_epochs,
        batch_size,
        crop_strategy,
        transform_strategy,
        residual_filter_strategy,
        overfit_run,
        network,
        verbose
    )
