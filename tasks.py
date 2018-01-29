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

@task
def experiment_residual_cnn(
        ctx,
        crop_size=32,
        n_epochs=100,
        batch_size=16,
        outer_crop_strategy='crop_center',
        inner_crop_strategy='crop_random',
        residual_filter_strategy='spam_11_5',
        test_limit=None,
        train_limit=None,
    ):
    data_dir = os.environ['DATA_DIR']
    conduct_residual_cnn(
        data_dir,
        crop_size,
        n_epochs,
        batch_size,
        outer_crop_strategy,
        inner_crop_strategy,
        residual_filter_strategy
        test_limit,
        train_limit
    )
