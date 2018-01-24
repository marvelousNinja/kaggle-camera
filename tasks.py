import os
from invoke import task
from tqdm import tqdm
import numpy as np
from dotenv import load_dotenv, find_dotenv
from camera.experiments.cnn_svm import conduct
from camera.experiments.residual import conduct as conduct_residual
from camera.experiments.demosaicing import conduct as conduct_demosaicing

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
def experiment_cnn(_):
    data_dir = os.environ['DATA_DIR']
    conduct(data_dir)

@task
def experiment_residual(_):
    data_dir = os.environ['DATA_DIR']
    conduct_residual(data_dir)

@task
def experiment_demosaicing(_):
    data_dir = os.environ['DATA_DIR']
    conduct_demosaicing(data_dir)
