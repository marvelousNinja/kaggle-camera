import os
from invoke import task
from tqdm import tqdm
import numpy as np
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

@task
def clean(ctx):
    data_dir = os.environ['DATA_DIR']
    os.path.join(data_dir, 'models')
    ctx.run(f'rm {os.path.join(data_dir, "models")}/*')

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
