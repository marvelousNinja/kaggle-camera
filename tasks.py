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

# TODO AS: Fails after first label
# Do it manually for now
# wget -nc --tries=10 -q --show-progress --max-redirect 0 -i urls
@task
def download_extras(ctx):
    data_dir = os.environ['DATA_DIR']
    ctx.run(f'cp -R extra {data_dir}')
    ctx.run(f'cp -R validation {data_dir}')

    labels = [
        'HTC-1-M7',
        'Motorola-X',
        'iPhone-4s',
        'LG-Nexus-5x',
        'Samsung-Galaxy-Note3',
        'iPhone-6',
        'Motorola-Droid-Maxx',
        'Samsung-Galaxy-S4',
        'Motorola-Nexus-6',
        'Sony-NEX-7'
    ]

    with ctx.cd(data_dir):
        for label in labels:
            ctx.run(f'wget -q --show-progress -nc --tries=10 --max-redirect 0 -i ./extra/{label}/urls -P ./extra/{label}', pty=True)
            ctx.run(f'wget -q --show-progress --tries=10 --max-redirect 0 -i ./validation/{label}/urls -P ./validation/{label}', pty=True)
