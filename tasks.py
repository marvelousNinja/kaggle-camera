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

    for label in labels:
        # TODO AS: Weird error handling from pyinvoke, catching to continue for now
        with ctx.cd(f'{data_dir}/flickr/{label}'):
            try:
                ctx.run(f'wget --content-on-error --show-progress --tries=10 --max-redirect 0 -nc -i urls')
            except Exception as e:
                print(e)

        with ctx.cd(f'{data_dir}/reviews/{label}'):
            try:
                ctx.run(f'wget --content-on-error --show-progress --tries=10 --max-redirect 0 -i urls')
            except Exception as e:
                print(e)

@task
def whitelist_extras(ctx):
    data_dir = os.environ['DATA_DIR']

    labels_to_patterns = {
        'HTC-1-M7': [
            'JPEG 95 1520x2688',
            'JPEG 95 2688x1520'
        ],
        'Motorola-X': [
            'JPEG 95 3120x4160',
            'JPEG 95 4160x3120'
        ],
        'iPhone-4s':[
            'JPEG 96 3264x2448'
        ],
        'LG-Nexus-5x': [
            'JPEG 95 3024x4032',
            'JPEG 95 4032x3024'
        ],
        'Samsung-Galaxy-Note3': [
            'JPEG 96 4128x2322'
        ],
        'iPhone-6': [
            'JPEG 93 3264x2448',
            'JPEG 96 3264x2448'
        ],
        'Motorola-Droid-Maxx': [
            'JPEG 95 2432x4320',
            'JPEG 95 4320x2432'
        ],
        'Samsung-Galaxy-S4': [
            'JPEG 96 4128x2322',
            'JPEG 98 4128x2322'
        ],
        'Motorola-Nexus-6': [
            'JPEG 95 1040x780',
            'JPEG 95 3088x4130',
            'JPEG 95 3088x4160',
            'JPEG 95 3120x4160',
            'JPEG 95 4160x3088',
            'JPEG 95 4160x3120'
        ],
        'Sony-NEX-7': [
            'JPEG 92 6000x4000',
            'JPEG 95 6000x4000'
        ]
    }

    for label, patterns in labels_to_patterns.items():
        with ctx.cd(f'{data_dir}/flickr/{label}'):
            pattern = '|'.join(patterns)
            ctx.run('> _whitelist')
            ctx.run(f'identify -format "%f %m %Q %G\n" *.jpg* | grep -E "{pattern}" | cut -d " " -f 1 >> _whitelist', echo=True)

        with ctx.cd(f'{data_dir}/reviews/{label}'):
            pattern = '|'.join(patterns)
            ctx.run('> _whitelist')
            ctx.run(f'identify -format "%f %m %Q %G\n" *.jpg* | grep -E "{pattern}" | cut -d " " -f 1 >> _whitelist', echo=True)
