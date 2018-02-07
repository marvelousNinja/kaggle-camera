import os
from functools import partial
from multiprocessing.pool import ThreadPool
from datetime import datetime
from urllib.request import urlopen
from tqdm import tqdm
import numpy as np
from fire import Fire
from flickrapi import FlickrAPI
from dotenv import load_dotenv, find_dotenv
import wget
from retry.api import retry_call
from bs4 import BeautifulSoup
load_dotenv(find_dotenv())

def download_image(directory, url):
    filename = url.split('/')[-1]

    if filename[-4:] == 'orig':
        filename = filename + '.jpeg'

    local_path = os.path.join(directory, filename)
    if os.path.isfile(local_path):
        return

    try:
        retry_call(lambda: wget.download(url, local_path, bar=None), tries=5, delay=2)
        return local_path
    except Exception as e:
        print(e)

def scrap_flickr(label, api_key, secret, page):
    label_to_camera = {
        # TODO AS: Make sure camera is the same
        # 'Motorola-Droid-Maxx': 'motorola/moto_maxx' # only 9 images
        'Motorola-Droid-Maxx': 'motorola/droid_ultra', # lots
        # TODO AS: Make sure camera is the same
        # 'HTC-1-M7': 'htc/one_m7', # 150 somewhat images
        'HTC-1-M7': 'htc/one', # lots
        'iPhone-4s': 'apple/iphone_4s', # lots
        'iPhone-6': 'apple/iphone_6', # lots
        'Sony-NEX-7': 'sony/nex-7', # lots
        'Samsung-Galaxy-S4': 'samsung/galaxy_s4', # lots
        'Samsung-Galaxy-Note3': 'samsung/galaxy-note-3', # lots
        'Motorola-Nexus-6': 'motorola/nexus_6', # lots
        #'Motorola-X': 'motorola/moto_x' # lots
    }

    if label == 'LG-Nexus-5x':
        params = {
            'text': 'nexus 5x',
            'extras': 'url_o',
            'per_page': 500,
            'page': page,
            # October of 2015, release date for 2nd Gen
            'min_taken_date': 1443646800
        }
    elif label == 'Motorola-X':
        params = {
            'camera': 'motorola/moto_x',
            'extras': 'url_o',
            'per_page': 500,
            'page': page,
            # September of 2014, release date
            'min_taken_date': 1409518800
        }
    else:
        params = {
            # cm=label_to_camera[label],
            'camera': label_to_camera[label],
            'extras': 'url_o',
            'per_page': 500,
            'page': page
            # TODO AS: Should I make results predictable and use pagination?
            # TODO AS: We can always fetch a random page
            # sort='date-posted-desc'
        }

    flickr = FlickrAPI(api_key, secret, format='parsed-json')
    response = flickr.photos.search(**params)

    print('Number of pages', response['photos']['pages'])
    photos = response['photos']['photo']

    urls = [photo.get('url_o', None) for photo in photos]
    return list(filter(None, urls))

def scrap_yandex(model, product_id, page):
    if product_id:
        html = urlopen(f'https://fotki.yandex.ru/search.xml?modelid={product_id}&p={page}').read()
    else:
        html = urlopen(f'https://fotki.yandex.ru/search.xml?text={model}&p={page}&type=model').read()

    soup = BeautifulSoup(html, 'html.parser')
    photo_links = soup.select('a.preview-link img')
    urls = list(map(lambda link: link['src'], photo_links))
    return list(map(lambda url: url[:-2] + 'orig', urls))

def scrap(
        label=None, data_dir=os.environ['DATA_DIR'],
        api_key=os.environ['FLICKR_API_KEY'], secret=os.environ['FLICKR_SECRET'],
        page_from=0, page_to=1, model=None, product_id=None
    ):

    label_directory = os.path.join(data_dir, 'scrapped', label)
    if not os.path.isdir(label_directory):
        os.makedirs(label_directory)

    urls = list()
    for page in range(page_from, page_to):
        if model:
            urls.extend(scrap_yandex(model, product_id, page))
        else:
            urls.extend(scrap_flickr(label, api_key, secret, page))

    pool = ThreadPool()
    for local_path in tqdm(pool.imap(partial(download_image, label_directory), urls), total=len(urls)):
        if local_path: tqdm.write(f'New image saved {local_path}')

if __name__ == '__main__':
    Fire(scrap)
