"""Yandex and Flickr scrappers"""
import os
from functools import partial
from multiprocessing.pool import ThreadPool
from urllib.request import urlopen

import wget
from bs4 import BeautifulSoup
from dotenv import find_dotenv
from dotenv import load_dotenv
from fire import Fire
from flickrapi import FlickrAPI
from retry.api import retry_call
from tqdm import tqdm

load_dotenv(find_dotenv())

def download_image(directory, url):
    filename = url.split('/')[-1]

    if filename[-4:] == 'orig':
        filename = filename + '.jpeg'

    local_path = os.path.join(directory, filename)
    if os.path.isfile(local_path):
        return None

    try:
        retry_call(lambda: wget.download(url, local_path, bar=None), tries=2, delay=1)
        return local_path
    except Exception as e: # pylint: disable=broad-except
        print(e)

def scrap_flickr(label, api_key, secret, page):
    label_to_params = {
        'Motorola-Droid-Maxx': {'camera': 'motorola/droid_ultra'},
        'HTC-1-M7': {'camera': 'htc/one'},
        'iPhone-4s': {'camera': 'apple/iphone_4s'},
        'iPhone-6': {'camera': 'apple/iphone_6'},
        'Sony-NEX-7': {'camera' : 'sony/nex-7'},
        'Samsung-Galaxy-S4': {'camera': 'samsung/galaxy_s4'},
        'Samsung-Galaxy-Note3': {'camera': 'samsung/galaxy-note-3'},
        'Motorola-Nexus-6': {'camera': 'motorola/nexus_6'},
        'LG-Nexus-5x': {'text': 'nexus 5x', 'min_taken_date': 1443646800},
        'Motorola-X': {'text': 'motorola moto x', 'min_taken_date': 1409518800}
    }

    params = {
        'extras': 'url_o',
        'per_page': 500,
        'page': page,
        **label_to_params[label]
    }

    flickr = FlickrAPI(api_key, secret, format='parsed-json')
    response = flickr.photos.search(**params)
    photos = response['photos']['photo']
    urls = [photo.get('url_o', None) for photo in photos]
    return list(filter(None, urls))

def scrap_yandex(model, product_id, page):
    if product_id:
        url = f'https://fotki.yandex.ru/search.xml?modelid={product_id}&p={page}&how=created'
    else:
        url = f'https://fotki.yandex.ru/search.xml?text={model}&p={page}&type=model&how=created'

    html = urlopen(url).read()
    soup = BeautifulSoup(html, 'html.parser')
    photo_links = soup.select('a.preview-link img')
    urls = list(map(lambda link: link['src'], photo_links))
    return list(map(lambda url: url[:-2] + 'orig', urls))

def scrap(
        label=None, data_dir=os.environ['DATA_DIR'],
        api_key=os.environ['FLICKR_API_KEY'], secret=os.environ['FLICKR_SECRET'],
        page_from=0, page_to=1, model=None, product_id=None
    ): # pylint: disable=too-many-arguments

    label_directory = os.path.join(data_dir, 'scrapped', label)
    if not os.path.isdir(label_directory):
        os.makedirs(label_directory)

    urls = list()
    for page in range(page_from, page_to):
        if model or product_id:
            urls.extend(scrap_yandex(model, product_id, page))
        else:
            urls.extend(scrap_flickr(label, api_key, secret, page))

    pool = ThreadPool()
    for local_path in tqdm(pool.imap(partial(download_image, label_directory), urls), total=len(urls)):
        if local_path: tqdm.write(f'New image saved {local_path}')

if __name__ == '__main__':
    Fire(scrap)
