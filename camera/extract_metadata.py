import os
from functools import partial
from multiprocessing.pool import ThreadPool

import numpy as np
from dotenv import load_dotenv
from dotenv import find_dotenv
from fire import Fire
from shell import shell
from tqdm import tqdm

from camera.db import insert_multiple
from camera.db import find_by_path
from camera.db import remove
from camera.db import find_by
from camera.data import list_unencoded_samples

load_dotenv(find_dotenv())

def extract_single(dataset, path_and_label):
    path = path_and_label[0]
    label = path_and_label[1]

    outcome = shell(f'identify -format "%[EXIF:Make]__%[EXIF:Model]__%[EXIF:Software]__%m__%Q__%G" {path}')
    if len(outcome.errors()) > 0:
        tqdm.write(str(outcome.errors()))

        if len(outcome.output()) == 0:
            return

    # Motorola__XT1080__Apobe Photoshop 3.0__JPEG__95__2432x4320
    make, model, software, ext, quality, dimensions = outcome.output()[0].split('__')
    width, height = dimensions.split('x')

    return {
        'path': path,
        'label': label,
        'make': make,
        'model': model,
        'software': software,
        'ext': ext,
        'quality': int(quality),
        'width': int(width),
        'height': int(height),
        'dataset': dataset
    }

def extract_metadata(data_dir=os.environ['DATA_DIR'], rewrite=False, dataset='scrapped'):
    paths_and_labels = list_unencoded_samples(os.path.join(data_dir, dataset))
    pairs_to_process = list()
    dataset_docs = find_by(lambda q: q.dataset == dataset)

    if rewrite:
        ids = list(map(lambda doc: doc.doc_id, dataset_docs))
        remove(ids)
        pairs_to_process = paths_and_labels
    else:
        dataset_paths = list(map(lambda doc: doc['path'], dataset_docs))
        for pair in paths_and_labels:
            if pair[0] not in dataset_paths:
                pairs_to_process.append(pair)

    pool = ThreadPool()
    records = list()
    for record in tqdm(pool.imap(partial(extract_single, dataset), pairs_to_process), total=len(pairs_to_process)):
        if record: records.append(record)
        if len(records) == 100:
            insert_multiple(records)
            records = list()

    if len(records) > 0:
        insert_multiple(records)

if __name__ == '__main__':
    Fire(extract_metadata)
