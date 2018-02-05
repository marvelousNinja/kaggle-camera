import os
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv
from fire import Fire
import shell
from shell import shell
from camera.db import insert, find_by_path, remove
from camera.data import list_unencoded_samples

load_dotenv(find_dotenv())

def extract_metadata(data_dir=os.environ['DATA_DIR'], rewrite=False):
    paths_and_labels = list_unencoded_samples(os.path.join(data_dir, 'scrapped'))

    for path, label in tqdm(paths_and_labels):
        doc = find_by_path(path)
        if rewrite and doc:
            remove(doc.doc_id)
        elif doc:
            continue

        # TODO AS: If you know how to do this with Wand, be my guest
        outcome = shell(f'identify -format "%[EXIF:Make]__%[EXIF:Model]__%m__%Q__%G" {path}')

        if len(outcome.errors()) > 0:
            tqdm.write(outcome.errors)
            continue

        # Motorola XT1080 JPEG 95 2432x4320
        make, model, ext, quality, dimensions = outcome.output()[0].split('__')
        width, height = dimensions.split('x')

        insert({
            'path': path,
            'label': label,
            'make': make,
            'model': model,
            'ext': ext,
            'quality': int(quality),
            'width': int(width),
            'height': int(height)
        })

if __name__ == '__main__':
    Fire(extract_metadata)
