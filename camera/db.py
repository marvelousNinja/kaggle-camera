import os
from dotenv import load_dotenv, find_dotenv
from tinydb import TinyDB, Query

load_dotenv(find_dotenv())

db = TinyDB(os.path.join(os.environ['DATA_DIR'], 'tinydb.json'))

def insert(obj):
    return db.insert(obj)

def find_by_path(path):
    return db.get(Query().path == path)

def remove(*ids):
    return db.remove(doc_ids=ids)

if __name__ == '__main__':
    import pdb; pdb.set_trace()
