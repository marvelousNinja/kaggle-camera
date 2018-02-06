import os
from dotenv import load_dotenv, find_dotenv
from tinydb import TinyDB, Query

load_dotenv(find_dotenv())

db = TinyDB(os.path.join(os.environ['DATA_DIR'], 'tinydb.json'))

def insert(record):
    return db.insert(record)

def insert_multiple(records):
    return db.insert_multiple(records)

def find_by_path(path):
    return db.get(Query().path == path)

def remove(ids):
    return db.remove(doc_ids=ids)

def find_by(func):
    return db.search(func(Query()))

if __name__ == '__main__':
    import pdb; pdb.set_trace()
