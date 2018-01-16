import os
import glob

def list_images_in(path):
    extensions = ['jpg', 'JPG', 'tiff', 'png']
    files = []
    for extension in extensions:
        files.extend(glob.glob(path + f'/*.{extension}'))
    return files

def list_dirs_in(path):
    return [dir for dir in os.listdir(path) if not dir.startswith('.')]
