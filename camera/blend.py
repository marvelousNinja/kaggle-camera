import os
import pandas as pd
import numpy as np
from fire import Fire
from dotenv import find_dotenv, load_dotenv
from scipy.stats import mode
from camera.utils import generate_blend_submission_name
load_dotenv(find_dotenv())

def blend(*files, data_dir=os.environ['DATA_DIR'], index_name='fname', target_name='camera'):
    submissions = [pd.read_csv(file, index_col=index_name) for file in files]
    index = submissions[0].index
    values = np.empty((len(index), len(submissions)), dtype=object)

    for i, submission in enumerate(submissions):
        values[:, i] = submission[target_name].values

    blended_submission = pd.DataFrame({
        index_name: index.values,
        target_name: mode(values, axis=1)[0].reshape(-1)
    })

    filename = generate_blend_submission_name(files)
    submission_path = os.path.join(data_dir, 'submissions', filename)

    blended_submission.to_csv(submission_path, index=False)
    print(f'Submission file generated at {submission_path}')

if __name__ == '__main__':
    Fire(blend)
