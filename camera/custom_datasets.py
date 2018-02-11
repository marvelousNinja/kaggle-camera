"""Data filters and loading utils to build custom datasets"""
import pandas as pd

from camera.db import find_by
from camera.data import label_mapping

def get_lg_nexus5x_samples(samples):
    return samples[
        samples.software.str.contains('bullhead') |
        ((samples.model == 'Nexus 5X') & samples.software.str.match('HDR'))
    ]

def get_htc_one_samples(samples):
    valid_models = ['HTC One', 'HTCONE']
    return samples[
        (samples.model.isin(valid_models)) &
        (samples.software == '')
    ]

def get_motorola_x_samples(samples):
    valid_models = ['XT1092', 'XT1093', 'XT1095', 'XT1096', 'XT1097']
    return samples[
        (samples.model.isin(valid_models)) &
        (samples.software == '')
    ]

def get_motorola_maxx_samples(samples):
    valid_models = ['XT1080']
    return samples[
        (samples.model.isin(valid_models)) &
        (samples.software == '')
    ]

def get_motorolla_nexus6_samples(samples):
    return samples[
        ((samples.model == 'Nexus 6') & (samples.software == '')) |
        ((samples.model == '') & (samples.software.str.match('angler'))) |
        ((samples.model == 'Nexus 6') & (samples.software.str.match('HDR')))
    ]

def get_samsung_galaxy_note3_samples(samples): # pylint: disable=invalid-name
    valid_models = ['SAMSUNG-SM-N900A', 'SM-N900P', 'SM-N9005', 'SM-N900A']
    software_patterns = ['N900']
    return samples[
        (samples.model.isin(valid_models)) &
        (samples.software.str.match('|'.join(software_patterns)))
    ]

def get_samsung_galaxy_s4_samples(samples):
    valid_models = ['SCH-I545', 'SPH-L720', 'GT-I9505', 'SPH-L720T']
    software_patterns = ['I545', 'L720', 'I9505']
    return samples[
        (samples.model.isin(valid_models)) &
        (samples.software.str.match('|'.join(software_patterns)))
    ]

def get_sony_nex7_samples(samples):
    valid_models = ['NEX-7']
    software_patterns = ['NEX-7']
    return samples[
        (samples.model.isin(valid_models)) &
        (samples.software.str.match('|'.join(software_patterns)))
    ]

def get_iphone4s_samples(samples):
    valid_models = ['iPhone 4S']
    software_patterns = [r'^\d\.', r'^\d\d\.']
    return samples[
        (samples.model.isin(valid_models)) &
        (samples.software.str.match('|'.join(software_patterns)))
    ]

def get_iphone6_samples(samples):
    valid_models = ['iPhone 6']
    software_patterns = [r'^\d\.', r'^\d\d\.']
    return samples[
        (samples.model.isin(valid_models)) &
        (samples.software.str.match('|'.join(software_patterns)))
    ]

def get_scrapped_samples():
    scrapped = find_by(lambda q: q.dataset == 'scrapped')
    scrapped = pd.DataFrame(scrapped)

    scrapped = scrapped[
        (scrapped.height > 780) &
        (scrapped.width > 780) &
        (scrapped.quality >= 95)
    ]

    scrapped = pd.concat([
        get_lg_nexus5x_samples(scrapped),
        get_htc_one_samples(scrapped),
        get_motorola_x_samples(scrapped),
        get_motorola_maxx_samples(scrapped),
        get_motorolla_nexus6_samples(scrapped),
        get_samsung_galaxy_note3_samples(scrapped),
        get_samsung_galaxy_s4_samples(scrapped),
        get_iphone4s_samples(scrapped),
        get_iphone6_samples(scrapped),
        get_sony_nex7_samples(scrapped)
    ])

    return scrapped

def get_scrapped_dataset():
    """Collect scrapped dataset samples as a list of (local_path, label) pairs
    """
    samples = get_scrapped_samples()
    samples['label'] = samples.label.map(label_mapping())
    return samples[['path', 'label']].values

if __name__ == '__main__':
    print(get_scrapped_samples().groupby(['label', 'make', 'model']).size())
    print(get_scrapped_samples().groupby(['label']).size().sort_values())
    import pdb; pdb.set_trace()
