import os
import random


inroot = 'tweeteval/datasets'
outroot = 'iaa-sets'

corpora = [
    'emoji',
    'emotion',
    'hate',
    'irony',
    'offensive',
    'sentiment',
    'stance/abortion',
    'stance/atheism',
    'stance/climate',
    'stance/feminist',
    'stance/hillary'
]

iaa_size = 120

for crp in corpora:
    indir = inroot + '/' + crp
    outdir = outroot + '/' + crp
    with open(indir + '/train_text.txt', 'r') as f:
        train_text = [line.rstrip('\n') for line in f]
    with open(indir + '/train_labels.txt', 'r') as f:
        train_labels = [line.rstrip('\n') for line in f]

    train_size = len(train_text)
    assert len(train_labels) == train_size

    smpl = set(random.sample(range(train_size), iaa_size))
    iaa_text = [t for i, t in enumerate(train_text) if i in smpl]
    iaa_labels = [t for i, t in enumerate(train_labels) if i in smpl]

    os.makedirs(outdir, exist_ok=True)
    with open(outdir + '/iaa_text.txt', 'w') as f:
        print('\n'.join(iaa_text), file=f)
    with open(outdir + '/iaa_labels.txt', 'w') as f:
        print('\n'.join(iaa_labels), file=f)
    with open(outdir + '/iaa_indices.txt', 'w') as f:
        print('\n'.join(str(i) for i in sorted(smpl)), file=f)
