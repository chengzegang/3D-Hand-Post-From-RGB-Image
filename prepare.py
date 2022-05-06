import os
import json
from tqdm.notebook import tqdm
from PIL import Image


for file in tqdm(os.listdir("data/hiu_dmtl_data/to_cv_community")):
    if file.endswith('.jpg'):
        name = os.path.basename(file).split('.')[0]
        img = Image.open(os.path.join('data', 'hiu_dmtl_data', 'to_cv_community', file))
        img.save(os.path.join('data', 'hiu_dmtl_data', 'to_cv_community', name + '.png'))
        os.remove(os.path.join('data', 'hiu_dmtl_data', 'to_cv_community', file))


if not os.path.exists("data/hiu_dmtl_data/train"):
    os.mkdir("data/hiu_dmtl_data/train")


usable = []
for file in tqdm(os.listdir("data/hiu_dmtl_data/to_cv_community")):
    if file.endswith('.json'):
        id = os.path.basename(file).split('.')[0]
        with open(os.path.join('data', 'hiu_dmtl_data', 'to_cv_community', file), 'r') as jsonfile:
            data = json.load(jsonfile)
            if sum(data['pts3d_w_2hand']) != 0:
                usable.append(id)


for id in tqdm(usable):
    image_path = os.path.join('data', 'hiu_dmtl_data', 'to_cv_community', id + '.png')
    label_path = os.path.join('data', 'hiu_dmtl_data', 'to_cv_community', id + '.json')
    os.rename(image_path, os.path.join('data', 'hiu_dmtl_data', 'train', id + '.png'))
    os.rename(label_path, os.path.join('data', 'hiu_dmtl_data', 'train', id + '.json'))



if not os.path.exists("data/hiu_dmtl_data/test"):
    os.mkdir("data/hiu_dmtl_data/test")
if not os.path.exists("data/hiu_dmtl_data/valid"):
    os.mkdir("data/hiu_dmtl_data/valid")
train_ids = []
test_ids = []
valid_ids = []
for idx, id in tqdm(enumerate(usable)):
    if idx % 10 == 0:
        image_path = os.path.join('data', 'hiu_dmtl_data', 'train', id + '.png')
        label_path = os.path.join('data', 'hiu_dmtl_data', 'train', id + '.json')
        os.rename(image_path, os.path.join('data', 'hiu_dmtl_data', 'test', id + '.png'))
        os.rename(label_path, os.path.join('data', 'hiu_dmtl_data', 'test', id + '.json'))
        test_ids.append(id)
    
    elif idx % 10 == 3:
        image_path = os.path.join('data', 'hiu_dmtl_data', 'train', id + '.png')
        label_path = os.path.join('data', 'hiu_dmtl_data', 'train', id + '.json')
        os.rename(image_path, os.path.join('data', 'hiu_dmtl_data', 'valid', id + '.png'))
        os.rename(label_path, os.path.join('data', 'hiu_dmtl_data', 'valid', id + '.json'))
        valid_ids.append(id)
    else:
        train_ids.append(id)


with open(os.path.join('data', 'hiu_dmtl_data', 'train','ids.json'), 'w') as jsonfile:
    json.dump(train_ids, jsonfile)

with open(os.path.join('data', 'hiu_dmtl_data', 'test','ids.json'), 'w') as jsonfile:
    json.dump(test_ids, jsonfile)

with open(os.path.join('data', 'hiu_dmtl_data', 'valid','ids.json'), 'w') as jsonfile:
    json.dump(valid_ids, jsonfile)
