import json

import requests

from collections import Counter
data = None
with open('active_train.json', 'r') as f:
    a = f.read()
    data = json.loads(a)


images = data["images"]

annotations = data["annotations"]
# 过滤全部包含人的
# 统计重复
image_ids_all = [a["image_id"] for a in annotations]

image_id_count = Counter(image_ids_all)
image_ids = []
# 帅选出只有2， 3 个人的图片
for k, v in image_id_count.items():
    if v in [2]:
        image_ids.append(k)




redownload = ['185250', '81594', '100510']
images = [image for image in images if image['id'] in image_ids]
annotations = [annotation for annotation in annotations if annotation['image_id'] in image_ids and
        annotation['num_keypoints'] > 9 and
               (annotation["keypoints"][-2] > 20)
               ]

# 统计重复
image_ids_all = [a["image_id"] for a in annotations]

image_id_count = Counter(image_ids_all)
image_ids = []
# 帅选出只有2， 3 个人的图片
for k, v in image_id_count.items():
    if v in [2]:
        image_ids.append(k)

images = [image for image in images if image['id'] in image_ids]
annotations = [annotation for annotation in annotations if annotation['image_id'] in image_ids
               ]

#
# data["images"] = images
# print(images[0])
# data["annotations"] = annotations
# data = json.dumps(data)
#
# with open('annotations/active_train.json', 'w') as f:
#     f.write(data)
# with open('annotations/active_val.json', 'w') as f:
#     f.write(data)

count = 0
total = len(images)
remove_image_ids = []
for image in images:
    count += 1
    # if count < 500:
    #     continue
    for i in range(4):
        try:
            r = requests.get(image['flickr_url'])
            if r.status_code == 200:
                with open(f'train/{image["file_name"]}', 'wb') as f:
                    print(f"download {count}/{image_id_count[image['id']]}/{total} --- {r.status_code}")
                    content = r.content
                    # print(f"content {content[:20]}")
                    f.write(content)
                pass
            else:
                print(f"download {count}/{image_id_count[image['id']]}/{total} --- {r.status_code}")
                remove_image_ids.append(image['id'])
            break
        except:
            print(f"try {i}")

# remove_image_ids = [456496, 281414, 337055, 183127, 39551, 60347, 439522, 314177, 103548, 261732, 191288, 576566, 364102, 364126, 170191, 10764, 391375, 340451, 281929, 442746, 177357, 130579, 426376, 104424, 556158, 493442, 491090, 81594, 94326, 389684, 20333, 38118, 563470, 51610, 481386, 203639, 161875, 210915, 94185, 492937, 47819, 295138, 187585, 455937, 198928, 7088, 227491, 52507, 60899, 13201, 369310, 193717, 529939, 493772, 365095, 128112, 263068, 324158, 68628, 76547, 397279, 269632, 442306, 343453, 451155, 161861, 306437, 574297]
print(f"removed images {remove_image_ids}")
image_ids = [image_id for image_id in image_ids if image_id not in remove_image_ids]

images = [image for image in images if image['id'] in image_ids]
annotations = [annotation for annotation in annotations if annotation['image_id']  in image_ids]


train_image_ids = image_ids[:(len(image_ids)//3) * 2]
train_images = [image for image in images if image['id'] in train_image_ids]
train_annotations = [annotation for annotation in annotations if annotation['image_id']  in train_image_ids]


val_image_ids = image_ids[(len(image_ids)//3) * 2:]
val_images = [image for image in images if image['id'] in val_image_ids]
val_annotations = [annotation for annotation in annotations if annotation['image_id']  in val_image_ids]

# data["images"] = images
# print(images[0])
# data["annotations"] = annotations
# data = json.dumps(data)

with open('annotations/active_train.json', 'w') as f:
    data["images"] = train_images
    print(images[0])
    data["annotations"] = train_annotations
    _data = json.dumps(data)
    f.write(_data)
with open('annotations/active_val.json', 'w') as f:
    data["images"] = val_images
    print(images[0])
    data["annotations"] = val_annotations
    _data = json.dumps(data)
    f.write(_data)
    # f.write(data)



