import os
import random
import shutil

train_data_ratio = 0.8
root = '/media/dl/本地磁盘/Remote-sensing-scene-classification-master/original_data/UCM21'
image_dir = root

train_dir = os.path.join('data/', 'train')
test_dir = os.path.join('data/', 'test')

if os.path.exists(train_dir):
    shutil.rmtree(train_dir)
if os.path.exists(test_dir):
    shutil.rmtree(test_dir)

os.makedirs(train_dir)  # 创建放训练数据的文件夹
os.makedirs(test_dir)

classes = os.listdir(image_dir)

for c in classes[0:21]:
    os.makedirs(os.path.join(train_dir, c), exist_ok=True)  # 创建训练数据下每个类别的文件夹
    os.makedirs(os.path.join(test_dir, c), exist_ok=True)

    class_dir = os.path.join(root, c)
    images = os.listdir(class_dir)

    random.shuffle(images)  # 随机打乱里面的数据

    n_train = int(len(images) * train_data_ratio)

    train_images = images[:n_train]
    test_images = images[n_train:]

    for images in train_images:
        image_src = os.path.join(class_dir, images)
        image_dst = os.path.join(train_dir, c, images)
        shutil.copyfile(image_src, image_dst)
    for images in test_images:
        image_src = os.path.join(class_dir, images)
        image_dst = os.path.join(test_dir, c, images)
        shutil.copyfile(image_src, image_dst)
