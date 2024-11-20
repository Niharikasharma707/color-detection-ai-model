# -*- coding: utf-8 -*-
import os
import torch
import requests
import zipfile
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fastai.vision import *
from fastai.metrics import accuracy, top_k_accuracy
from annoy import AnnoyIndex

# Define the root path for all project components
project_root = "D:/Projects/real-time-fashion-recommendation"
os.makedirs(project_root, exist_ok=True)

# Helper function to download files if they do not exist
def download_file(url, output):
    if not os.path.exists(output):
        print(f"Downloading {output}...")
        r = requests.get(url, allow_redirects=True)
        if r.status_code == 200:
            with open(output, 'wb') as f:
                f.write(r.content)
            print(f"Downloaded {output}.")
        else:
            print(f"Failed to download {output}. Status code: {r.status_code}")
    else:
        print(f"{output} already exists. Skipping download.")

# Function to handle Google Drive file downloading with redirection
def download_google_drive_file(gdrive_id, output):
    # Google Drive download URL
    url = f"https://drive.google.com/uc?id={gdrive_id}"
    session = requests.Session()
    response = session.get(url, allow_redirects=True)
    
    # Check if we need to follow redirect
    if "confirm=" in response.url:
        # Follow the redirect to the confirmation page
        response = session.get(response.url, allow_redirects=True)
    
    if response.status_code == 200:
        with open(output, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {output}.")
    else:
        print(f"Failed to download {output}. Status code: {response.status_code}")

# Check GPU availability
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Training on CPU...')
else:
    print('CUDA is available! Training on GPU...')
    print(torch.cuda.get_device_name(0))

# Define the paths for files
cloth_txt = os.path.join(project_root, 'list_category_cloth.txt')
img_txt = os.path.join(project_root, 'list_category_img.txt')
eval_txt = os.path.join(project_root, 'list_eval_partition.txt')
zip_file = os.path.join(project_root, 'img.zip')

# Download the required text files
download_google_drive_file('0B7EVK8r0v71pWnFiNlNGTVloLUk', cloth_txt)
download_google_drive_file('0B7EVK8r0v71pTGNoWkhZeVpzbFk', img_txt)
download_google_drive_file('0B7EVK8r0v71pdS1FMlNreEwtc1E', eval_txt)
download_google_drive_file('1j5fCPgh0gnY6v7ChkWlgnnHH6unxuAbb&confirm=t&uuid=6554c12b-2854-48a4-a9eb-6124619da58e', zip_file)

# Unzip the images if not already unzipped
img_folder = os.path.join(project_root, 'img')
if not os.path.exists(img_folder):
    print("Extracting image data...")
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(project_root)
        print("Image data extracted successfully.")
    except zipfile.BadZipFile:
        print("Error: The zip file is corrupted or not a zip file.")
else:
    print("Image data already extracted.")

# Load the metadata
category_list = []
image_path_list = []
data_type_list = []

with open(cloth_txt, 'r') as f:
    for i, line in enumerate(f.readlines()):
        if i > 1:
            category_list.append(line.split(' ')[0])

with open(img_txt, 'r') as f:
    for i, line in enumerate(f.readlines()):
        if i > 1:
            image_path_list.append([word.strip() for word in line.split(' ') if len(word) > 0])

with open(eval_txt, 'r') as f:
    for i, line in enumerate(f.readlines()):
        if i > 1:
            data_type_list.append([word.strip() for word in line.split(' ') if len(word) > 0])

data_df = pd.DataFrame(image_path_list, columns=['image_path', 'category_number'])
data_df['category_number'] = data_df['category_number'].astype(int)
data_df = data_df.merge(pd.DataFrame(data_type_list, columns=['image_path', 'dataset_type']), on='image_path')
data_df['category'] = data_df['category_number'].apply(lambda x: category_list[int(x) - 1])
data_df = data_df.drop('category_number', axis=1)

# Show dataset summary
print(data_df.head())
print(f"Total number of images: {len(data_df)}")

# Image path setup
train_image_list = ImageList.from_df(df=data_df, path=project_root, cols='image_path').split_by_idxs(
    (data_df[data_df['dataset_type'] == 'train'].index),
    (data_df[data_df['dataset_type'] == 'val'].index)
).label_from_df(cols='category')

test_image_list = ImageList.from_df(df=data_df[data_df['dataset_type'] == 'test'], path=project_root, cols='image_path')

data = train_image_list.transform(get_transforms(), size=224).databunch(bs=128).normalize(imagenet_stats)
data.add_test(test_image_list)

data.show_batch(rows=3, figsize=(8, 8))

# Train ResNet Model
def train_model(data, pretrained_model, model_metrics):
    learner = cnn_learner(data, pretrained_model, metrics=model_metrics)
    learner.model = torch.nn.DataParallel(learner.model)
    learner.lr_find()
    learner.recorder.plot(suggestion=True)
    return learner

pretrained_model = models.resnet18
model_metrics = [accuracy, partial(top_k_accuracy, k=1), partial(top_k_accuracy, k=5)]
learner = train_model(data, pretrained_model, model_metrics)

learner.fit_one_cycle(10, max_lr=1e-02)

# Evaluate model
interp = ClassificationInterpretation.from_learner(learner)
interp.plot_top_losses(9, largest=False, figsize=(15, 11), heatmap_thresh=5)
interp.plot_confusion_matrix(figsize=(12, 12), dpi=60)

# Save the model to local disk
model_save_path = os.path.join(project_root, 'resnet18-fashion')
learner.save(model_save_path)

# Hook for extracting embeddings
class SaveFeatures():
    features = None
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)
        self.features = None
    def hook_fn(self, module, input, output):
        out = output.detach().cpu().numpy()
        if isinstance(self.features, type(None)):
            self.features = out
        else:
            self.features = np.row_stack((self.features, out))
    def remove(self):
        self.hook.remove()

# Load the trained model
learner = learner.load(model_save_path)

# Extract embeddings
saved_features = SaveFeatures(learner.model.module[1][4])
_= learner.get_preds(data.train_ds)
_= learner.get_preds(DatasetType.Valid)

# Prepare output dataframe with embeddings
img_path = [str(x) for x in (list(data.train_ds.items) + list(data.valid_ds.items))]
label = [data.classes[x] for x in (list(data.train_ds.y.items) + list(data.valid_ds.y.items))]
label_id = [x for x in (list(data.train_ds.y.items) + list(data.valid_ds.y.items))]
data_df_output = pd.DataFrame({'img_path': img_path, 'label': label, 'label_id': label_id})
data_df_output['embeddings'] = np.array(saved_features.features).tolist()

# Annoy for Approximate Nearest Neighbors
def get_similar_images_annoy(annoy_tree, img_index, number_of_items=12):
    start = time.time()
    img_id, img_label = data_df_output.iloc[img_index, [0, 1]]
    similar_img_ids = annoy_tree.get_nns_by_item(img_index, number_of_items + 1)
    end = time.time()
    print(f'{(end - start) * 1000} ms')
    return img_id, img_label, data_df_output.iloc[similar_img_ids[1:]]

def get_similar_images_annoy_centroid(annoy_tree, vector_value, number_of_items=12):
    start = time.time()
    similar_img_ids = annoy_tree.get_nns_by_vector(vector_value, number_of_items + 1)
    end = time.time()
    print(f'{(end - start) * 1000} ms')
    return data_df_output.iloc[similar_img_ids[1:]]

def show_similar_images(similar_images_df, fig_size=[10, 10], hide_labels=True):
    if hide_labels:
        category_list = [''] * len(similar_images_df)
    else:
        category_list = [learner.data.train_ds.y.reconstruct(y) for y in similar_images_df['label_id']]
    return learner.data.show_xys([open_image(img_id) for img_id in similar_images_df['img_path']],
                                category_list, figsize=fig_size)

# Build the Annoy tree
ntree = 100
metric_choice = 'angular'
annoy_tree = AnnoyIndex(len(data_df_output['embeddings'][0]), metric=metric_choice)

for i, vector in enumerate(data_df_output['embeddings']):
    annoy_tree.add_item(i, vector)
annoy_tree.build(ntree)