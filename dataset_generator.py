from datasets import Dataset, DatasetDict, Image
import os
from sklearn.model_selection import train_test_split

def find_image_label_dirs(root_dir):
    image_label_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if 'img' in dirnames and 'mask' in dirnames:
            image_dir = os.path.join(dirpath, 'img')
            label_dir = os.path.join(dirpath, 'mask')
            image_label_dirs.append((image_dir, label_dir))
    return image_label_dirs

def load_image_label_paths(image_label_dirs):
    image_paths = []
    label_paths = []
    for image_dir, label_dir in image_label_dirs:
        for filename in os.listdir(image_dir):
            if filename.endswith('.tif'):
                image_path = os.path.join(image_dir, filename)
                label_path = os.path.join(label_dir, filename)
                image_paths.append(image_path)
                label_paths.append(label_path)
    return image_paths, label_paths

def create_dataset(image_paths, label_paths):
    dataset = Dataset.from_dict({"image": image_paths, "annotation": label_paths})
    dataset = dataset.cast_column("image", Image())
    dataset = dataset.cast_column("annotation", Image())
    return dataset

root_dir = '.'

image_label_dirs = find_image_label_dirs(root_dir)

image_paths, label_paths = load_image_label_paths(image_label_dirs)

image_paths_train, image_paths_test, label_paths_train, label_paths_test = train_test_split(
    image_paths, label_paths, test_size=0.2, random_state=42
)
image_paths_train, image_paths_val, label_paths_train, label_paths_val = train_test_split(
    image_paths_train, label_paths_train, test_size=0.2, random_state=42
)

train_dataset = create_dataset(image_paths_train, label_paths_train)
validation_dataset = create_dataset(image_paths_val, label_paths_val)
test_dataset = create_dataset(image_paths_test, label_paths_test)

dataset = DatasetDict({
    "train": train_dataset,
    "validation": validation_dataset,
    "test": test_dataset,
})


for split_name, split_dataset in dataset.items():
    print(f"Number of samples in {split_name}: {len(split_dataset)}")

dataset.push_to_hub('syeddhasnainn/landslide-uav-sat')
