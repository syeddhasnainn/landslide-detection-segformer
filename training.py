from huggingface_hub import notebook_login
from datasets import load_dataset
import matplotlib.pyplot as plt
import json
from transformers import AutoImageProcessor
import numpy as np
import torch
import torch.nn as nn
from transformers import TrainerCallback
from transformers import AutoModelForSemanticSegmentation, TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
from PIL import Image
import random
import os
from numba import cuda 
import csv
import gc

os.makedirs('output_figures',exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class SaveMetricsCallback(TrainerCallback):
    def __init__(self, output_file):
        self.output_file = output_file
        self.metrics = []
        self.train_losses = []
        self.val_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if 'loss' in logs:
            self.train_losses.append(logs['loss'])
        if 'eval_loss' in logs:
            self.val_losses.append(logs['eval_loss'])

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        self.metrics.append(metrics)

    def on_train_end(self, args, state, control, **kwargs):
        with open(self.output_file, "w", newline="") as csvfile:
            fieldnames = list(self.metrics[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.metrics)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits_tensor = torch.from_numpy(logits).to("cpu")  # Move logits tensor to the GPU
    labels = torch.from_numpy(labels).to("cpu")  # Move labels tensor to the GPU

    
    # Scale the logits to the size of the label
    logits_tensor = nn.functional.interpolate(
        logits_tensor,
        size=labels.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).argmax(dim=1)

    # Compute IoU, recall, precision, and F1 score for each class
    num_classes = logits_tensor.max().item() + 1
    iou_per_class = torch.zeros(num_classes, device=device)
    recall_per_class = torch.zeros(num_classes, device=device)
    precision_per_class = torch.zeros(num_classes, device=device)
    f1_per_class = torch.zeros(num_classes, device=device)

    for class_idx in range(num_classes):
        tp = torch.sum((logits_tensor == class_idx) & (labels == class_idx))
        fn = torch.sum((logits_tensor != class_idx) & (labels == class_idx))
        fp = torch.sum((logits_tensor == class_idx) & (labels != class_idx))
        tn = torch.sum((logits_tensor != class_idx) & (labels != class_idx))

        iou = tp / (tp + fn + fp)
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        f1 = 2 * (precision * recall) / (precision + recall)

        iou_per_class[class_idx] = iou
        recall_per_class[class_idx] = recall
        precision_per_class[class_idx] = precision
        f1_per_class[class_idx] = f1

    # Compute macro-averaged IoU, recall, precision, and F1 score
    macro_iou = iou_per_class.mean().item()
    macro_recall = recall_per_class.mean().item()
    macro_precision = precision_per_class.mean().item()
    macro_f1 = f1_per_class.mean().item()
    iou = iou_per_class[1].item()

    # Compute overall accuracy (accuracy)
    accuracy = torch.sum(logits_tensor == labels).float() / labels.numel()

    metrics = {
        "mIoU": macro_iou,
        "recall": macro_recall,
        "precision": macro_precision,
        "f1": macro_f1,
        "iou": iou,
        "accuracy": accuracy.item(),
    }

    return metrics


def train_transforms(example_batch):
    images = [x for x in example_batch["image"]]
    labels = [x for x in example_batch["annotation"]]
    inputs = image_processor(images, labels)
    return inputs


def val_transforms(example_batch):
    images = [x for x in example_batch["image"]]
    labels = [x for x in example_batch["annotation"]]
    inputs = image_processor(images, labels)
    return inputs

def display_images(dataset_name, model_variant,img_number,original_image, ground_truth, prediction, title, figsize=(15, 5)):
    """Display the original image, ground truth mask, and predicted mask."""
    output_dir = f"output_figures/{dataset_name}/{model_variant}"
    os.makedirs(output_dir,exist_ok=True)
    plt.figure(figsize=figsize)

    # Check if the original image is channel-first and convert it
    if original_image.shape[0] == 3:  # Assuming 3 channels for RGB
        original_image = np.transpose(original_image, (1, 2, 0))

    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(ground_truth, cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    # Convert one-hot encoded predictions to single channel if necessary
    if prediction.shape[0] == 2:  # Assuming 2 classes
        prediction = np.argmax(prediction, axis=0)  # Take argmax over the class dimension
    plt.imshow(prediction, cmap='gray')
    plt.title('Prediction')
    plt.axis('off')

    plt.suptitle(title)
    # plt.show()
    plt.savefig(f'{output_dir}/{img_number}.png')
    

model_variants = ["mit-b5","mit-b4","mit-b3","mit-b2","mit-b1","mit-b0"]
datasets = ["landslide-uav-sat", "landslide-uav-all", "landslide-sat-all"]


for dataset_name in datasets:
    for model_variant  in model_variants:

        checkpoint = f'nvidia/{model_variant}'
        output_dir = f'{os.getcwd()}/{dataset_name}/{model_variant}'
        os.makedirs(output_dir,exist_ok=True)
        metrics_file = f'{model_variant}-{dataset_name}.csv'
        dataset = load_dataset(f'syeddhasnainn/{dataset_name}')

        dataset['train'][0]
        np.array(dataset['validation'][0]['image'])
        np.unique(np.array(dataset['train'][0]['annotation']))
        image = dataset['train'][1]['image']
        annotation = dataset['train'][1]['annotation']

    
        label_data = {"0": "non-landslide", "1": "landslide"}
        # Create a mapping from label IDs to label names
        id2label = {int(k): v for k, v in label_data.items()}
        # Create a mapping from label names to label IDs
        label2id = {v: k for k, v in id2label.items()}
        # Get the number of labels
        num_labels = len(id2label)

        model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint, id2label=id2label, label2id=label2id)
        image_processor = AutoImageProcessor.from_pretrained(checkpoint, reduce_labels=False)

        train_ds = dataset["train"]
        valid_ds = dataset["validation"]
        test_ds = dataset['test']

        train_ds.set_transform(train_transforms)
        valid_ds.set_transform(val_transforms)
        test_ds.set_transform(val_transforms)
        

        metrics_file = os.path.join(output_dir, metrics_file)

        best_model_dir = os.path.join(output_dir, "best_model")

        batch_size = 30
        if "b2" in model_variant:
            batch_size = 14
        if "b3" in model_variant or "b4" in model_variant:
            batch_size = 8
        if "b5" in model_variant:
            batch_size = 6

        print('BATCH_SIZE:',batch_size)
        print('BATCH_SIZE:',batch_size)
        print('BATCH_SIZE:',batch_size)

        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=6e-5,
            num_train_epochs=10,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            save_total_limit=3,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_steps=50,
            eval_steps=50,
            logging_steps=50,
            eval_accumulation_steps=10,
            remove_unused_columns=False,
            push_to_hub=False,
            greater_is_better=True,
            fp16=True,
        )

        save_metrics_callback = SaveMetricsCallback(metrics_file)
        print("build trainer with on device:", training_args.device, "with n gpus:", training_args.n_gpu)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=valid_ds,
            compute_metrics=compute_metrics,
            callbacks=[save_metrics_callback],
        )

        trainer.train()
        os.makedirs(best_model_dir, exist_ok=True)
        trainer.save_model(best_model_dir)

        best_model = trainer.model

        # Evaluate the best model on the test set
        test_results = trainer.predict(test_dataset=test_ds)
        test_metrics = test_results.metrics
        # print("Test metrics:", test_metrics)

        # Save the test predictions
        test_predictions = test_results.predictions

        num_samples = 10 # Number of samples to visualize

        for i in range(num_samples):
            # Extracting data from the dataset
            # i = random.randint(0,200)
            original_img = np.array(test_ds[i]['pixel_values'])  # Convert PIL Image to NumPy array
            ground_truth_mask = np.array(test_ds[i]['labels'])
            
            # Handling predictions similarly
            predicted_mask = test_predictions[i]

           
            display_images(dataset_name, model_variant,i,original_img, ground_truth_mask, predicted_mask, f"Sample {i+1}")

        torch.cuda.empty_cache()
        del model
        del train_ds
        del valid_ds
        del test_ds
        del best_model
        del image_processor
        gc.collect()
        


        