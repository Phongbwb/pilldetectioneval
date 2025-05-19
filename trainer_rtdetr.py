import torch
from transformers import AutoModelForObjectDetection, AutoImageProcessor, TrainingArguments, Trainer
from datasets import load_dataset
import argparse
import os

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': [x['labels'] for x in batch],
    }

def train_rtdetr(dataset_path, checkpoint_name, output_dir, epochs, batch_size):
    dataset = load_dataset("imagefolder", data_dir=dataset_path)
    processor = AutoImageProcessor.from_pretrained(checkpoint_name)
    model = AutoModelForObjectDetection.from_pretrained(checkpoint_name)

    def transform(example):
        image = example["image"]
        w, h = image.size
        boxes = example["objects"]["bbox"]  # normalized [x, y, width, height]
        class_labels = example["objects"]["category"]
        encoding = processor(images=image, annotations={"image_id": 0, "annotations": [{"bbox": box, "category_id": cid} for box, cid in zip(boxes, class_labels)]}, return_tensors="pt")
        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels": encoding["labels"][0],
        }

    dataset = dataset.with_transform(transform)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        save_steps=500,
        evaluation_strategy="epoch",
        save_total_limit=2,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        remove_unused_columns=False,
        push_to_hub=False
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=processor,
        data_collator=collate_fn
    )

    trainer.train()
    model.save_pretrained(output_dir)
    print(" Training RT-DETR completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="IDEA-Research/RT-DETR")
    parser.add_argument("--output_dir", type=str, default="./rtdetr_checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    train_rtdetr(args.dataset, args.checkpoint, args.output_dir, args.epochs, args.batch_size)