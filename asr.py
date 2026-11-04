import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset, Audio
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# Load the Model and Processor
model_id = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_id, language="Hindi", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(model_id)

# Apply LoRA
config = LoraConfig(
    r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none"
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, config)
print("\n--- LoRA Setup Complete ---")
model.print_trainable_parameters() 

# Load a Small Hindi Dataset (FLEURS - 100 samples)
dataset = load_dataset("google/fleurs", "hi_in", split="train[:100]")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
    return batch

dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)

# Custom Data Collator (Fixed the [0] bug)
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-hindi-lora",
    per_device_train_batch_size=8,
    learning_rate=1e-3,
    num_train_epochs=3,
    eval_strategy="no",
    predict_with_generate=True,
    fp16=True,
    report_to="none",
    logging_steps=5
)

# Train!
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)

print("\nStarting fine-tuning...")
trainer.train()

print("\n--- Evaluation Results ---")
results = trainer.evaluate(eval_dataset=dataset)
print(results)