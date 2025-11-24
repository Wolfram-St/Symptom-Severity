from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import pipeline



dataset = load_dataset("csv", data_files={
	"Symptom-severity.csv"
})

df =dataset['train'].to_pandas()

train_df, test_df = train_test_split(df,test_size=0.2, random_state=42)
train_df.to_csv("train.csv",index=False)
test_df.to_csv("test.csv", index=False)

dataset= load_dataset("csv", data_files={
    "train": "train.csv",
    "test": "test.csv"
})

tokenizer= AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["Symptom"], padding="max_length",truncation=True)

dataset=dataset.map(tokenize, batched=True)
dataset= dataset.rename_column("weight","labels")
dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

model= AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=8)

trainer=Trainer(
    model=model,
    args= TrainingArguments(
        output_dir="./symptom_model",
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=8,
        num_train_epochs=3,
        save_total_limit=1,
        load_best_model_at_end=True,
    ),
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)


trainer.train()

trainer.save_model("./symptom_model")
tokenizer.save_pretrained("./symptom_model")

classifier=pipeline("text-classification", model="./symptom_model", tokenizer="distilbert-base-uncased")

result = classifier("I have chest pain and difficulty breathing.")
print(result)




