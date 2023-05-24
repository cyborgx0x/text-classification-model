
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import BertConfig, BertModel


class Config(object):
    pretrained_model = "vinai/phobert-base"
    id2label = {0: "normal", 1: "spam"}
    label2id = {"normal": 0, "spam": 1}
    output_directory = "sms_spam_detection"
    epoch = 2
    training_argument = TrainingArguments(
            output_dir=output_directory,
            learning_rate=2e-6,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=epoch,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=True,)
    model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model, num_labels=2, id2label=id2label, label2id=label2id
        )
