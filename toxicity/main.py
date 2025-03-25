from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


TOKENIXER = "distilbert-base-uncased"


def main():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def preprocess_function(examples):
        return tokenizer(examples["user_input"], truncation=True, padding=True)

    toxic = load_dataset("lmsys/toxic-chat", "toxicchat0124")
    tokenized_test_dataset = toxic["test"].map(preprocess_function, batched=True)

    # Rename column to match the expected label key
    tokenized_test_dataset = tokenized_test_dataset.rename_column("toxicity", "label")
    columns_to_remove = ['conv_id', 'model_output', 'human_annotation', 'jailbreaking', 'openai_moderation', 'user_input']
    tokenized_test_dataset = tokenized_test_dataset.remove_columns(columns_to_remove)

    
    fine_tuned_model = AutoModelForSequenceClassification.from_pretrained(model)
    classifier = pipeline("text-classification", model=fine_tuned_model, tokenizer=tokenizer)

    for i in range(100):
        input_text = toxic["test"][i]["user_input"]
        result = classifier(input_text) 
        print(f"Entry {i}: {result}")


if __name__ == "__main__":
    main()
