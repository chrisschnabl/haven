from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


def main():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def preprocess_function(examples):
        return tokenizer(examples["user_input"], truncation=True, padding=True)

    # Load dataset and preprocess
    toxic = load_dataset("lmsys/toxic-chat", "toxicchat0124")
    tokenized_test_dataset = toxic["test"].map(preprocess_function, batched=True)

    # Rename column to match the expected label key
    tokenized_test_dataset = tokenized_test_dataset.rename_column("toxicity", "label")
    columns_to_remove = ['conv_id', 'model_output', 'human_annotation', 'jailbreaking', 'openai_moderation', 'user_input']
    tokenized_test_dataset = tokenized_test_dataset.remove_columns(columns_to_remove)

    # Load fine-tuned model and create pipeline
    fine_tuned_model = AutoModelForSequenceClassification.from_pretrained("inxoy/my_awesome_model")
    classifier = pipeline("text-classification", model=fine_tuned_model, tokenizer=tokenizer)

    # Run inference on the first 10 entries of the test dataset
    for i in range(100):
        # Retrieve the original user_input text
        input_text = toxic["test"][i]["user_input"]  # Use the original dataset's "user_input"
        result = classifier(input_text)  # Classify the text
        print(f"Entry {i}: {result}")


if __name__ == "__main__":
    main()
