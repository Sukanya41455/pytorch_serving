from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model():
    tokenizer = AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")
    model = AutoModelForSequenceClassification.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")

    tokenizer.save_pretrained('./my_tokenizer')
    model.save_pretrained('./my_model')


load_model()