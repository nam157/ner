import gradio as gr
import os

os.system("python -m spacy download en_core_web_sm")
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import re

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForTokenClassification.from_pretrained("./checkpoint/")
nlp = pipeline(task="token-classification", model=model, tokenizer=tokenizer)


def text_analysis(text):
    ner_result = nlp(text)
    string_pred = ""
    for key in ner_result:
        if key["entity"] == "LABEL_1":
            if key["word"].startswith("#") or key["word"].startswith("@"):
                tweet = re.sub(r"#", "", key["word"])
                tweet = re.sub(r"@", "", tweet)
                string_pred += tweet
            else:
                string_pred += " " + key["word"]
    list_strings = string_pred.split()
    pos_tokens = []
    for token in list_strings:
        pos_tokens.extend([(token, "collections"), (" ", None)])
    return pos_tokens


demo = gr.Interface(
    text_analysis,
    gr.Textbox(placeholder="Enter sentence here..."),
    ["highlight"],
    examples=[
        ["@brettcovington @pudgypenguins Nice I’m going to buy one also"],
        ["flat bill hat mfers!!!!!"],
        [
            "WIN #CryptoArt #Collectible  \n\nFor fraction get =&gt;  @BoredApeYC @doodles\n@moonbirds @goblintown @DinosChibi @alienfrens\n@MFTMKKUS more\n\nJoin MetaWin Now #NFT #Competition #Marketplace win Blue Chip 👇👇 \nhttps://t.co/UtmddlEvqM\n\ne.g thekid won Mutant Ape Yacht Club worth 17 ETH"
        ],
    ],
)

demo.launch()
