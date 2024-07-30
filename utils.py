from prompts import TEMPLATE_PROMPT

def format_prompt(examples, EOS_TOKEN):
    messages = examples["Patient"]
    responses = examples["Doctor"]
    texts = []
    for message, response in zip(messages, responses):
        text = TEMPLATE_PROMPT.format(message, response) + EOS_TOKEN
        texts.append(text)
    return {"text" : texts}


