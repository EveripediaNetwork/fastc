<p align="center">
    <img src="./misc/logo.svg" alt="Logo" width="300"/>
<p>

<p align="center">
    <a href="https://pypi.python.org/pypi/fastc/"><img alt="PyPi" src="https://img.shields.io/pypi/v/fastc.svg?style=flat-square"></a>
    <a href="https://github.com/EveripediaNetwork/fastc/releases"><img alt="GitHub releases" src="https://img.shields.io/github/release/EveripediaNetwork/fastc.svg?style=flat-square"></a>
    <a href="https://github.com/EveripediaNetwork/fastc/blob/master/LICENSE"><img alt="License" src="https://img.shields.io/github/license/EveripediaNetwork/fastc.svg?style=flat-square"></a>
</p>


# Key features
- **Focused on CPU execution:** Use efficient models like `deepset/tinyroberta-6l-768d` for embedding generation.
- **Cosine Similarity Classification:** Instead of fine-tuning, classify texts using cosine similarity between class embedding centroids and text embeddings.
- **Efficient Multi-Classifier Execution:** Run multiple classifiers without extra overhead when using the same model for embeddings.


# Installation
```bash
pip install -U fastc
```

# Train a model
You can train a text classifier with just a few lines of code:
```python
from fastc import SentenceClassifier

tuples = [
    ("I just got a promotion! Feeling fantastic.", 'positive'),
    ("Today was terrible. I lost my wallet and missed the bus.", 'negative'),
    ("I had a great time with my friends at the party.", 'positive'),
    ("I'm so frustrated with the traffic jam this morning.", 'negative'),
    ("My vacation was wonderful and relaxing.", 'positive'),
    ("I didn't get any sleep last night because of the noise.", 'negative'),
    ("I'm so excited for the concert tonight!", 'positive'),
    ("I'm disappointed with the service at the restaurant.", 'negative'),
    ("The weather is beautiful and I enjoyed my walk.", 'positive'),
    ("I had a bad day. Nothing went right.", 'negative'),
    ("I'm thrilled to announce that we are expecting a baby!", 'positive'),
    ("I feel so lonely and sad today.", 'negative'),
    ("My team won the championship! We are the champions.", 'positive'),
    ("I can't stand my job anymore, it's so stressful.", 'negative'),
    ("I love spending time with my family during the holidays.", 'positive'),
    ("My computer crashed and I lost all my work.", 'negative'),
    ("I'm proud of my achievements this year.", 'positive'),
    ("I'm exhausted and overwhelmed with everything.", 'positive'),
]

classifier = SentenceClassifier(embeddings_model='microsoft/deberta-base')
classifier.load_dataset(tuples)
classifier.train()
```

# Export a model
After training, you can save the model for future use:
```python
classifier.save_model('./sentiment-classifier/')
```

# Publish model to HuggingFace
> [!IMPORTANT]  
> Log in to HuggingFace first with `huggingface-cli login`

```python
classifier.push_to_hub('brunneis/sentiment-classifier')
```

# Load an existing model
You can load a pre-trained model either from a directory or from HuggingFace:
```python
# From a directory
classifier = SentenceClassifier('./sentiment-classifier/')

# From HuggingFace
classifier = SentenceClassifier('brunneis/sentiment-classifier')
```

# Class prediction
```python
sentences = [
    'I am feeling well.',
    'I am in pain.',
]

# Single prediction
scores = classifier.predict_one(sentences[0])
print(max(scores, key=scores.get))

# Batch predictions
scores_list = classifier.predict(sentences)
for scores in scores_list:
    print(max(scores, key=scores.get))
```

# Templates and Instruct Models
You can use instruct templates with instruct models such as `intfloat/multilingual-e5-large-instruct`. Other models may also improve in performance by using templates, even if they were not explicitly trained with them.

```python
from fastc import ModelTemplates, SentenceClassifier, Template

# template_text = 'Instruct: {instruction}\nQuery: {text}'
template_text = ModelTemplates.E5_INSTRUCT

classifier = SentenceClassifier(
    embeddings_model='intfloat/multilingual-e5-large-instruct',
    template=Template(
        template_text,
        instruction='Classify as positive or negative'
    ),
)
```
# Inference Server

To launch the dockerized inference server, use the following script:
```bash
./server/scripts/start-docker.sh
```

Alternatively, on the host machine:
```bash
./server/scripts/start-server.sh
```

In both cases, an HTTP API will be available, listening on the `fastc-server` *[hashport](https://github.com/labteral/hashport)* `53256`.

## Inference

To classify text, use `POST /` with a JSON payload such as:
```json
{
    "model": "braindao/tinyroberta-6l-768d-language-identifier-en-es-ko-zh-fastc",
    "text": "오늘 저녁에 친구들과 함께 pizza를 먹을 거예요."
}
```

Response:
```json
{
    "label": "ko",
    "scores": {
        "en": 0.23850876092910767,
        "es": 0.24473699927330017,
        "ko": 0.2621513605117798,
        "zh": 0.25460284948349
    }
}
```

## Version

To check the `fastc` version, use `GET /version`:

Response:
```json
{
    "version": "2.2406.0"
}
```