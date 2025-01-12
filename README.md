# Anki Deck for Learning Simple Chinese Sentences

This blog post describes how to create an Anki deck to learn simple Chinese sentences.

First, I gathered a large collection of English sentences from a dataset of dialogues. I then used several AI tools to process and refine these sentences. I employed `NLTK` for sentence tokenization and basic text processing. To remove redundant sentences, I utilized `NLTK`'s `WordNetLemmatizer` for lemma-based deduplication and the `SentenceTransformers` library for sentence embedding-based deduplication. Furthermore, I prioritized sentences with high-frequency vocabulary using a word frequency list.

Next, I translated the English sentences into Chinese using the `Qwen/QwQ-32B-Preview` LLM. I also used the `pinyin-jyutping` library to generate Pinyin transcriptions for each Chinese translation.

Finally, I used the `GenAnki` library to create the Anki flashcards. Each flashcard has an English sentence on one side and the Chinese translation and Pinyin on the other side.

This project demonstrates the power of AI in language learning, showcasing how tools like large language models, sentence embedding models, and natural language processing libraries can be effectively utilized to create personalized and efficient learning resources.

Below, I describe the project in detail.

## Prepare English Sentences
The foundation of our Anki deck will be a collection of simple and natural-sounding English sentences. To gather these sentences, we'll leverage a readily available dataset:

### Load Dialogue Dataset
We'll start by loading the `knkarthick/dialogsum` dataset from the Huggingface Hub. This dataset contains a large collection of dialogues, which are rich sources of conversational language.

```python
from datasets import load_dataset

dataset = load_dataset("knkarthick/dialogsum")

# Extract dialogues
dialogues = []
for _, split_data in dataset.items():
    split_dialogues = [dialog['dialogue'] for dialog in split_data]
    dialogues.extend(split_dialogues)
```

### Extract Sentences
Next, we'll extract individual sentences from the dialogues. This involves identifying dialogue turns between speakers using regular expressions and then extracting sentences within each speaker's turn using NLTK's sentence tokenizer.


```python
import re
import nltk
from tqdm import tqdm

# Extract dialog turns and split sentences
sentences = set()
for dialogue in tqdm(dialogues, desc="Extracting sentences"):
    turns = dialogue.split('\n')
    for turn in turns:
        turn = re.sub(r"#.*?#:", "", turn).strip()
        sentences.update(nltk.sent_tokenize(turn))
```

### Filter Sentences

To ensure we're working with sentences suitable for beginner learners, we'll apply several filtering criteria:

* Length: We'll exclude sentences that are too short (less than 3 non-punctuation tokens) or too long (more than 7 non-punctuation tokens).
* Vocabulary: We'll restrict the vocabulary to the 10,000 most frequent English words. This ensures that the sentences contain common and easily learnable vocabulary.

```python
import string

# Filter sentences on surface criteria
# Load 10000 most frequent English words from GitHub page
url = "https://raw.githubusercontent.com/first20hours/google-10000-english/refs/heads/master/google-10000-english-no-swears.txt"
english_words = set([word.strip() for word in requests.get(url).text.split("\n") if word.strip()])

filtered_sentences = []
lowercase_sentences = set()

for sentence in tqdm(list(sentences), desc="Filtering sentences"):
    tokens = nltk.word_tokenize(sentence)
    # Remove punctuation
    tokens = [token for token in tokens if token not in string.punctuation]
    if len(tokens) < 3:
        continue
    if len(tokens) > 7:
        continue

    # Filter sentences where at least one word is not in the English word list
    if not any(token in english_words for token in tokens):
        continue

    if sentence.lower() in lowercase_sentences: 
        continue

    filtered_sentences.append(sentence)
    lowercase_sentences.add(sentence.lower())
```


### Deduplicate Sentences:

To avoid redundancy and focus on diverse sentence structures, we'll implement a two-step deduplication process:

1. **Lemma-based Deduplication**: We'll lemmatize all tokens in the sentences (e.g., "running" becomes "run"). Then, we'll group sentences based on their lemmas. If a new sentence contains all the lemmas of an existing sentence, we'll discard the new sentence. For lemmatization, we use `NLTK`'s `WordNetLemmatizer`.

```python
from collections import defaultdict

# Build inverse mapping from lemmas to sentences
lemma_to_sentences = defaultdict(set)
sentences = list(filtered_sentences)

lemmatizer = nltk.WordNetLemmatizer()
stopwords = set(nltk.corpus.stopwords.words('english'))

for sentence in tqdm(sentences, desc="Building lemma to sentences mapping"):
    tokens = nltk.word_tokenize(sentence)
    lemmas = set([lemmatizer.lemmatize(token) for token in tokens])
    # Filter out stopwords
    lemmas = [lemma for lemma in lemmas if lemma not in stopwords]

    # Get intersection of sentences with the same lemmas
    intersection = set.intersection(*[lemma_to_sentences[lemma] for lemma in lemmas])
    if len(intersection) > 0:
        continue

    for lemma in lemmas:
        lemma_to_sentences[lemma].add(sentence)

filtered_sentences = list(set.union(*lemma_to_sentences.values()))
```

2. **Sentence Embedding Deduplication**: We'll embed all remaining sentences using a sentence embedding model from the `SentenceTransformers` library. Sentences with a cosine similarity greater than 0.7 will be grouped together, and only the shortest sentence from each group will be retained.

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# Embed sentences
model = SentenceTransformer("all-MiniLM-L6-v2")
sentences = list(filtered_sentences)
sentence_embeddings = model.encode(sentences, batch_size=100)

# Normalize embeddings for cosine distance
sentence_embeddings = sentence_embeddings / np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)

# Group sentences by similarity
clusters = []
processed_sentence_ids = set()
batch_size = 100

progress_bar = tqdm(total=len(sentences), desc="Building clusters")
start_idx = 0

while start_idx < len(sentences):
    batch_idxs = []
    while len(batch_idxs) < batch_size:
        if start_idx >= len(sentences):
            break
        if start_idx in processed_sentence_ids:
            start_idx += 1
            continue

        batch_idxs.append(start_idx)
        start_idx += 1

    cosine_similarities = np.dot(sentence_embeddings, sentence_embeddings[batch_idxs].T).T
    cluster_mask = cosine_similarities > 0.7
    for sentence_idx, mask in zip(batch_idxs, cluster_mask):
        mask_idxs = np.nonzero(mask)[0].tolist()
        mask_idxs = [idx for idx in mask_idxs if idx not in processed_sentence_ids]
        if len(mask_idxs) == 0:
            continue

        clusters.append(mask_idxs)
        processed_sentence_ids.update(set(mask_idxs))

    progress_bar.update(len(batch_idxs))

# From each cluster, select the sentence with fewest words
filtered_sentences = []

for cluster in clusters:
    cluster_sentences = [sentences[idx] for idx in cluster]
    cluster_sentences = sorted(cluster_sentences, key=lambda x: len(x.split()))
    filtered_sentences.append(cluster_sentences[0])

sentences = filtered_sentences
```

## Extract a List of Sentences Suitable for Learning
To further refine our sentence selection, we'll prioritize sentences that are likely to be most beneficial for learning.

### Group Sentences by Length

We'll group the sentences based on their number of non-punctuation tokens. This will allow us to gradually increase the complexity of the sentences.

```python
# Chunk sentences by number of tokens
sentences_by_num_tokens = defaultdict(list)
for sentence in sentences:
    # Remove punctuation
    tokens = [token for token in nltk.word_tokenize(sentence) if token not in string.punctuation]
    sentences_by_num_tokens[len(tokens)].append(sentence)
```

### Sort and Select Sentences:
For each group, we'll calculate the frequency of non-stopword tokens based on a common English frequency list. We'll then sort the sentences within each group by the minimum word frequency in descending order. This ensures that sentences with higher-frequency vocabulary are prioritized. Finally, we'll select the top 200 sentences from each group to include in our Anki deck.

```python
# Load word counts
import pandas as pd
from io import StringIO

url = "https://norvig.com/ngrams/count_1w.txt"
url_content = requests.get(url).text
word_counts = pd.read_csv(StringIO(url_content), sep='\t', header=None, names=['word', 'count'])
word_counts = word_counts.set_index("word")
word_counts = word_counts.to_dict()["count"]

def minimum_word_count(sentence):
    sentence = sentence.replace("'", "")
    tokens = nltk.word_tokenize(sentence)
    # remove stopwords
    tokens = [token.lower() for token in tokens if token not in stopwords]
    # remove punctuation
    tokens = [token for token in tokens if token not in string.punctuation]
    if len(tokens) == 0:
        return 0
    
    return min([word_counts.get(token, 0) for token in tokens])

# For each num_tokens_value, keep the top 200 sentences with highest minimum word count
filtered_sentences_by_num_tokens = dict()
for num_tokens_value, sentences_with_num_tokens in sorted(sentences_by_num_tokens.items(), key=lambda x: x[0]):
    if num_tokens_value > 8 or num_tokens_value < 3:
        continue

    sentences_with_num_tokens = sorted(sentences_with_num_tokens, key=minimum_word_count, reverse=True)
    filtered_sentences_by_num_tokens[num_tokens_value] = sentences_with_num_tokens[:200]

sentences = list(set.union(*(set(sentences) for sentences in filtered_sentences_by_num_tokens.values())))
```

## Translate Sentences to Chinese
Now that we have a curated list of English sentences, it's time to translate them into Chinese.

### Machine Translation:
We'll use a powerful language model, such as `Qwen/QwQ-32B-Preview`, to translate the English sentences into Chinese. To guide the model towards producing natural and beginner-friendly translations, we'll provide the following system prompt: 
```python
"You are an experienced translator."
```
And for each translation request, we'll use the following prompt:
```python
"Please translate into easy Chinese: \"{prompt}\""
```

### Generate Pinyin:
To facilitate pronunciation practice, we'll use the [pinyin-jyutping](https://github.com/Vocab-Apps/pinyin-jyutping) library to generate Pinyin transcriptions for each Chinese translation.

### Build a List of Dictionaries:
Finally, we'll create a list of dictionaries, where each dictionary represents a single entry in our Anki deck. Each dictionary will contain three key-value pairs:

* English sentence: The original English sentence.
* Mandarin translation: The Chinese translation generated by the LLM.
* Pinyin transcription: The Pinyin transcription of the Chinese translation.

Here is the combined code:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model
model_name = "Qwen/QwQ-32B-Preview"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define a function to get the Chinese translation for one sentence
def generate_response(prompt, template):
    messages = [
        {"role": "system", "content": "You are an experienced translator."},
        {"role": "user", "content": template.format(prompt=prompt)}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# Define translation template
translation_template = "Please translate into easy Chinese: \"{prompt}\""

# Translate sentences and get pinyin
for i, sentence in enumerate(sentences):
    translated_sentence = generate_response(sentence, translation_template).replace("\n", " ")
    translated_sentence = translated_sentence.strip('"')
    # Ignore too long outputs, likely something went wrong
    if len(translated_sentence) > 20:
        failure_count += 1
        continue
    
    # Get pinyin
    pinyin = pinyin_jyutping_sentence.pinyin(translated_sentence)
    pinyin = pinyin.strip('"')
    
    # Collect in dictionary
    mandarin = translated_sentence.strip()
    pinyin = pinyin.strip()
    translated_sentence_records.append({
        "sentence": sentence,
        "mandarin": mandarin,
        "pinyin": pinyin
    })

    # Custom progress bar
    print(" " * 100, end="\r")
    print(f"[{i+1}/{len(sentences)}/{failure_count}] Mandarin: {mandarin}, Pinyin: {pinyin}", end="\r")
```


## Create the Anki Deck
With our data prepared, we can now create the Anki deck. We'll create two card types:
1. English to Mandarin: The front of the card will display the English sentence, and the back will show the Mandarin translation and Pinyin transcription. The task is to know the Chinese translation and pronunciation.
2. Mandarin to English: The front of the card will display the Mandarin translation, and the back will show the English sentence and the Pinyin transcription.

We use [GenAnki](https://github.com/kerrickstaley/genanki) to create the Anki deck:
```python
import random
import genanki

model_id = random.randrange(1 << 30, 1 << 31)

chinese_deck_model = genanki.Model(
    model_id,
    'Chinese Sentence Model',
    fields=[
        {'name': 'English'},
        {'name': 'Mandarin'},
        {'name': 'Pinyin'},
        {'name': 'index'},
    ],
  templates=[
    {
      'name': 'English -> Mandarin',
      'qfmt': '{{English}}',
      # Show Mandarin and Pinyin side by side
      'afmt': '{{FrontSide}}<hr id="answer">{{Mandarin}}<br>{{Pinyin}}',
    },
    {
      'name': 'Mandarin -> English',
      'qfmt': '{{Mandarin}}',
      'afmt': '{{FrontSide}}<hr id="answer">{{Pinyin}}<br>{{English}}',
    },
  ])

# Sort translated_sentence_df by length of Mandarin
translated_sentence_df = translated_sentence_df.sort_values(by="mandarin", key=lambda x: x.str.len())

# Make notes for all rows in translated_sentence_df
chinese_sentences_deck = genanki.Deck(
    model_id + 1,
    'Chinese Sentences'
)

for index, row in translated_sentence_df.iterrows():
    chinese_sentences_deck.add_note(genanki.Note(
        model=chinese_deck_model,
        fields=[row["sentence"], row["mandarin"], row["pinyin"], str(index)],
        sort_field="index"
    ))

genanki.Package(chinese_sentences_deck).write_to_file('chinese_sentences.apkg')
```

## Conclusion
This blog demonstrates the power of AI in language learning. By employing a combination of NLP tools, sentence embedding models, and LLMs like Qwen/QwQ-32B-Preview, we are able to automate many aspects of the deck creation process, from sentence selection and deduplication to machine translation and Pinyin generation.

This not only significantly reduces the manual effort involved but also ensures a high-quality and personalized learning experience. Furthermore, we can easily adapt this method to other languages.
