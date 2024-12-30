import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Download stopwords
nltk.download("stopwords")
nltk.download("punkt")

# Example text for summarization
text = """
Artificial Intelligence (AI) is a branch of computer science that focuses on creating machines capable of simulating human intelligence. It enables systems to learn from data, recognize patterns, and make decisions with minimal human intervention. AI is widely applied in various fields such as healthcare, finance, education, and entertainment. Technologies like machine learning, natural language processing, and computer vision are integral to AI development. As AI continues to evolve, it promises to bring transformative changes to industries and everyday life.
"""

# Function to generate a frequency-based summary
def summarize_text(text, num_sentences=2):
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())

    # Filter out stopwords and non-alphabetic words
    stop_words = set(stopwords.words("english"))
    word_frequencies = {}

    for word in words:
        if word.isalpha() and word not in stop_words:
            word_frequencies[word] = word_frequencies.get(word, 0) + 1

    # Score each sentence based on work frequency
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_frequencies:
                sentence_scores[sentence] = sentence_scores.get(sentence, 0) + word_frequencies[word]

    # Sort sentences by score and select the top 'num_sentences'
    summarize_sentence = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    summary = ' '.join(summarize_sentence)
    return summary

# Generate and print the summary
summary = summarize_text(text, num_sentences=2)
print(summary)