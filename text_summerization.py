import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest


def summerizer(rawtext):
    nlp = spacy.load('en_core_web_sm')
    stopwords = list(STOP_WORDS)

    doc = nlp(rawtext)

    word_freq = {}
    for word in doc:
        if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
            if word.text not in word_freq:
                word_freq[word.text] = 1
            else:
                word_freq[word.text] += 1

    max_freq = max(word_freq.values())
    for word in word_freq:
        word_freq[word] = word_freq[word] / max_freq

    sent_scores = {}
    for sent in doc.sents:
        for word in sent:
            if word.text in word_freq:
                if sent not in sent_scores:
                    sent_scores[sent] = word_freq[word.text]
                else:
                    sent_scores[sent] += word_freq[word.text]

    select_len = int(len(list(doc.sents)) * 0.3)
    summary_sentences = nlargest(select_len, sent_scores, key=sent_scores.get)
    summary = ' '.join([str(sent) for sent in summary_sentences])

    return rawtext, summary, len(rawtext.split()), len(summary.split())
