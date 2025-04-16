from flask import Flask, request, jsonify, render_template
import bs4 as bs
import urllib.request as url
import re
import nltk
import heapq
from string import punctuation

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

def summarize_article(article_url):
    # Fetch and parse the article
    scraped_data = url.urlopen(article_url)
    article = scraped_data.read()
    parsed_article = bs.BeautifulSoup(article, 'lxml')
    paragraphs = parsed_article.find_all('p')
    article_text = " ".join([p.text for p in paragraphs])

    # Preprocess the text
    article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)
    article_text = re.sub(r'\s+', ' ', article_text)
    formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text)
    formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)

    # Tokenize sentences
    sentence_list = nltk.sent_tokenize(article_text)

    # Find weighted frequency of occurrence
    stopwords = nltk.corpus.stopwords.words('english')
    word_frequencies = {}
    for word in nltk.word_tokenize(formatted_article_text):
        if word not in stopwords and word not in punctuation:
            word_frequencies[word] = word_frequencies.get(word, 0) + 1

    maximum_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] /= maximum_frequency

    # Calculate sentence scores
    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    sentence_scores[sent] = sentence_scores.get(sent, 0) + word_frequencies[word]

    # Extract summary
    summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)
    return summary

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    article_url = data.get('url')
    if not article_url:
        return jsonify({'error': 'No URL provided'}), 400
    try:
        summary = summarize_article(article_url)
        return jsonify({'summary': summary})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
