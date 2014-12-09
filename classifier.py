from gensim.models import Word2Vec
import pdb
from collections import Counter

f = open('stopwords.txt')
stoplist = set(line.split('\n')[0] for line in f)
docs = []

def transform_news_article(news_file, model):
	with open(news_file, 'rb') as f:
		doc = []
		for line in f:
			words = filter(lambda x: x in model.vocab, line.split())
			words = [w for w in words if w not in stoplist]
			doc += words
	docs.append(doc)
	return doc

def label_news_group(news_articles, label, model):
	labeled_articles = []
	for news_article in news_articles:
		processed_article = transform_news_article(news_article, model)
		labeled_articles.append((label,processed_article))
	return labeled_articles

def separate_training_and_test_examples(news_articles, num_train):
	train_examples, test_examples = [], []
	for idx, article in enumerate(news_articles):
		if idx < num_train:
			train_examples.append(article)
		else:
			test_examples.append(article)
	return train_examples, test_examples

def knn_classifier(k, model, train_examples, test_example):
	sims = []
	for i, train_example in enumerate(train_examples):
		sim = model.n_similarity(train_example[1], test_example[1])
		# print "SIM:", sim, "idx:", i, "label:", train_example[0]
		sims.append((sim, train_example[0]))
	sims.sort(reverse=True)
	top_neighbors = sims[:k]
	mode_counter = Counter(top_neighbors)
	predicted_label = mode_counter.most_common(1)[0][0][1] # label
	return predicted_label

def run_tests(test_examples, train_examples, k, model):
	num_total = len(test_examples)
	num_correct = 0
	for test_example in test_examples:
		predicted_label = knn_classifier(k, model, train_examples, test_example)
		#print "ACTUAL:", test_example[0], "PREDICTED:", predicted_label
		if predicted_label == test_example[0]:
			num_correct += 1
	accuracy = num_correct/float(num_total)
	print "K-Value:", k, "ACCCURACY:", accuracy
	return accuracy

def get_articles(label, num_articles, model):
	news_articles = []
	for i in range(num_articles):
		file_name = "data/" + label + "_" + str(i)
		article = transform_news_article(file_name, model)
		news_articles.append((label,article))
	return news_articles

def main():
	model = Word2Vec.load_word2vec_format('googlenews_model.bin', binary=True)
	#model = 0
	test_examples, train_examples = [], []
	science_articles = get_articles("science", 100, model)
	sports_articles = get_articles("sports", 100, model)
	entertainment_articles = get_articles("entertainment", 100, model)
	science_train, science_test = separate_training_and_test_examples(science_articles, 75)
	sports_train, sports_test = separate_training_and_test_examples(sports_articles, 75)
	entertainment_train, entertainment_test = separate_training_and_test_examples(entertainment_articles, 75)
	test_examples = entertainment_test + science_test + sports_test
	train_examples = entertainment_train + science_train + sports_train
	for i in range(1, 101):
		run_tests(test_examples, train_examples, i, model)

if __name__ == "__main__":
    main()
