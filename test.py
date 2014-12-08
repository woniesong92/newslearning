from gensim.models import Word2Vec
import pdb
from collections import Counter

f = open('stopwords.txt')
stoplist = set(line.split('\n')[0] for line in f)

def transform_news_article(news_file, model):
	with open(news_file, 'rb') as f:
		for line in f:
			words = filter(lambda x: x in model.vocab, line.split())
			words = [w for w in words if w not in stoplist]
	return words

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
	for train_example in train_examples:
		sim = model.n_similarity(train_example, test_example)
		sims.append(sim)
	sims.sort()
	top_neighbors = sims[:k]
	d = Counter(top_neighbors)
	d.most_common(1)[0] # label

def run_tests(test_examples, train_examples, k, model):
	num_total = len(test_examples)
	num_correct = 0
	for test_example in test_examples:
		predicted_label = knn_classifier(k, model, train_examples, test_example)
		print "PREDICTED:", predicted_label, "ACTUAL:", test_example[0]
		if predicted_label == test_example[0]:
			num_correct += 1
	accuracy = num_correct/float(num_total)
	print "ACCCURACY:", accuracy
	return accuracy

def get_articles(label, num_articles, model):
	news_articles = []
	for i in range(num_articles):
		file_name = label + "_" + str(i)
		with open(file_name, 'rb') as f:
			article = transform_news_article(f, model)
			news_articles.append((label,article))
	return news_articles



def main():
	model = Word2Vec.load_word2vec_format('googlenews_model.bin', binary=True)
	test_examples, train_examples = [], []
	science_articles = get_articles("science", 100, model)
	sports_articles = get_articles("sports", 100, model)
	entertainment_articles = get_articles("entertainment", 50, model)
	# science_articles = label_news_group(science_articles, "science", model)
	# sports_articles = label_news_group(sports_articles, "sports", model)
	# entertainment_articles = label_news_group(entertainment_articles, "entertainment", model)
	pdb.set_trace()
	science_train, science_test = separate_training_and_test_examples(science_articles, 75)
	sports_train, sports_test = separate_training_and_test_examples(sports_articles, 75)
	entertainment_train, entertainment_test = separate_training_and_test_examples(entertainment_articles, 25)
	test_examples = entertainment_test + science_test + sports_test
	train_examples = entertainment_train + science_train + sports_train
	run_tests(test_examples, train_examples, 3, model)
	pdb.set_trace()

if __name__ == "__main__":
    main()
