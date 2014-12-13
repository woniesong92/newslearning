from gensim.models import Word2Vec
import pdb
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC as SupportVectorClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import codecs
from sklearn.lda import LDA
from sklearn import tree

f = open('stopwords.txt')
stoplist = set(line.split('\n')[0] for line in f)
docs = []


def transform_news_article(news_file, model):
	with open(news_file, 'rb') as f:
		doc = []
		for line in f:
			#words = filter(lambda x: x in model.vocab, line.split())
			words = line.split()
			words = [w for w in words if w not in stoplist]
			doc += words
	#docs.append(doc)
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

def get_knn_classifier_with_stats(labels, feature_vectors, k=5):
	knn = KNeighborsClassifier(n_neighbors=k)
	knn.fit(feature_vectors, labels)
	return knn

def get_svm_classifier_with_stats(labels, feature_vectors):
	svm = SupportVectorClassifier()
	svm.fit(feature_vectors, labels)
	return svm

def get_randomforest_classifier_with_stats(labels, feature_vectors):
	randomforest = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0)
	randomforest.fit(feature_vectors, labels)
	return randomforest

def get_lda(labels, feature_vectors):
	lda = LDA()
	lda.fit(feature_vectors, labels)
	return lda

def get_decision_tree(labels, feature_vectors):
	dt = tree.DecisionTreeClassifier()
	dt = dt.fit(feature_vectors, labels)
	return dt

def run_tests(test_examples, train_examples, k, model):
	num_total = len(test_examples)
	num_correct = 0
	for test_example in test_examples:
		predicted_label = knn_classifier(k, model, train_examples, test_example)
		#print "ACTUAL:", test_example[0], "PREDICTED:", predicted_label
		if predicted_label == test_example[0]:
			num_correct += 1
	accuracy = num_correct/float(num_total)
	return accuracy

def get_articles(label, num_articles, model):
	news_articles = []
	for i in range(num_articles):
		file_name = "data/" + label + "_" + str(i)
		article = transform_news_article(file_name, model)
		news_articles.append((label,article))
	return news_articles

class Model:
	vocab = ["hello", "hi", "hey", "whatsup"]

def get_num_unique_words(words):
	s = set()
	for word in words:
		if word not in s:
			s.add(word)
	return len(s)

def extract_features(examples):
	labels = []
	feature_vectors = []
	for i, example in enumerate(examples):
		labels.append(example[0])
		words = example[1]
		num_words = len(words)
		num_unique_words = get_num_unique_words(words)
		length_article = sum(map(lambda x: len(x), words))
		feature_vectors.append([num_words, num_unique_words, length_article])
	return labels, feature_vectors

def norm_l1(feature_vector):
	sum_features = sum(feature_vector)
	new_feature_vector = []
	for feature in feature_vector:
		new_feature_vector.append(feature/float(sum_features))
	return new_feature_vector

def run_knn_test(test_examples, test_labels, knn):
	predicted_labels = []
	num_correct = 0
	for example in test_examples:
		predicted_label = knn.predict(example)[0]
		predicted_labels.append(predicted_label)
	for i in range(len(predicted_labels)):
		# print "PREDICTED:", predicted_labels[i], "ACTUAL:", test_labels[i]
		if predicted_labels[i] == test_labels[i]:
			num_correct += 1
	accuracy = num_correct / float(len(test_examples))
	return accuracy

def run_generic_test(test_examples, test_labels, classifier):
	predicted_labels = []
	num_correct = 0
	for example in test_examples:
		predicted_label = classifier.predict(example)[0]
		predicted_labels.append(predicted_label)
	for i in range(len(predicted_labels)):
		# print "PREDICTED:", predicted_labels[i], "ACTUAL:", test_labels[i]
		if predicted_labels[i] == test_labels[i]:
			num_correct += 1
	accuracy = num_correct / float(len(test_examples))
	return accuracy

def draw_k_values(accuracy_lst):
	x_vals = range(1, len(accuracy_lst)+1)
	label1, = plt.plot(x_vals, accuracy_lst, 'r-', label='label1')
	plt.legend([label1], ['accuracy'])
	plt.xticks(range(0, 101, 5))
	plt.xlabel('k-value')
	plt.show()

def extract_bodies(train_examples):
	article_bodies = []
	for example in train_examples:
		body = ' '.join(example[1])
		article_bodies.append(body)
	return article_bodies

def vectorize_using_tfidf(train_examples):
	news_bodies = extract_bodies(train_examples[:150])
	tfidf = TfidfVectorizer(norm='l1')
	pdb.set_trace()
	tfs = tfidf.fit_transform(news_bodies)



def main():
	model = Model()
	test_examples, train_examples = [], []

	print "getting articles..."
	science_articles = get_articles("science", 100, model)
	sports_articles = get_articles("sports", 100, model)
	entertainment_articles = get_articles("entertainment", 100, model)

	print "got all the articles.."
	science_train, science_test = separate_training_and_test_examples(science_articles, 75)
	sports_train, sports_test = separate_training_and_test_examples(sports_articles, 75)
	entertainment_train, entertainment_test = separate_training_and_test_examples(entertainment_articles, 75)
	test_examples = entertainment_test + science_test + sports_test
	train_examples = entertainment_train + science_train + sports_train

	print "extract_features..."
	train_labels, train_features = extract_features(train_examples)
	test_labels, test_features = extract_features(test_examples)

	train_features = [norm_l1(x) for x in train_features]
	test_features = [norm_l1(y) for y in test_features]


	# KNN TESTING
	# accuracy_lst = []
	# for k in range(1,101):	
	# 	knn_classifier = get_knn_classifier_with_stats(train_labels, train_features, k=k)
	# 	accuracy = run_knn_test(test_features, test_labels, knn_classifier)
	# 	print "k-value:", k, "accuracy:", accuracy
	# 	accuracy_lst.append(accuracy)
	# draw_k_values(accuracy_lst)
	# KNN accuracy = 0.706666666667

	k = 70
	knn_classifier = get_knn_classifier_with_stats(train_labels, train_features, k=k)
	accuracy = run_knn_test(test_features, test_labels, knn_classifier)
	print "KNN with k=%i" % k, accuracy

	# SVM TESTING
	svm_classifier = get_svm_classifier_with_stats(train_labels, train_features)
	accuracy = run_generic_test(test_features, test_labels, svm_classifier)
	print "SVM", accuracy
	# SVM accuracy = 0.6133333333333333

	# Random Forest testing
	random_forest = get_randomforest_classifier_with_stats(train_labels, train_features)
	accuracy = run_generic_test(test_features, test_labels, random_forest)
	print "Random Forest", accuracy
	# RF accuracy = 0.5333


	#LDA
	lda = get_lda(train_labels, train_features)
	accuracy = run_generic_test(test_features, test_labels, lda)
	print "LDA", accuracy
	# LDA accuracy = 0.65333333

	#Decision Tree
	dt = get_decision_tree(train_labels, train_features)
	accuracy = run_generic_test(test_features, test_labels, dt)
	print "Decision Tree", accuracy
	# Decision tree accuracy = 0.613333333333

	# pdb.set_trace()

	# # LDA
	# lda = get_lda(test_features, test_labels)
	# pdb.set_trace()
	# accuracy = run_generic_test(test_features, test_labels, lda)
	# pdb.set_trace()

	# print "separated examples... running tests."
	# for i in range(1, 101):
	# 	run_tests(test_examples, train_examples, i, model)

if __name__ == "__main__":
    main()
