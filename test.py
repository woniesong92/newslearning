from gensim.models import Word2Vec
import pdb

f = open('stopwords.txt')
stoplist = set(line.split('\n')[0] for line in f)

def transform_news_article(news_file, model):
	with open(news_file, 'rb') as f:
		for line in f:
			words = filter(lambda x: x in model.vocab, line.split())
			words = [w for w in words if w not in stoplist]
	return words

def main():
	model = Word2Vec.load_word2vec_format('googlenews_model.bin', binary=True)
	science1 = transform_news_article("s_1", model)
	science2 = transform_news_article("s_2", model)
	science3 = transform_news_article("s_3", model)
	entertainment1 = transform_news_article("e_1", model)
	sports_1 = transform_news_article("sports", model)
	pdb.set_trace()
	print "science1 and science2:", model.n_similarity(science1, science2)
	print "science1 and science3:", model.n_similarity(science1, science3)
	print "science2 and science3:", model.n_similarity(science2, science3)
	print "science1 and entertainment1:", model.n_similarity(science1, entertainment1)
	print "science1 and sports1:", model.n_similarity(science1, sports_1)

main()
