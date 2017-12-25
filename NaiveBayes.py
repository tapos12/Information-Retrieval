from collections import defaultdict
from functools import reduce
from operator import mul
import re
import heapq

class NaiveBayes:
	def __init__(self, *corpora):
		self.priors = defaultdict(int)
		self.features = set()
		self.likelihood = defaultdict(lambda: defaultdict(int))
		self.establish_features(*corpora)

	def establish_features(self, *corpora):
		for corpus in corpora:
			for reviews in corpus:
				for word in reviews.reviewText:
					self.features.add(word)
	def train(self, reviews):
		prior_freqs = defaultdict(int)
		raw_likelihood = defaultdict(lambda: defaultdict(int))
		for i, review in enumerate(reviews, 1):
			prior_freqs[review.label] += 1

			for word in review.reviewText:
				raw_likelihood[review.label][word] += 1
			for word in review.reviewTitle:
				raw_likelihood[review.label][word] += 1

		for label, freq in prior_freqs.items():
			self.priors[label] = freq/i

		alpha = .005
		smoothing_div = alpha* len(self.features)
		for label in prior_freqs:
			label_freq = prior_freqs[label]
			for word in self.features:
				self.likelihood[label][word] = (raw_likelihood[label][word] + alpha) / (label_freq + smoothing_div)
		nlarge_gut = heapq.nlargest(15, self.likelihood['gut'], key=self.likelihood['gut'].get)
		nlarge_bad = heapq.nlargest(15, self.likelihood['schlecht'], key=self.likelihood['schlecht'].get)
		for item, item2 in zip(nlarge_gut, nlarge_bad):
			print(str(item))
			#self.likelihood['gut'].pop(item)
			#self.likelihood['schlecht'].pop(item2)

	def predict(self, review):
		prediction = []
		for c in self.priors.keys():
			class_likelihood = self.likelihood[c]
			p = self.priors[c]
			features = set()
			for word, word2 in zip(review.reviewText, review.reviewTitle):
				features.add(word)
				features.add(word2)
				l = reduce(mul, (class_likelihood[feature] for feature in features), 1)
				prediction.append((p*l, c))

		return max(prediction)[1]

class Corpus:
	def __init__(self, path):
		self.path = path
		self.reviews = []
		self._indices = []
		self.read_file()

	def __iter__(self):
		indices = self._indices[:]
		for i in indices:
			yield self.reviews[i]

	def read_file(self):
		with open(self.path) as handle:
			for line in handle:
				fields = line.strip('\n').split('\t')
				gameTitle, label, reviewTitle, reviewText = fields
				reviewText = reviewText.lower()
				reviewText = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", reviewText) #digits
				reviewText =  re.sub(r'[^\w\s]','',reviewText) #punctuation

				reviewTitle = reviewTitle.lower()
				reviewTitle = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", reviewTitle) #digits
				reviewTitle =  re.sub(r'[^\w\s]','',reviewTitle) #punctuation
				review = Review(gameTitle, label, tuple(reviewTitle.split(' ')), tuple(reviewText.split(' ')))
				self.reviews.append(review)
				self._indices.append(len(self._indices))
		"""
		for item in self.reviews:
			for attr, value in item.__dict__.items():
				if attr == 'reviewTitle':
					print(str(attr)+" "+str(value))
					"""
	def evaluate(self):
        #if not self.review:
        #    raise EmptyCorpus
		labels = set()
		tp = defaultdict(int)
		fp = defaultdict(int)
		fn = defaultdict(int)

		global_tp = 0
		global_fp = 0

		for review in self.reviews:
			if review.prediction is None:
				continue
			labels.add(review.label)
			labels.add(review.prediction)
			if review.label == review.prediction:
				tp[review.prediction] += 1
				global_tp += 1
			else:
				fp[review.prediction] += 1
				fn[review.label] += 1
				global_fp += 1

		eval_results = {}
		eval_results['accuracy'] = 100*(global_tp/(global_tp+global_fp))
		eval_results['labels'] = {}

		for label in sorted(labels):
			eval_results['labels'][label] = {}
			try:
				precision = 100*(tp[label]/(tp[label]+fp[label]))
			except ZeroDivisionError:
				precision = 0
			try:
				recall = 100*(tp[label]/(tp[label]+fn[label]))
			except ZeroDivisionError:
				recall = 0
			try:
				f_score = 2*((precision*recall)/(precision+recall))
			except ZeroDivisionError:
				f_score = 0

			eval_results['labels'][label]['precision'] = precision
			eval_results['labels'][label]['recall'] = recall
			eval_results['labels'][label]['f_score'] = f_score
		
		eval_results['macro_avg_f_score'] = sum(eval_results['labels'][label]['f_score'] for label in labels)/len(labels)
		return eval_results

class Review:
	def __init__(self, gameTitle, label, reviewTitle, reviewText, prediction=None):
		self.gameTitle = gameTitle
		self.label = label
		self.reviewTitle = reviewTitle
		self.reviewText = reviewText
		self.prediction = prediction

def pformat(eval_results):
    string = '\r'
    eval_width = 10
    label_width = len(max(eval_results['labels'], key=lambda x: len(x)))

    string += ' '*eval_width
    for emotion in sorted(eval_results['labels']):
        string += ' %s ' %emotion.rjust(label_width)

    string += '\n%s' %'Precision'.rjust(eval_width)
    for label in sorted(eval_results['labels']):
        precision = '%.2f' %eval_results['labels'][label]['precision']
        string += ' %s ' %precision.rjust(label_width)

    string += '\n%s' %'Recall'.rjust(eval_width)
    for label in sorted(eval_results['labels']):
        recall = '%.2f' %eval_results['labels'][label]['recall']
        string += ' %s ' %recall.rjust(label_width)

    string += '\n%s' %'F-Score'.rjust(eval_width)
    for label in sorted(eval_results['labels']):
        f_score = '%.2f' %eval_results['labels'][label]['f_score']
        string += ' %s ' %f_score.rjust(label_width)

    return string

if __name__ == "__main__":
	train_corpus = Corpus('games-train.csv')
	test_corpus = Corpus('games-test.csv')
	Nb = NaiveBayes(train_corpus, test_corpus)
	Nb.train(train_corpus)
	#i=0
	#j=0
	for review in test_corpus:
		review.prediction = Nb.predict(review)
		"""if pred == review.label:
			i+=1
		j+=1"""
	eval_results = test_corpus.evaluate()
	print('\r'+pformat(eval_results)+'\n')
	print(' == Macro Avg F_Score: %.2f ==\n' %(eval_results['macro_avg_f_score']))
	print(' == Eval Accuracy:     %.2f ==\n' %(eval_results['accuracy']))
	print(' %s\n' %('-'*78))

	#print("acc:" + str(i/j))
