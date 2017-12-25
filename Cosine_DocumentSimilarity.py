import math
from random import randrange
import re
import heapq

class Cosine_sim():

	def __init__(self, tweets_list, summ, tf_idf1, common):
		self.tweets_list = tweets_list
		self.summ = summ
		self.tf_idf1 = tf_idf1
		self.common = common

	#this takes the input tweet data and store tweet and tweet id on a tweet list
	def index(self, filename):
		raw_data = open(filename)
		data_lines = raw_data.read().split('\n')
		raw_data.close()

		i = 0
		for line in data_lines:
			tweet_fields = line.strip('\n').split('\t')
			#length 5 were mentioned as some tweet we found, tweet is empty, we ignore those
			if(len(tweet_fields)==5):
				#we only take account of tweet id and tweet content
				_, tweet_id, _, _, content = tweet_fields
				#if the tweet is any other alphabet than English, we skip it
				if not self.isEnglish(content):
					continue
				# we skip links containing 'http'/'https'
				content = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', " ",content)
				# we skip @{USERNAME}
				content = re.sub('^@[^[]+', " ", content)
				content = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", content)
				content =  re.sub(r'[^\w\s]'," ",content)
				content = content.lower()
				self.tweets_list[i] = content
				i += 1
		return self.tweets_list

	#we only consider tweet that contains English alphabets only
	def isEnglish(self, s):
	    try:
	        s.encode(encoding='utf-8').decode('ascii')
	    except UnicodeDecodeError:
	        return False
	    else:
	        return True

	def main(self):
		tweet = self.index('tweets')
		# we choose a random tweet to find similarity with other tweets
		random_index = randrange(0,len(tweet))
		# we considering tweet that has atleast one word
		while len(tweet[random_index]) == 0:
			random_index = randrange(0,len(tweet))
		#print(tweet[random_index])

		doc1 = tweet[random_index]
		#we compute tf and tf-idf score for primary tweet we would compare with
		# we also compute idf for each existing word
		tf_doc1 = self.tf(doc1)
		idf = self.idf(tweet)
		for item in set(doc1.split(" ")):
			self.tf_idf1[item] = tf_doc1[item] * idf[item]

		#we compute magnitude of the doc1 and store it 
		self.summ = sum(self.tf_idf1[x]**2 for x in self.tf_idf1)
		self.summ = math.sqrt(self.summ)
		doc1 = tweet[random_index].split()
		maxdict = {}
		for i in range(0,len(tweet)):

			doc2 = tweet[i].split()
			#we take the intersection of primary tweets with other every tweets 
			self.common = set(doc1).intersection(set(doc2))
			#if the intersection set is 0, the numerator would be 0, so the cosine-sim 0, we skip this
			if not self.common:
				continue
			if tweet[i] == tweet[random_index]:
				continue
			doc2 = tweet[i]
			tf_idf2 = {}
			#we compute tf for each documents
			tf_doc2 = self.tf(doc2)

			for item in set(doc2.split(" ")):
				tf_idf2[item] = tf_doc2[item] * idf[item]

			#we store the cosine_score in maxdict for each doc pairs score
			maxdict[i] = self.cosine(tf_idf2)
			
		#store the largest 100 from the dictionary based on its cosine score
		nlarge = heapq.nlargest(100, maxdict, key=maxdict.get)
		count = 0
		myfile = open('output.txt', 'w')
		myfile.write("Main doc: " + tweet[random_index])
		for item in nlarge:
			count += 1
			#it writes the cosine similarity score and corresponding normalized tweets
			myfile.write("\n"+ str(maxdict[item])+": " + tweet[item])
		myfile.close()

	def tf(self, doc):
		tf_dict = {}
		for item in doc.split(" "):
			if item not in tf_dict:
				tf_dict[item] = 0
			tf_dict[item]+= 1
		#after counting the term, it updates according to formula
		tf_dict.update((k, 1+math.log10(v)) for k,v in tf_dict.items())
		return tf_dict

	def idf(self, doclist):
		df_dict = {}
		for i in range(0, len(doclist)):
			for item in set(doclist[i].split(" ")):
				if item not in df_dict:
					df_dict[item] = 0
				df_dict[item] += 1
		df_dict.update((k, math.log10(len(doclist)/v)) for k,v in df_dict.items())
		return df_dict

	def cosine(self, tfidf2):
		#numerator, dot product of two intersected documents tf-idf value
		numerator = sum(self.tf_idf1[x] * tfidf2[x] for x in self.common)

		sum2 = sum(tfidf2[x]**2 for x in tfidf2)
		#denominator, compute magnitude of two documents
		denominator = self.summ * math.sqrt(sum2)

		if not denominator:
			return 0.0
		else:
			return float(numerator) / denominator
if __name__ == "__main__":
	tweets_list = {}
	summ = 0.0
	tf_idf1 = {}
	common = set()
	#we initialize tweet_list for storing tweet data, 
	Cosine_sim(tweets_list, summ, tf_idf1, common).main()