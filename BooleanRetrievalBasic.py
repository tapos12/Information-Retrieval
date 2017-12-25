"""
@author: Anna Khlyzova, Cennet Oguz and Touhidul Alam
"""

from nltk import wordpunct_tokenize
import uuid
from collections import defaultdict
import re

#this holds the individual term and their corresponding tweet id
class TermList():
	def __init__(self, term, doc_id):
		self.term = term
		self.doc_id = doc_id

#this data structure store the term, size of posting list and the pointer to the posting list
class BuildDictionary():
	def __init__(self, term, size_of_postinglist,pointer_of_postinglist):
		self.term = term
		self.size_of_postinglist = size_of_postinglist
		self.pointer_of_postinglist = pointer_of_postinglist

#our main class, which build index and query and store all posting list in memory
class BooleanRetrieval():
	def __init__(self, tweets_list, postings_list, term_dict_list):
		self.tweets_list = tweets_list
		self.postings_list = postings_list
		self.term_dict_list = term_dict_list

	#this takes the input tweet data and store tweet and tweet id on a tweet list
	def index(self, filename):
		raw_data = open(filename)
		data_lines = raw_data.read().split('\n')
		raw_data.close()

		for line in data_lines:
			tweet_fields = line.strip('\n').split('\t')
			#length 5 were mentioned as some tweet we found, tweet is empty, we ignore those
			if(len(tweet_fields)==5):
				#we only take account of tweet id and tweet content
				_, tweet_id, _, _, content = tweet_fields
				self.tweets_list[tweet_id] = content
		return self.tweets_list

	def main(self):
		"""index function take an argument as "tweet" file, if the file is on different directories, it needs to provide
		full directories"""
		tweets = self.index("tweets")
		#tokenization function tokenize the terms
		tokens = self.tokenization(tweets)
		#generate_posting_list generate posting for each terms
		self.postings_list = self.generate_postings_list(tokens)

		#this shows posting list of a single term
		#self.query("stuttgart")
		#this shows intersected posting list of two terms
		self.query("stuttgart","bahn")

	def tokenization(self, tweets):
		tokens = []
		emoji_pattern = re.compile("["
		        u"\U0001F600-\U0001F64F"  # emoticons
		        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
		        u"\U0001F680-\U0001F6FF"  # transport & map symbols
		        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
		                           "]+", flags=re.UNICODE)
		#we normalize based on digits, punctuation, emojis and lower case them and store the unique words only
		for tweet_id in tweets:
			tweet = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", tweets.get(tweet_id)) #digits
			tweet =  re.sub(r'[^\w\s]','',tweet) #punctuation
			tweet = emoji_pattern.sub(r'', tweet) #emojis
			unique_words = set(wordpunct_tokenize(tweet.lower())) #unique words in each tweet, we use nltk library for this
			for term in unique_words:
				token_list = TermList(term,tweet_id)
				tokens.append(token_list)
		#after storing every unique term and their tweet_id on list, we sort them based on term
		tokens.sort(key = lambda x: x.term)
		return tokens
				
	def generate_postings_list(self, token_list):
		term_doc_list = defaultdict(list)
		for line in token_list:
			#in the list, if a same term is found in more than one doc, we append them on the list
			if line.term in term_doc_list:
				term_doc_list[line.term].append(line.doc_id)
			else:
				term_doc_list[line.term] = [line.doc_id]

		for item in term_doc_list:
			#we use a random identifier UUID as a pointer for our build-dictionary with the term to the posting list
			pointer_of_postingslist = str(uuid.uuid4())
			#this random number points to the term
			self.postings_list[pointer_of_postingslist] = term_doc_list.get(item)
			size_of_postingslist = len(term_doc_list.get(item))
			#this build-dictionary stores, terms, size of posting list and pointer to the posting list of the list
			term_dictionary = BuildDictionary(item , size_of_postingslist, pointer_of_postingslist)
			self.term_dict_list.append(term_dictionary)

		return self.postings_list

	"""this function computes the query operation of the given term. If a single term is provided it shows
	the term, size of the term, and it'sposting list. If two terms are given, it shows the 'AND' intersection
	and writes the posting list of the intersected twitter id and their respective tweets on a file."""
	def query(self, term1, term2=None):
		#for two terms
		if(term2!=None):
			term1_posting = []
			term2_posting = []
			intersect_posting = []

			for xy in self.term_dict_list:
				#fetching the posting list of the respective terms and sorting them
				if(xy.term == term1):
					term1_posting = sorted(self.postings_list.get(xy.pointer_of_postinglist))
				if(xy.term == term2):
					term2_posting = sorted(self.postings_list.get(xy.pointer_of_postinglist))
			#iteration to go to the next item of the set
			term1_iteration = iter(term1_posting)
			term2_iteration = iter(term2_posting)

			#fetching the first item from the iteration
			current_term1 = next(term1_iteration, None)
			current_term2 = next(term2_iteration, None)

			#go over the loop until one of the item set has been finished
			while current_term1 != None and current_term2 != None:
				#while a match found in both list, it stores the intersected item
				if current_term1 == current_term2:
					intersect_posting.append(current_term1)
					current_term1 = next(term1_iteration, None)
					current_term2 = next(term2_iteration, None)
				#only move the valued set to next item
				elif current_term1 > current_term2:
					current_term2 = next(term2_iteration, None)
				else:
					current_term1 = next(term1_iteration, None)

			print ("\nMerged Output: " + str(intersect_posting))
			print ("\nTotal merged document: " + str(len(intersect_posting)))

			count = 0
			myfile = open('output.txt', 'w')
			for document_id in intersect_posting:
				count += 1
				#write the intersected items, their tweet id and tweet data to the file
				myfile.write("\n"+ "Item: "+str(count)+" TweetID: " + str(document_id) + " Tweet: " + self.tweets_list[document_id])
			myfile.close()
		#for single term
		else:
			for xy in self.term_dict_list:
				if(xy.term == term1):
					print(xy.term + " " + str(xy.size_of_postinglist) + " " + str(self.postings_list.get(xy.pointer_of_postinglist)))

if __name__ == "__main__":
	#we use tweet_list dictionary to store individual tweet based on its twitter id
	tweets_list = {}
	#we use posting_list as a dictionary to hold the posting_list as list value and a unique identifier as id of term
	postings_list = defaultdict(list)
	#we use this list data structure to hold the term, their size of posting list, and posting list data
	term_dict_list = []
	BooleanRetrieval(tweets_list, postings_list, term_dict_list).main()
