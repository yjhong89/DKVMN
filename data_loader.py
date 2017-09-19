import numpy as np
import tensorflow as tf

class DATA_LOADER():
	def __init__(self, n_questions, seqlen, seperate_char):
		# assist2009 : seq_len(200), n_questions(110)
		# Each value is seperated by seperate_char
		self.seperate_char = seperate_char
		self.n_questions = n_questions
		self.seq_len = seqlen

	'''
	Data format as followed
	1) Number of exercies
	2) Exercise tag
	3) Answers
	'''
	# path : data location
	def load_data(self, path):
		f_data = open(path, 'r')
		# Question/Answer container
		q_data = list()
		qa_data = list()
		# Read data
		for lineid, line in enumerate(f_data):
			# strip
			line = line.strip()
			# Exercise tag line
			if lineid % 3 == 1:
				# split by ',', returns tag list
				print 'Excercies tag',
				q_tag_list = line.split(self.seperate_char)
			
			# Answer
			elif lineid % 3 == 2:
				print(', Answers')
				answer_list = line.split(self.seperate_char)
			
				# Divide case by seq_len
				if len(q_tag_list) > self.seq_len:
					n_split = len(q_tag_list) // self.seq_len
					if len(q_tag_list) % self.seq_len:
						n_split += 1
				else:
					n_split = 1
				print('Number of split : %d' % n_split)
	
				# Contain as many as seq_len, then contain remainder
				for k in xrange(n_split):
					q_container = list()
					qa_container = list()
					# Less than 'seq_len' element remained
					if k == n_split - 1:
						end_index = len(answer_list)
					else:
						end_index = (k+1)*self.seq_len
					for i in xrange(k*self.seq_len, end_index):
						# answers in {0,1}
						qa_values = int(q_tag_list[i]) + int(answer_list[i]) * self.n_questions
						q_container.append(int(q_tag_list[i]))
						qa_container.append(qa_values)
						print('Question tag : %s, Answer : %s, QA : %s' %(q_tag_list[i], answer_list[i], qa_values))
					# List of list(seq_len, seq_len, seq_len, less than seq_len, seq_len, seq_len...
					q_data.append(q_container)
					qa_data.append(qa_container)
		f_data.close()

		# Convert it to numpy array
		q_data_array = np.zeros((len(q_data), self.seq_len))
		for i in xrange(len(q_data)):
			data = q_data[i]
			# if q_data[i] less than seq_len, remainder would be 0 
			q_data_array[i, :len(data)] = data

		qa_data_array = np.zeros((len(qa_data), self.seq_len))
		for i in xrange(len(qa_data)):
			data = qa_data[i]
			# if qa_data[i] less than seq_len, remainder would be 0
			qa_data_array[i,:len(data)] = data

		return q_data_array, qa_data_array

