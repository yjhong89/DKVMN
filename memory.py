import numpy as np
import os
import tensorflow as tf
import operations


# This class defines Memory architecture in DKVMN
class DKVMN_Memory():
	def __init__(self, memory_size, memory_state_dim, name):
		self.name = name
		# Memory size : N
		self.memory_size = memory_size
		# Memory state dim : D_V or D_K
		self.memory_state_dim = memory_state_dim
		'''
			Key matrix or Value matrix
			Key matrix is used for calculating correlation weight(attention weight)
		'''
			

	def cor_weight(self, embedded, key_matrix, name=self.name+'Correlation_Weight'):
		'''
			embedded : [batch size, memory state dim(d_k)]
			Key_matrix : [memory size * memory state dim(d_k)]
			Correlation weight : w(i) = k * Key matrix(i)
			=> batch size * memory size
		'''	
   		# embedding_result : [batch size, memory size], each row contains each concept correlation weight for 1 question
		embedding_result = tf.matmul(embedded, tf.transpose(key_matrix))
		correlation_weight = tf.nn.softmax(q_embedding_result)
		print('Correlation weight shape : %s' % (correlation_weight.shape))
		return correlation_weight

	
	# Getting read content
	def read(self, value_matrix, correlation_weight):
		'''
			Value matrix : batch size * memory size * memory state dim
			Correlation weight : batch size * memory size, each element represents each concept embedding for 1 question
		'''
		# Reshaping
		# [batch size * memory size, memory state dim(d_v)]
		vmtx_reshaped = np.reshape(value_matrix, [-1, self.memory_state_dim])
		# [batch size * memory size, 1]
		cw_reshaped = np.reshape(correlation_weight, [-1,1])		
		# Read content, will be [batch size * memory size, memory state dim] and reshape it to [batch size, memory size, memory state dim]
		read_content = np.reshape(np.multiply(vmtx_reshaped, cw_reshaped), [-1,self.memory_size,self.memory_state_dim])
		# Summation through memory size axis, make it [batch size, memory state dim(d_v)]
		read_content = np.sum(read_content, axis=1)
		print('Read content shape : %s' % (read_content.shape))
		return read_content


	def write(self, value_matrix, correlation matrix, qa_embedded, reuse=False):
		'''
			Value matrix : [batch size, memory size, memory state dim(d_k)]
			Correlation weight : [batch size, memory size]
			qa_embedded : (q, r) pair embedded, [batch size, memory state dim(d_v)]
		'''
		erase_vector = operations.linear(qa_embedded, self.memory_state_dim, name=self.name+'/Erase_Vector', reuse=reuse)
		# [batch size, memory state dim(d_v)]
		erase_signal = tf.sigmoid(erase_vector)
		add_vector = operations.linear(qa_embedded, self.memory_state_dim, name=self.name+'/Add_Vector', reuse=reuse)
		# [batch size, memory state dim(d_v)]
		add_signal = tf.tanh(add_vector)

		# Add vector after erase
		# [batch size, 1, memory state dim(d_v)]
		erase_reshaped = np.reshape(erase_signal, [-1,1,self.memory_state_dim])
		# [batch size, memory size, 1]
		cw_reshaped = np.reshape(correlation_weight, [-1,self.memory_size,1])
		# w_t(i) * e_t
		erase_mul = np.multiply(erase_reshaped, cw_reshaped)
		# Elementwise multiply between [batch size, memory size, memory state dim(d_v)]
		erase = value_matrix * (1 - erase_mul)
		# [batch size, 1, memory state dim(d_v)]
		add_reshaped = np.reshape(add_signal, [-1, 1, self.memory_state_dim])
		add_mul = np.multiply(add_reshaped, cw_reshaped)
		
		new_memory = erase + add_mul
		# [batch size, memory size, memory value staet dim]
		print('Memory shape : %s' % (new_memory.shape))
		return new_memory


# This class construct key matrix and value matrix
class DKVMN():
	def __init__(self, memory_size, memory_key_state_dim, memory_value_state_dim, init_memory_key, init_memory_value, name='DKVMN'):
		self.name = name
		self.memory_size = memory_size
		self.memory_key_state_dim = memory_key_state_dim
		self.memory_value_state_dim = memory_value_state_dim
		
		self.key = DKVMN_Memory(self.memory_size, self.memory_key_state_dim, name=self.name+'/key_matrix')
		self.value = DKVMN_Memory(self.memory_size, self.memory_value_state_dim, name=self.name+'/value_matrix')

		self.memory_key = self.init_memory_key
		self.memory_value = self.init_memory_value

	def attention(self, q_embedded):
		correlation_weight = self.key.cor_weight(embedded=q_embedded, key_matrix=self.memory_key)
		return correlation_weight

	def read(self, c_weight):
		read_content = self.value.read(value_matrix=self.memory_value, correlation_weight=c_weight)
		return read_content

	def write(self, c_weight, qa_embedded):
		new_memory_value = self.value.write(value_matrix=self.memory_value, correlation_weight=c_weight, qa_embedded=qa_embedded)
		return new_memory_value



