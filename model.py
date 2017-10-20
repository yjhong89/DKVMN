import numpy as np
import os, time
import tensorflow as tf
import operations
import shutil
from memory import DKVMN
from sklearn import metrics


class Model():
	def __init__(self, args, sess, name='KT'):
		self.args = args
		self.name = name
		self.sess = sess

		self.create_model()

	def create_model(self):
		# 'seq_len' means question sequences
		self.q_data = tf.placeholder(tf.int32, [self.args.batch_size, self.args.seq_len], name='q_data') 
		self.qa_data = tf.placeholder(tf.int32, [self.args.batch_size, self.args.seq_len], name='qa_data')
		self.target = tf.placeholder(tf.float32, [self.args.batch_size, self.args.seq_len], name='target')

		# Initialize Memory
		with tf.variable_scope('Memory'):
			init_memory_key = tf.get_variable('key', [self.args.memory_size, self.args.memory_key_state_dim], \
				initializer=tf.truncated_normal_initializer(stddev=0.1))
			init_memory_value = tf.get_variable('value', [self.args.memory_size,self.args.memory_value_state_dim], \
				initializer=tf.truncated_normal_initializer(stddev=0.1))
		# Broadcast memory value tensor to match [batch size, memory size, memory state dim]
		# First expand dim at axis 0 so that makes 'batch size' axis and tile it along 'batch size' axis
		# tf.tile(inputs, multiples) : multiples length must be thes saame as the number of dimensions in input
		# tf.stack takes a list and convert each element to a tensor
		init_memory_value = tf.tile(tf.expand_dims(init_memory_value, 0), tf.stack([self.args.batch_size, 1, 1]))
		print(init_memory_value.get_shape())
				
		self.memory = DKVMN(self.args.memory_size, self.args.memory_key_state_dim, \
				self.args.memory_value_state_dim, init_memory_key=init_memory_key, init_memory_value=init_memory_value, name='DKVMN')

		# Embedding to [batch size, seq_len, memory_state_dim(d_k or d_v)]
		with tf.variable_scope('Embedding'):
			# A
			q_embed_mtx = tf.get_variable('q_embed', [self.args.n_questions+1, self.args.memory_key_state_dim],\
				initializer=tf.truncated_normal_initializer(stddev=0.1))
			# B
			qa_embed_mtx = tf.get_variable('qa_embed', [2*self.args.n_questions+1, self.args.memory_value_state_dim], initializer=tf.truncated_normal_initializer(stddev=0.1))		

		# Embedding to [batch size, seq_len, memory key state dim]
		q_embed_data = tf.nn.embedding_lookup(q_embed_mtx, self.q_data)
		# List of [batch size, 1, memory key state dim] with 'seq_len' elements
		#print('Q_embedding shape : %s' % q_embed_data.get_shape())
		slice_q_embed_data = tf.split(q_embed_data, self.args.seq_len, 1)
		#print(len(slice_q_embed_data), type(slice_q_embed_data), slice_q_embed_data[0].get_shape())
		# Embedding to [batch size, seq_len, memory value state dim]
		qa_embed_data = tf.nn.embedding_lookup(qa_embed_mtx, self.qa_data)
		#print('QA_embedding shape: %s' % qa_embed_data.get_shape())
		# List of [batch size, 1, memory value state dim] with 'seq_len' elements
		slice_qa_embed_data = tf.split(qa_embed_data, self.args.seq_len, 1)
		
		prediction = list()
		reuse_flag = False

		# Logics
		for i in xrange(self.args.seq_len):
			# To reuse linear vectors
			if i != 0:
				reuse_flag = True
			# k_t : [batch size, memory key state dim]
			q = tf.squeeze(slice_q_embed_data[i], 1)
			# Attention, [batch size, memory size]
			self.correlation_weight = self.memory.attention(q)
			
			# Read process, [batch size, memory value state dim]
			self.read_content = self.memory.read(self.correlation_weight)
			
			# Write process, [batch size, memory size, memory value state dim]
			# qa : [batch size, memory value state dim]
			qa = tf.squeeze(slice_qa_embed_data[i], 1)
			# Only last time step value is necessary
			self.new_memory_value = self.memory.write(self.correlation_weight, qa, reuse=reuse_flag)

			mastery_level_prior_difficulty = tf.concat([self.read_content, q], 1)
			# f_t
			summary_vector = tf.tanh(operations.linear(mastery_level_prior_difficulty, self.args.final_fc_dim, name='Summary_Vector', reuse=reuse_flag))
			# p_t
			pred_logits = operations.linear(summary_vector, 1, name='Prediction', reuse=reuse_flag)

			prediction.append(pred_logits)

		# 'prediction' : seq_len length list of [batch size ,1], make it [batch size, seq_len] tensor
		# tf.stack convert to [batch size, seq_len, 1]
		self.pred_logits = tf.reshape(tf.stack(prediction, axis=1), [self.args.batch_size, self.args.seq_len]) 

		# Define loss : standard cross entropy loss, need to ignore '-1' label example
		# Make target/label 1-d array
		target_1d = tf.reshape(self.target, [-1])
		pred_logits_1d = tf.reshape(self.pred_logits, [-1])
		index = tf.where(tf.not_equal(target_1d, tf.constant(-1., dtype=tf.float32)))
		# tf.gather(params, indices) : Gather slices from params according to indices
		filtered_target = tf.gather(target_1d, index)
		filtered_logits = tf.gather(pred_logits_1d, index)
		self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=filtered_logits, labels=filtered_target))
		self.pred = tf.sigmoid(self.pred_logits)

		# Optimizer : SGD + MOMENTUM with learning rate decay
		self.global_step = tf.Variable(0, trainable=False)
		self.lr = tf.placeholder(tf.float32, [], name='learning_rate')
#		self.lr_decay = tf.train.exponential_decay(self.args.initial_lr, global_step=global_step, decay_steps=10000, decay_rate=0.667, staircase=True)
#		self.learning_rate = tf.maximum(lr, self.args.lr_lowerbound)
		optimizer = tf.train.MomentumOptimizer(self.lr, self.args.momentum)
		grads, vrbs = zip(*optimizer.compute_gradients(self.loss))
		grad, _ = tf.clip_by_global_norm(grads, self.args.maxgradnorm)
		self.train_op = optimizer.apply_gradients(zip(grad, vrbs), global_step=self.global_step)
#		grad_clip = [(tf.clip_by_value(grad, -self.args.maxgradnorm, self.args.maxgradnorm), var) for grad, var in grads]
		self.tr_vrbs = tf.trainable_variables()
		for i in self.tr_vrbs:
			print(i.name)

		self.saver = tf.train.Saver()


	def train(self, train_q_data, train_qa_data, valid_q_data, valid_qa_data):
		# q_data, qa_data : [samples, seq_len]
		shuffle_index = np.random.permutation(train_q_data.shape[0])
		q_data_shuffled = train_q_data[shuffle_index]
		qa_data_shuffled = train_qa_data[shuffle_index]

		training_step = train_q_data.shape[0] // self.args.batch_size
		self.sess.run(tf.global_variables_initializer())
		
		if self.args.show:
			from utils import ProgressBar
			bar = ProgressBar(label, max=training_step)

		self.train_count = 0
		if self.args.init_from:
			if self.load():
				print('Checkpoint_loaded')
			else:
				print('No checkpoint')
		else:
			if os.path.exists(os.path.join(self.args.checkpoint_dir, self.model_dir)):
				try:
					shutil.rmtree(os.path.join(self.args.checkpoint_dir, self.model_dir))
					shutil.rmtree(os.path.join(self.args.log_dir, self.mode_dir+'.csv'))
				except(FileNotFoundError, IOError) as e:
					print('[Delete Error] %s - %s' % (e.filename, e.strerror))
		
		best_valid_auc = 0

		# Training
		for epoch in xrange(0, self.args.num_epochs):
			if self.args.show:
				bar.next()

			pred_list = list()
			target_list = list()		
			epoch_loss = 0
			learning_rate = tf.train.exponential_decay(self.args.initial_lr, global_step=self.global_step, decay_steps=self.args.anneal_interval*training_step, decay_rate=0.667, staircase=True)

			#print('Epoch %d starts with learning rate : %3.5f' % (epoch+1, self.sess.run(learning_rate)))
			for steps in xrange(training_step):
				# [batch size, seq_len]
				q_batch_seq = q_data_shuffled[steps*self.args.batch_size:(steps+1)*self.args.batch_size, :]
				qa_batch_seq = qa_data_shuffled[steps*self.args.batch_size:(steps+1)*self.args.batch_size, :]
	
				# qa : exercise index + answer(0 or 1)*exercies_number
				# right : 1, wrong : 0, padding : -1
				target = qa_batch_seq[:,:]
				# Make integer type to calculate target
				target = target.astype(np.int)
				target_batch = (target - 1) // self.args.n_questions  
				target_batch = target_batch.astype(np.float)

				feed_dict = {self.q_data:q_batch_seq, self.qa_data:qa_batch_seq, self.target:target_batch, self.lr:self.args.initial_lr}
				#self.lr:self.sess.run(learning_rate)
				loss_, pred_, _, = self.sess.run([self.loss, self.pred, self.train_op], feed_dict=feed_dict)
				# Get right answer index
				# Make [batch size * seq_len, 1]
				right_target = np.asarray(target_batch).reshape(-1,1)
				right_pred = np.asarray(pred_).reshape(-1,1)
				# np.flatnonzero returns indices which is nonzero, convert it list 
				right_index = np.flatnonzero(right_target != -1.).tolist()
				# Number of 'training_step' elements list with [batch size * seq_len, ]
				pred_list.append(right_pred[right_index])
				target_list.append(right_target[right_index])

				epoch_loss += loss_
				#print('Epoch %d/%d, steps %d/%d, loss : %3.5f' % (epoch+1, self.args.num_epochs, steps+1, training_step, loss_))
				

			if self.args.show:
				bar.finish()		
			
			all_pred = np.concatenate(pred_list, axis=0)
			all_target = np.concatenate(target_list, axis=0)

			# Compute metrics
			self.auc = metrics.roc_auc_score(all_target, all_pred)
			# Extract elements with boolean index
			# Make '1' for elements higher than 0.5
			# Make '0' for elements lower than 0.5
			all_pred[all_pred > 0.5] = 1
			all_pred[all_pred <= 0.5] = 0
			self.accuracy = metrics.accuracy_score(all_target, all_pred)

			epoch_loss = epoch_loss / training_step	
			print('Epoch %d/%d, loss : %3.5f, auc : %3.5f, accuracy : %3.5f' % (epoch+1, self.args.num_epochs, epoch_loss, self.auc, self.accuracy))
			self.write_log(epoch=epoch+1, auc=self.auc, accuracy=self.accuracy, loss=epoch_loss, name='training_')

			valid_steps = valid_q_data.shape[0] // self.args.batch_size
			valid_pred_list = list()
			valid_target_list = list()
			for s in range(valid_steps):
				# Validation
				valid_q = valid_q_data[s*self.args.batch_size:(s+1)*self.args.batch_size, :]
				valid_qa = valid_qa_data[s*self.args.batch_size:(s+1)*self.args.batch_size, :]
				# right : 1, wrong : 0, padding : -1
				valid_target = (valid_qa - 1) // self.args.n_questions
				valid_feed_dict = {self.q_data : valid_q, self.qa_data : valid_qa, self.target : valid_target}
				valid_loss, valid_pred = self.sess.run([self.loss, self.pred], feed_dict=valid_feed_dict)
				# Same with training set
				valid_right_target = np.asarray(valid_target).reshape(-1,)
				valid_right_pred = np.asarray(valid_pred).reshape(-1,)
				valid_right_index = np.flatnonzero(valid_right_target != -1).tolist()	
				valid_target_list.append(valid_right_target[valid_right_index])
				valid_pred_list.append(valid_right_pred[valid_right_index])
			
			all_valid_pred = np.concatenate(valid_pred_list, axis=0)
			all_valid_target = np.concatenate(valid_target_list, axis=0)

			valid_auc = metrics.roc_auc_score(all_valid_target, all_valid_pred)
		 	# For validation accuracy
			all_valid_pred[all_valid_pred > 0.5] = 1
			all_valid_pred[all_valid_pred <= 0.5] = 0
			valid_accuracy = metrics.accuracy_score(all_valid_target, all_valid_pred)
			print('Epoch %d/%d, valid auc : %3.5f, valid accuracy : %3.5f' %(epoch+1, self.args.num_epochs, valid_auc, valid_accuracy))
			# Valid log
			self.write_log(epoch=epoch+1, auc=valid_auc, accuracy=valid_accuracy, loss=valid_loss, name='valid_')
			if valid_auc > best_valid_auc:
				print('%3.4f to %3.4f' % (best_valid_auc, valid_auc))
				best_valid_auc = valid_auc
				best_epoch = epoch + 1
				self.save(best_epoch)

		return best_epoch	
			
	def test(self, test_q, test_qa):
		steps = test_q.shape[0] // self.args.batch_size
		self.sess.run(tf.global_variables_initializer())
		if self.load():
			print('CKPT Loaded')
		else:
			raise Exception('CKPT need')

		pred_list = list()
		target_list = list()

		for s in range(steps):
			test_q_batch = test_q[s*self.args.batch_size:(s+1)*self.args.batch_size, :]
			test_qa_batch = test_qa[s*self.args.batch_size:(s+1)*self.args.batch_size, :]
			target = test_qa_batch[:,:]
			target = target.astype(np.int)
			target_batch = (target - 1) // self.args.n_questions  
			target_batch = target_batch.astype(np.float)
			feed_dict = {self.q_data:test_q_batch, self.qa_data:test_qa_batch, self.target:target_batch}
			loss_, pred_ = self.sess.run([self.loss, self.pred], feed_dict=feed_dict)
			# Get right answer index
			# Make [batch size * seq_len, 1]
			right_target = np.asarray(target_batch).reshape(-1,1)
			right_pred = np.asarray(pred_).reshape(-1,1)
			# np.flatnonzero returns indices which is nonzero, convert it list 
			right_index = np.flatnonzero(right_target != -1.).tolist()
			# Number of 'training_step' elements list with [batch size * seq_len, ]
			pred_list.append(right_pred[right_index])
			target_list.append(right_target[right_index])

		all_pred = np.concatenate(pred_list, axis=0)
		all_target = np.concatenate(target_list, axis=0)

		# Compute metrics
		self.test_auc = metrics.roc_auc_score(all_target, all_pred)
		# Extract elements with boolean index
		# Make '1' for elements higher than 0.5
		# Make '0' for elements lower than 0.5
		all_pred[all_pred > 0.5] = 1
		all_pred[all_pred <= 0.5] = 0

		self.test_accuracy = metrics.accuracy_score(all_target, all_pred)

		print('Test auc : %3.4f, Test accuracy : %3.4f' % (self.test_auc, self.test_accuracy))


	@property
	def model_dir(self):
		return '{}_{}batch_{}epochs'.format(self.args.dataset, self.args.batch_size, self.args.num_epochs)

	def load(self):
		checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.model_dir)
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.train_count = int(ckpt_name.split('-')[-1])
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			return True
		else:
			return False

	def save(self, global_step):
		model_name = 'DKVMN'
		checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.model_dir)
		if not os.path.exists(checkpoint_dir):
			os.mkdir(checkpoint_dir)
		self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=global_step)
		print('Save checkpoint at %d' % (global_step+1))

	# Log file
	def write_log(self, auc, accuracy, loss, epoch, name='training_'):
		log_path = os.path.join(self.args.log_dir, name+self.model_dir+'.csv')
		if not os.path.exists(log_path):
			self.log_file = open(log_path, 'w')
			self.log_file.write('Epoch\tAuc\tAccuracy\tloss\n')
		else:
			self.log_file = open(log_path, 'a')	
		
		self.log_file.write(str(epoch) + '\t' + str(auc) + '\t' + str(accuracy) + '\t' + str(loss) + '\n')
		self.log_file.flush()	
		
