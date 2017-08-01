import numpy as np
import os, time
import tensorflow as tf
import operations
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
		self.target = tf.placeholder(tf.int32, [self.args.batch_size, self.args.seq_len], name='target')

		# Initialize Memory
		with tf.variable_scope('Memory'):
			init_memory_key = tf.get_variable('key', [self.args.memory_size, self.args.memory_key_state_dim], \
				initializer=tf.truncated_normal_initializer(stddev=0.1))
			init_memory_value = tf.get_variable('value', [self.args.batch_size, self.args.memory_size,\
				 self.args.memory_value_state_dim], initializer=tf.truncated_normal_initializer(stddev=0.1))
		
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
		# tf.split(axis, num_splits, value) in tensorflow 0.12v
		print('Q_embedding shape : %s' % q_embed_data.get_shape())
		slice_q_embed_data = tf.split(1, self.args.seq_len, q_embed_data)
		print(len(slice_q_embed_data), type(slice_q_embed_data), slice_q_embed_data[0].get_shape())
		# Embedding to [batch size, seq_len, memory value state dim]
		qa_embed_data = tf.nn.embedding_lookup(qa_embed_mtx, self.qa_data)
		print('QA_embedding shape: %s' % qa_embed_data.get_shape())
		# List of [batch size, 1, memory value state dim] with 'seq_len' elements
		slice_qa_embed_data = tf.split(1, self.args.seq_len, qa_embed_data)
		
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
			# tf.concat(axis, valuelist) in tensorflow v0.12
			mastery_level_prior_difficulty = tf.concat(1, [self.read_content, q])
			# f_t
			summary_vector = tf.tanh(operations.linear(mastery_level_prior_difficulty, self.args.final_fc_dim, name='Summary_Vector', reuse=reuse_flag))
			# p_t
			pred = tf.sigmoid(operations.linear(summary_vector, 1, name='Prediction', reuse=reuse_flag))
			print('Prediction shape : %s' % pred.get_shape())

			prediction.append(pred)

		# 'prediction' : seq_len length list of [batch size ,1], make it [batch size, seq_len] tensor
		# tf.stack convert to [batch size, seq_len, 1]
		self.pred = tf.reshape(tf.stack(prediction, axis=1), [self.args.batch_size, self.args.seq_len]) 
		print(self.pred.get_shape())
		print(self.target.get_shape())
		# Define loss : standard cross entropy loss
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.target, logits=self.pred))

		# Optimizer : SGD + MOMENTUM with learning rate decay
		global_step = tf.Variable(0, trainable=False)
		lr = tf.train.exponential_decay(self.args.initial_lr, global_step=global_step, decay_steps=2000, decay_rate=0.667, staircase=True)
		learning_rate = tf.maximum(lr, self.args.final_lr)
		optimizer = tf.train.MomentumOptimizer(learning_rate, self.args.momentum)
		grads = optimizer.compute_gradients(self.loss)
		for grad, var in grads:
			# In case of None gradient
			if grad is not None:
				clipping = tf.clip_by_value(grad, -self.args.maxgradnorm, self.args.maxgradnorm)
		with tf.control_dependencies([clipping]):
			self.train_op = optimizer.apply_gradients(grads, global_step=global_step)
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
		if self.load():
			print('Checkpoint_loaded')
		else:
			print('No checkpoint')

		# Training
		for epoch in xrange(self.train_count, self.args.num_epochs):
			if self.args.show:
				bar.next()

			pred_list = list()
			target_list = list()		
			epoch_loss = 0
			best_valid_auc = 0

			print('Epoch %d starts' % (epoch+1))
			for steps in xrange(training_step):
				# [batch size, seq_len]
				q_batch_seq = q_data_shuffled[steps*self.args.batch_size:(steps+1)*self.args.batch_size, :]
				qa_batch_seq = qa_data_shuffled[steps*self.args.batch_size:(steps+1)*self.args.batch_size, :]
	
				# qa : exercise index + answer(0 or 1)*exercies_number
				# right : 1, wrong : 0, padding : -1
				target_batch = (qa_batch_seq - 1) // self.args.n_questions  

				feed_dict = {self.q_data:q_batch_seq, self.qa_data:qa_batch_seq, self.target:target_batch}
				loss_, pred_, _, = self.sess.run([self.loss, self.pred, self.train_op], feed_dict=feed_dict)

				# Get right answer index
				# Make [batch size * seq_len, 1]
				right_target = np.asarray(target_batch).reshape(-1,)
				right_pred = np.asarray(pred_).reshape(-1,)
				# np.flatnonzero returns indices which is nonzero, convert it list 
				right_index = np.flatnonzero(right_target != -1).tolist()
				
				# 'training_step' elements list with [batch size * seq_len, ]
				pred_list.append(right_pred[right_index])
				target_list.append(right_target[right_index])

				epoch_loss += loss_
				print('Epoch %d/%d, steps %d/%d, loss : %3.5f' % (epoch+1, self.args.num_epochs, steps+1, training_step, loss_))

			if self.args.show:
				bar.finish()		
			
			all_pred = np.concatenate(pred_list)
			all_target = np.concatenate(target_list)

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
			self.write_log(epoch+1, self.auc, self.accuracy, epoch_loss, name='training_')

			# Validation
			valid_q = valid_q_data[:self.args.validation_index, :]
			valid_qa = valid_qa_data[:self.args.validation_index, :]
			# right : 1, wrong : 0, padding : -1
			valid_target = (valid_qa - 1) // self.args.n_questions
			valid_feed_dict = {self.q_data : valid_q, self.qa_data : valid_qa, self.target : valid_target}
			valid_loss, valid_pred = self.sess.run([self.loss, self.pred], feed_dict=valid_feed_dict)
			# Same with training set
			valid_right_target = valid_target.asnumpy().reshape(-1,)
			valid_right_pred = valid_pred.asnumpy().reshape(-1,)
			valid_right_index = np.flatnonzero(valid_right_target != -1).tolist()
			valid_auc = metrics.roc_auc_score(valid_right_target[valid_right_index], valid_right_pred[valid_right_index])
		 	# For validation accuracy
			valid_right_pred[valid_right_index][valid_right_pred[valid_right_index] > 0.5] = 1
			valid_right_pred[valid_right_index][valid_right_pred[valid_right_index] <= 0.5] = 0
			valid_accuracy = metrics.accuracy_score(valid_right_target[valid_right_index], valid_right_pred[valid_right_index])
			# Valid log
			self.write_log(epoch+1, valid_auc, valid_accuracy, valid_loss, name='valid_')
			if valid_auc > best_valid_auc:
				best_valid_auc = valid_auc
				best_epoch = epoch + 1

			if np.mod(epoch+1, self.args.save_interval) == 0:
				self.save(epoch)
		
		return best_epoch	
			



	@property
	def model_dir(self):
		return '{}_{}batch_{}epochs'.format(self.args.dataset, self.args.batch_size, self.args.num_epochs)

	def load(self):
		checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.model_dir)
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_chekcpoint_path)
			self.train_count = int(ckpt_name.split('-')[-1])
			self.saver.restore(self.sess, os.path.join(chekcpoint_dir, ckpt_name))
			return True
		else:
			return False

	def save(self, global_step):
		model_name = 'DKVMN'
		checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.model_dir)
		if not os.path.exists(checkpoint_dir):
			os.mkdir(checkpoint_dir)
		self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=global_step)
		print('Save checkpoint at %d' % global_step)

	# Log file
	def write_log(self, auc, accuracy, loss, epoch, name='training_'):
		try:
			self.log_file = open(os.path.join(self.args.log_dir, name+self.model_dir), 'a')
		except:
			self.log_file = open(os.path.join(self.args.log_dir, self.args.filename), 'w')
			self.log_file.write('Epoch\tAuc\tAccurac\tloss\n')
		
		self.log_file.write(str(epoch) + '\t' + str(auc) + '\t' + str(accuracy) + '\t' + str(loss) + '\n')
		self.log_file.flush()	
		











	
		




