import tensorflow as tf
import numpy as np
from model import Model
import os, time, argparse


def main():
	parser.argparse.ArgumentParser()
	parser.add_argument('--num_epochs', type=int, default=50)
	parser.add_argument('--train', type=bool, default=True)
	parser.add_argument('--show', type=bool, default=False)
	parser.add_argument('--checkpoint_dir', type=str, defualt='checkpoint')
	parser.add_argument('--log_dir', type=str, default='logs')
	
	# synthetic / assist2009 / assist2015 / STATIC
	dataset = 'assist2009'

	if dataset == 'assist2009':
		parser.add_argument('--batch_size', type=int, default=32)
		parser.add_argument('--memory_size', type=int, default=20)
		parser.add_argument('--memory_key_state_dim', type=int, defaulti=50)
		parser.add_argument('--memory_value_state_dim', type=int, default=200)
		parser.add_argument('--initial_lr', type=float, default=0.05)
		parser.add_argument('--final_lr', type=float, default=1e-5)
		parser.add_argument('--momentum', type=float, default=0.9)
		parser.add_argument('--maxgradnorm', type=float, default=10)
		parser.add_argument('--final_fc_dim', type=int, default=50)
		parser.add_argument('--n_questions', type=int, default=110)
		parser.add_argument('--seq_len', type=int, default=200)

	elif dataset == 'synthetic':
		parser.add_argument('--batch_size', type=int, default=32)
		parser.add_argument('--memory_size', type=int, default=20)
		parser.add_argument('--memory_key_state_dim', type=int, defaulti=50)
		parser.add_argument('--memory_value_state_dim', type=int, default=200)
		parser.add_argument('--initial_lr', type=float, default=0.05)
		parser.add_argument('--final_lr', type=float, default=1e-5)
		parser.add_argument('--momentum', type=float, default=0.9)
		parser.add_argument('--maxgradnorm', type=float, default=10)
		parser.add_argument('--final_fc_dim', type=int, default=50)
		parser.add_argument('--n_questions', type=int, default=110)
		parser.add_argument('--seq_len', type=int, default=200)

	elif dataset == 'assist2015':
		parser.add_argument('--batch_size', type=int, default=32)
		parser.add_argument('--memory_size', type=int, default=20)
		parser.add_argument('--memory_key_state_dim', type=int, defaulti=50)
		parser.add_argument('--memory_value_state_dim', type=int, default=200)
		parser.add_argument('--initial_lr', type=float, default=0.05)
		parser.add_argument('--final_lr', type=float, default=1e-5)
		parser.add_argument('--momentum', type=float, default=0.9)
		parser.add_argument('--maxgradnorm', type=float, default=10)
		parser.add_argument('--final_fc_dim', type=int, default=50)
		parser.add_argument('--n_questions', type=int, default=110)
		parser.add_argument('--seq_len', type=int, default=200)

	elif dataset == 'STATIC':
		parser.add_argument('--batch_size', type=int, default=32)
		parser.add_argument('--memory_size', type=int, default=20)
		parser.add_argument('--memory_key_state_dim', type=int, defaulti=50)
		parser.add_argument('--memory_value_state_dim', type=int, default=200)
		parser.add_argument('--initial_lr', type=float, default=0.05)
		parser.add_argument('--final_lr', type=float, default=1e-5)
		parser.add_argument('--momentum', type=float, default=0.9)
		parser.add_argument('--maxgradnorm', type=float, default=10)
		parser.add_argument('--final_fc_dim', type=int, default=50)
		parser.add_argument('--n_questions', type=int, default=110)
		parser.add_argument('--seq_len', type=int, default=200)

	args = parser.parse_args()
	args.dataset = dataset

	run_config = tf.ConfigProto()
	run_config.gpu_options.allow_growth = True

















def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'n', 'f', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Not expected boolean type')

if __name__ == "__main__":
	main()



