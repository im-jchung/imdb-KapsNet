# Hardcoded, used only for experimentation

import os
import datetime

def fileWrite(data, loss_func):
	epochs, words, length, n_class, embed, conv1, caps1, caps2, vloss, vacc = data

	if not os.path.exists('./result/archs/'):
		os.mkdir('./result/archs/')

	now = datetime.datetime.now()
	now = str(now)[0:-7]

	f = open('./result/archs/archs.txt', 'a+')
	start = '=' * 50 + ' ' + now + ' ' + '=' * 50
	f.write(start + '\n')

	Epochs    = '{0: <20}'.format(str.format('Epochs: {}', epochs))
	Words     = '{0: <20}'.format(str.format('Words: {}', words))
	Length    = '{0: <20}'.format(str.format('Length: {}', length))
	n_class   = '{0: <20}'.format(str.format('n_class: {}', n_class))
	loss_func = '{0: <20}'.format(str.format('Loss_func: {}', loss_func))

	embed_out = '{0: <20}'.format(str.format('embed_out: {}', embed[0]))
	trainable = '{0: <20}'.format(str.format('trainable: {}', embed[1]))

	conv1_filters = '{0: <20}'.format(str.format('filters: {}', conv1[0]))
	conv1_kernel  = '{0: <20}'.format(str.format('kernel: {}', conv1[1]))
	conv1_stride  = '{0: <20}'.format(str.format('stride: {}', conv1[2]))
	conv1_pad     = '{0: <20}'.format(str.format('padding: {}', conv1[3]))
	conv1_activ   = '{0: <20}'.format(str.format('activ: {}', conv1[4]))

	caps1_dim    = '{0: <20}'.format(str.format('dim_cap: {}', caps1[0]))
	caps1_n_ch   = '{0: <20}'.format(str.format('n_channels: {}', caps1[1]))
	caps1_kernel = '{0: <20}'.format(str.format('kernel: {}', caps1[2]))
	caps1_stride = '{0: <20}'.format(str.format('stride: {}', caps1[3]))
	caps1_pad    = '{0: <20}'.format(str.format('padding: {}', caps1[4]))

	caps2_num = '{0: <20}'.format(str.format('num_caps: {}', caps2[0]))
	caps2_dim = '{0: <20}'.format(str.format('dim_caps: {}', caps2[1]))

	val_loss = '{0: <20}'.format(str.format('val_loss: {:.3f}', vloss))
	val_acc  = '{0: <20}'.format(str.format('val_acc: {:.3f}', vacc))

	gen   = 'Gen.  | ' + Epochs + Words + Length + n_class + loss_func + '\n'
	embed = 'Embed | ' + embed_out + trainable + '\n'
	conv1 = 'Conv1 | ' + conv1_filters + conv1_kernel + conv1_stride + conv1_pad + conv1_activ + '\n'
	caps1 = 'Caps1 | ' + caps1_dim + caps1_n_ch + caps1_kernel + caps1_stride + caps1_pad + '\n'
	caps2 = 'Caps2 | ' + caps2_num + caps2_dim + '\n'
	out   = 'Out   | ' + val_loss + val_acc + '\n'

	content = gen + embed + conv1 + caps1 + caps2 + out

	f.write(content)

	end = '=' * 58 + ' end ' + '=' * 58 + '\n'
	f.write(end + '\n')

	f.close()
