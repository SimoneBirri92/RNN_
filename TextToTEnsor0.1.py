#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np 
import tensorflow as tf
import math as ma 
import random as rand 
from io import open
from conllu import parse_incr, parse
from gensim.models import Word2Vec
from collections import OrderedDict

epochs = 500
embedding_size = 300
encoder_hidden_dim = 512
decoder_hidden_dim = encoder_hidden_dim * 2
max_gradient_norm = 5
learning_rate = 0.0001 
batch_size = 256
max_gradient_norm = 5.0
UNK = 'UNK'
PAD = 'PAD'
GO = 'GO'
EOS = 'EOS'

tf.reset_default_graph()
# Carica il pretraining Glove per la lingua italiana
model = Word2Vec.load('/home/simone/RNN_/glove_WIKI')

def load_glove_embedding(model):
    embeddings = {}
    for word in model.wv.vocab:
        embeddings[word] = model.wv[word]
    return (embeddings)

pretrain_vectors = load_glove_embedding(model)

#print(pretrain_vectors['corriere'])

data = open('/home/simone/RNN_/it_isdt-ud-dev.conllu', 'r', encoding = 'utf-8')
data_matform = []   
for token in parse_incr(data):
    data_matform.append(token)

tok_dict = data_matform[0][0]

#Lista che contiene per ogni frase le teste delle parole che la compongono
head_list = []
#Lista che contiene per ogni riga una frase di input 
sentence_list = []
#Matrice che contiene le features definite nel Conll quali head positions, tipo, genere, numero ecc
mat_token = []
#Parole uniche in tutto il data set (servirà per l'embedding delle parole)
words_in_dataset = []
words_in_dataset.append(PAD) #0
words_in_dataset.append(GO)  #1
words_in_dataset.append(EOS) #2
words_in_dataset.append(UNK) #3
#Lista che contiene tutte le possibili posizioni relative del dataset (servirà per l'embedding delle posizioni relative)
all_relative_positions = []
#numero di features per ogni parola
n_features = 10
max_length = 0  
#estrai_feauter estra il vocabolario e la posizione relativa di ogni parola del vocabolario e li inserisce in 2 vettori
#(vocabulary, head_positions)
def estrai_features_training(data,max_length,words_list):
    #Variabile che tiene traccia dell'indice della frase che stiamo processando ci serve
    #per ricavare le informazioni per gestire il problema delle proprosizioni composte
    current_sentence_index = 0
    for tokenlist in data:
        sentence = []
        heads = []
        #Comtatore che tiene traccia del numero di parole composte in una frase.
        #Dal momento che l'indicizzazione sfasa a causa del formato del corpus del Conll
        composed_words_counter = 0
        #print(len(tokenlist))
        # number_of_sentences += 1
        if (len(tokenlist) > max_length):
            max_length = len(tokenlist)
        for words in tokenlist:
            #print('Forma ',words['form'])
            #print('Testa ',words['head'])
            sentence.append(words['form'].lower())
            if (words['form'].lower() not in words_list):
                words_list.append(words['form'].lower())
            #['3-4', 'coi', '_', '_', '_', '_', '_', '_', '_', '_'] problema con le preposizioni articolate, possiamo cercare di 
            # ricavare le informazioni dai token successivi. Per adesso se abbiamo una preposizione articolata non eseguiamo il 
            # calcolo della posizione relativa
            if (words['head'] != None):
                #print(words['head'])
                position = words['head'] - words['id']
                if (position < 0):
                    relative_position = ("L%d" %ma.fabs(position))
                else:
                    relative_position = ("R%d" %ma.fabs(position))
                heads.append(relative_position)
                if(relative_position not in all_relative_positions):
                    all_relative_positions.append(relative_position)
            else:
                word_composing_index = words['id']
                #print (word_composing_index)
                #print(word_composing_index[0])
                #print('Numero di parole composte ',composed_words_counter)
                word_composing = data[current_sentence_index][word_composing_index[0]+composed_words_counter]
                #print (word_composing)
                position = np.int64(word_composing['head']) - np.int64(word_composing['id']) 
                if (position < 0):
                    relative_position = ("L%d" %ma.fabs(position))
                else:
                    relative_position = ("R%d" %ma.fabs(position))
                heads.append(relative_position)
                if(relative_position not in all_relative_positions):
                    all_relative_positions.append(relative_position)
                #print(relative_position)
                composed_words_counter +=1
        current_sentence_index += 1
        sentence_list.append(sentence)   
        head_list.append(heads)
    return(mat_token,max_length,all_relative_positions, words_list,sentence_list,head_list)

mat_token, max_length,all_relative_positions, vocabulary ,list_of_sentence,list_of_heads = estrai_features_training(data_matform,max_length,words_in_dataset)

heads_words_vocabulary = np.concatenate((vocabulary,all_relative_positions), axis = 0)#print(vocabulary)

#print(all_relative_positions)
#print(mat_token)
#print(words_in_dataset)
#print(len(list_of_sentence))
#print(len(list_of_heads))
#print(all_relative_positions)

#Training set 
#print(len(data_matform))
def Define_training_test_set(data_sample,data_labels,training_percentage,data_size):
    training_length = round((data_size*training_percentage)/100)
    #print (training_length)
    training_set_sample = data_sample[:training_length]
    test_set_sample = data_sample[training_length:]
    training_set_labels = data_labels[:training_length]
    test_set_labels = data_labels[training_length:]
    return training_set_sample, test_set_sample, training_set_labels, test_set_labels

training_set_sample, test_set_sample, training_set_labels, test_set_labels = Define_training_test_set(sentence_list,head_list,60,len(sentence_list))

#print(training_set_sample[:5])
#print(training_set_labels[:5])

vocab_size = len(vocabulary)
heads_number = len(all_relative_positions)
#print(vocab_size)
#print(heads_number)
#word_to_index contiene un valore unico per ogni chiave, dove la chiave è un elemento della lista heads_words_vocabulary che ricordiamo 
#contiene simboli speciali (PAD,UNK,EOS,GO), parole e posizione relative estrapolate dal dataset

word_to_index = dict([(w,i) for i,w in enumerate(heads_words_vocabulary)])


def embedding_matrix_build(vocabulary,pretrained_vec,vocab_size, embedding_size):
    embedding_matrix = np.random.uniform(-1, 1, size=(vocab_size, embedding_size))
    i = 0
    for w in vocabulary:
        try:
            v = pretrained_vec.get(w)
            if v is not None and i < vocab_size:
                embedding_matrix[i] = v
            i += 1
        except:
            v = np.array(np.random.uniform(-1.0, 1.0, embedding_size))
            if v is not None and i < vocab_size:
                embedding_matrix[i] = v
            i += 1
    return(embedding_matrix)

def one_hot_encode_idx(index_array):
    #Matrice che conterrà l'embedding per ogni posizione esistente.
    pos_embedding_matrix = np.random.uniform(-1, 1, size=(len(index_array), embedding_size))
    i=0
    for idx in index_array:
        one_hot = np.zeros(embedding_size)
        #print(index_array.index(idx))
        one_hot[index_array.index(idx)] += 1
        pos_embedding_matrix[i] = one_hot
        i += 1
    return pos_embedding_matrix

pos_embedding_mat = one_hot_encode_idx(all_relative_positions)

#print(pos_embedding_mat.shape)

words_embedding_mat = embedding_matrix_build(vocabulary,pretrain_vectors,vocab_size,embedding_size)

words_heads_embedding_mat = np.concatenate((words_embedding_mat, pos_embedding_mat),axis=0)

#print(pos_embedding_mat[6])
#print(all_relative_positions[6])
#print(pretrain_vectors['corriere'])


# words_id conterrà la frase di input da processare.
# shape = (batch size, max length of sentence in batch)
words_id = tf.placeholder(tf.int32, shape = [None, batch_size])
#paddings_4_words = [[0, 0,], [0, max_length - tf.shape(words_id)[0]]]
#words_id = tf.pad(words_id, paddings_4_words, mode = 'CONSTANT', name = 'pad_words_id', constant_values = PAD)
#pos_id conterrà la codifica delle posizioni relative
pos_id = tf.placeholder(tf.int32, shape = [None, batch_size])
#paddings_4_labels_sos = tf.constant([[0, 0,], [1, 0]])
#pos_id = tf.pad(pos_id, paddings_4_labels_sos, mode = 'CONSTANT', name = 'pad_pos_id', constant_values = GO )
#paddings_4_labels_eos = tf.constant([[0, 0,], [0, 1]])
#pos_id = tf.pad(pos_id, paddings_4_labels_eos, mode = 'CONSTANT', name = 'pad_pos_id', constant_values = EOS ) 
#paddings_4_labels = [[0, 0,], [0,max_length - tf.shape(pos_id)[0]]]
#pos_id = tf.pad(pos_id, paddings_4_labels, mode = 'CONSTANT', name = 'pad_pos_id', constant_values = PAD ) 

# to contain the sentence length for each sentence in the batch
enc_train_inp_lengths = tf.placeholder(tf.int32, shape=[batch_size],name='train_input_lengths')
#supponiamo che la nostra frase in input è la seguente 
dec_train_imp_lengths = tf.placeholder(tf.int32, shape=[batch_size]) 
#input_str = 'Corriere Sport da pagina 23 a pagina 26'

#word_to_idx = OrderedDict({w:vocabulary.index(w) for w in input_str.lower().split() if w in vocabulary})

W_id = tf.Variable(words_heads_embedding_mat, dtype = tf.float32, trainable = True)
P_id = tf.Variable(words_embedding_mat, dtype = tf.float32, trainable = True)
pos_embedding = tf.nn.embedding_lookup(P_id, pos_id)
pretreined_embedding = tf.nn.embedding_lookup(W_id, words_id)

#print(pretreined_embedding)

#encoder
encoder_cell = tf.nn.rnn_cell.LSTMCell(encoder_hidden_dim)
((encoder_fw_outputs,
  encoder_bw_outputs),
  (encoder_fw_final_state,
  encoder_bw_final_state)) = tf.nn.bidirectional_dynamic_rnn( cell_fw = encoder_cell,
                                                              cell_bw = encoder_cell,
                                                              inputs = pretreined_embedding,
                                                              sequence_length = enc_train_inp_lengths,
                                                              dtype = tf.float32, time_major=True) 


encoder_output = tf.concat([encoder_fw_outputs, encoder_bw_outputs], 2)

encoder_final_state_c = tf.concat([encoder_bw_final_state.c, encoder_fw_final_state.c], 1)

encoder_final_state_h = tf.concat([encoder_bw_final_state.h, encoder_fw_final_state.h], 1)

encoder_final_State = tf.nn.rnn_cell.LSTMStateTuple(encoder_final_state_c, encoder_final_state_h)

#print('pos embedding',pos_embedding)
#print('dec train input length', dec_train_imp_lengths)
#print('encoder final state', encoder_final_State)

#building RNN Cell 

decoder_cell = tf.nn.rnn_cell.LSTMCell(decoder_hidden_dim)

#Building helper
helper = tf.contrib.seq2seq.TrainingHelper(
    pos_embedding, dec_train_imp_lengths, time_major = True)
#print(helper)
#Decoder
projection_layer = tf.layers.Dense(vocab_size, use_bias = False)

decoder = tf.contrib.seq2seq.BasicDecoder(
    decoder_cell, helper, encoder_final_State , output_layer = projection_layer)

outputs, _, _= tf.contrib.seq2seq.dynamic_decode(decoder)
logits = outputs.rnn_output

#print(word_to_index.get('GO'))

#print(logits)

target_labels = tf.placeholder(tf.int32, shape = (batch_size, None))
'''
paddings_4_target_eos = tf.constant([[0, 0,], [0, 1]])
target_labels = tf.pad(target_labels, paddings_4_target_eos, mode = 'CONSTANT', name = 'pad_pos_id', constant_values = EOS ) 
paddings_4_target = [[0, 0,], [0, max_length - tf.shape(target_labels)[0]]]
target_labels = tf.pad(target_labels, paddings_4_target, mode = 'CONSTANT', name = 'pad_pos_id', constant_values = PAD ) 
'''
crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=target_labels, logits=logits)

global_step = tf.Variable(0,name = 'global_step', trainable= False)

params = tf.trainable_variables()
gradients = tf.gradients(crossent, params)
clipped_gradients, _ = tf.clip_by_global_norm(
    gradients, max_gradient_norm)

optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.apply_gradients(
    zip(clipped_gradients, params), global_step=global_step)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
#sess.run(print(tf.shape(words_id)))

#print(sentence_list)
#print(ma.ceil(len(sentence_list)/batch_size))
#Dividiamo la lista delle frasi in batch size 
start = 0
#max_length + 2 perchè con i token GO and END necessari per la traduzione la dimensione
#massima aumenta di 2
train_encoder_inputs = np.empty((max_length+2,len(list_of_sentence)))
train_decoder_inputs = np.empty((max_length+2,len(list_of_sentence)))
train_target_labels = np.empty((len(list_of_sentence),max_length+2))
#print(head_list)
print (max_length)
for i in range(len(list_of_sentence)):
    sequence_indexes = []
    labels_indexes = []
    target_indixes = []
    for labels in head_list[i]:
        index_label = word_to_index.get(labels)
        labels_indexes.append(index_label)
        target_indixes.append(index_label)
    for words in sentence_list[i]:
        index_word = word_to_index.get(words)
        sequence_indexes.append(index_word)
    pad_sentence_labels = np.pad(labels_indexes,(1,1), 'constant', constant_values = (word_to_index.get('GO'),word_to_index.get('EOS')))
    #max_length + 2 perchè con i token GO and END necessari per la traduzione la dimensione
    #massima aumenta di 2
    pad_sentence_labels = np.pad(pad_sentence_labels,(0,(max_length + 2) - len(pad_sentence_labels)), 'constant', constant_values = (word_to_index.get('PAD')))
    pad_targets = np.pad(target_indixes,(0,1),'constant', constant_values = word_to_index.get('EOS'))
    pad_targets = np.pad(pad_targets,(0,(max_length +2) - len(pad_targets)), 'constant', constant_values = word_to_index.get('PAD'))
    pad_sentence = np.pad(sequence_indexes,(0,(max_length +2) - len(sequence_indexes)),'constant', constant_values = word_to_index.get('PAD'))
    train_encoder_inputs[:,i] = pad_sentence
    train_decoder_inputs[:,i] = pad_sentence_labels
    train_target_labels[i,:] = pad_targets

#print(sentence_list[0])
#print(head_list[0])
#print(train_encoder_inputs[:,0])
#print(np.int64(train_decoder_inputs[:,0]))
#print(np.int64(train_target_labels[0]))
#print (len(sentence_list))
#print(train_encoder_inputs.shape)
# Train 
for i in range(100):
    for steps in range(1,ma.ceil(len(sentence_list)/batch_size)+1): 
        end = steps*batch_size
        #print('START',start)
        #print('END',end)

        try:
            feed_dict = {
            words_id: train_encoder_inputs[:,start:end],
            target_labels: train_target_labels[start:end,:],
            pos_id: train_decoder_inputs[:,start:end],
            dec_train_imp_lengths: np.ones((batch_size), dtype=int) * decoder_hidden_dim
        }
            #slice_sentence = sentence_list[start:end]
        except:
            feed_dict = {
                words_id: train_encoder_inputs[:,start:],
                target_labels: train_target_labels[:,start:],
                pos_id: train_decoder_inputs[:,start:],
                dec_train_imp_lengths: np.ones((batch_size), dtype=int) * decoder_hidden_dim
            }
            #slice_sentence = sentence_list[start:]
            #print(len(slice_sentence))
        
        start = end
        #print(feed_dict.get(words_id))
        #print(feed_dict.get(target_labels))
        #print(np.int64(feed_dict.get(pos_id)))
    _, loss_value = sess.run([train_op, crossent], feed_dict=feed_dict)
        
    
    #print(len(slice_sentence))

# Inference
inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
    P_id,
    tf.fill([batch_size], word_to_index.get('GO')), word_to_index.get('EOS'))

# Inference Decoder
inference_decoder = tf.contrib.seq2seq.BasicDecoder(
    decoder_cell, inference_helper, encoder_final_State,
    output_layer=projection_layer)


# Dynamic decoding
outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
    inference_decoder, maximum_iterations=epochs)
predicted_heads = outputs.sample_id

# Inference input
inference_sentence = 'la mia casa è molto grande'
inference_sentence = inference_sentence.split('')
inference_encoder_inputs = np.empty((max_length, 1))
inference_words_list = []
for words in inference_sentence:
    index_words = word_to_index.get(words)
    inference_words_list.append(index_words)
inference_encoder_inputs [:,0] = inference_words_list

feed_dict = {
    words_id: inference_encoder_inputs,
}

replies = sess.run([predicted_heads], feed_dict=feed_dict)
print(replies)




'''
# eseguiamo il pad del training_sample
for steps in range(1,(len(sentence_list)/batch_size)): 
    feed_dict = {
        words_id: sentence_list[steps,(steps*batch_size)],
        target_labels: head_list[steps,(steps*batch_size)],
        pos_id: head_list[steps,(steps*batch_size)],
        dec_train_imp_lengths: np.ones((batch_size), dtype=int) * decoder_hidden_dim
    }

    # Train 
    for i in range(100):
        _, loss_value = sess.run([train_op, crossent], feed_dict=feed_dict)
'''