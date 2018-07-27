# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.util import nest
import copynet


class Seq2Seq(object):
    def __init__(self,rnn_size,num_layer,emb_size,lr,mode,beam_search,beam_size,
                 vocab_size,start=2,stop=2,max_iterations=40,max_gradient_norm=5.0):
        self.rnn_size = rnn_size
        self.num_layer = num_layer
        self.emb_size = emb_size
        self.lr = lr
        self.mode = mode
        self.beam_search = beam_search
        self.beam_size = beam_size
        self.vocab_size = vocab_size
        self.start = start
        self.stop = stop
        self.max_iterations = max_iterations
        self.max_gradient_norm = max_gradient_norm
        
        self.model()
        
        
    def create_rnnCell(self):
        def single_rnn_cell():
            # 创建单个cell，这里需要注意的是一定要使用一个single_rnn_cell的函数，不然直接把cell放在MultiRNNCell
            # 的列表中最终模型会发生错误
            single_cell = tf.contrib.rnn.LSTMCell(self.rnn_size)
            #添加dropout
            cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=self.keep_prob_placeholder)
            return cell
        #列表中每个元素都是调用single_rnn_cell函数
        cell = tf.contrib.rnn.MultiRNNCell([single_rnn_cell() for _ in range(self.num_layer)])
        return cell
    
    def model(self):
        print("building model....")
        print("learning rate: \t"+str(self.lr))
        
        '''==================================1. placeholder'''
        self.encoder_inputs = tf.placeholder(tf.int32,[None,None],name="encoder_inputs")
        self.encoder_length = tf.placeholder(tf.int32,[None],name="encoder_length")
        self.decoder_inputs = tf.placeholder(tf.int32,[None,None],name="decoder_inputs")
        self.decoder_length = tf.placeholder(tf.int32,[None],name="decoder_length")
#        self.decoder_outputs =tf.placeholder(tf.int32,[None,None],name="decoder_outputs")
        
        self.batch_size = tf.placeholder(tf.int32,[],name="batch_size")
        self.keep_prob_placeholder = tf.placeholder(tf.float32,name="keep_prob_placeholder")
        
        self.max_target_sequence_length = tf.reduce_max(self.decoder_length,name="max_target_sequence_length")
        self.mask = tf.sequence_mask(self.decoder_length,self.max_target_sequence_length,dtype=tf.float32,name="mask")
        '''==================================2. encoder'''

        with tf.variable_scope("encoder"):
            
            encoder_cell = self.create_rnnCell()
            
            embedding = tf.get_variable("embedding",[self.vocab_size,self.emb_size])
            encoder_inputs_embeded = tf.nn.embedding_lookup(embedding,self.encoder_inputs)
            
            encoder_outputs,encoder_state = tf.nn.dynamic_rnn(encoder_cell,encoder_inputs_embeded,
                                                              sequence_length=self.encoder_length,
                                                              dtype=tf.float32)
            print("========encoder_state======")
            print(encoder_state)
            
        '''=================================3. decoder'''
        with tf.variable_scope("decoder"):
            encoder_inputs_length = self.encoder_length
            if self.beam_search:
                print("beamsearch decoding...")
                #tile_batch
                encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs,multiplier=self.beam_size)
                encoder_state = nest.map_structure(lambda s: tf.contrib.seq2seq.tile_batch(s,self.beam_size),encoder_state)
                encoder_inputs_length = tf.contrib.seq2seq.tile_batch(self.encoder_length,multiplier=self.beam_size)
            
            #attention mechanism
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=self.rnn_size,memory=encoder_outputs,
                                                                       memory_sequence_length=encoder_inputs_length)
            
            decoder_cell = self.create_rnnCell()
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cell,attention_mechanism=attention_mechanism,
                                                               attention_layer_size=self.rnn_size,name="Attention_Wrapper")
            batch_size = self.batch_size if not self.beam_search else self.batch_size*self.beam_size
            
            decoder_initial_state = decoder_cell.zero_state(batch_size=batch_size,dtype=tf.float32).clone(cell_state=encoder_state)
            output_layer = tf.layers.Dense(self.vocab_size,kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.1))
            
            if self.mode == 'train':
                print("========train======")
                #decoder_inputs_embedded:[batch_size,decoder_length,embedding_size]
                ending = tf.strided_slice(self.decoder_inputs,[0,0],[self.batch_size,-1],[1,1])
                decoder_input = tf.concat([tf.fill([self.batch_size,1],self.start),ending],1)
                decoder_inputs_embeded = tf.nn.embedding_lookup(embedding,decoder_input)
                
                training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_inputs_embeded,
                                                                  sequence_length=self.decoder_length,
                                                                  time_major=False,name="training_helper")
                training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,helper=training_helper,
                                                                   initial_state=decoder_initial_state,output_layer=output_layer)
                
                decoder_outputs,_,_ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder,
                                                                        impute_finished=True,
                                                                        maximum_iterations=self.max_target_sequence_length)
                self.decoder_logits_train = tf.identity(decoder_outputs.rnn_output)
                self.decoder_predict_train = tf.argmax(self.decoder_logits_train,axis=-1,name="decoder_pred_train")
                #?????
                self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.decoder_logits_train,
                                                             targets=self.decoder_inputs,weights=self.mask)
                
                #training summary for the current batch_loss
                tf.summary.scalar('loss',self.loss)
                self.summary_op = tf.summary.merge_all()
                
                optimizer = tf.train.AdamOptimizer(self.lr)
                trainable_params = tf.trainable_variables()
                gradients = tf.gradients(self.loss,trainable_params)
                clip_gradients,_ = tf.clip_by_global_norm(gradients,self.max_gradient_norm)
                self.train_op = optimizer.apply_gradients(zip(clip_gradients,trainable_params))
            
            elif self.mode == 'test':
                print("========test======")
                start_tokens = tf.ones([self.batch_size,],tf.int32)*self.start
                end_token = self.stop
                
                if self.beam_search:
                    inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=decoder_cell,embedding=embedding,
                                                                             start_tokens=start_tokens,end_token=end_token,
                                                                             initial_state=decoder_initial_state,
                                                                             beam_width=self.beam_size,
                                                                             output_layer=output_layer)
                else:
                    decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=embedding,
                                                                               start_tokens=start_tokens,end_token=end_token)
                    inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,helper=decoding_helper,
                                                                        initial_state=decoder_initial_state,
                                                                        output_layer=output_layer)
                decoder_outputs,_,_ = tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder,
                                                                        maximum_iterations=self.max_iterations)
                
                if self.beam_search:
                    self.decoder_predict_decode = decoder_outputs.predicted_ids
                else:
                    self.decoder_predict_decode = tf.expand_dims(decoder_outputs.sample_id,-1)
            
            elif self.mode == 'valid':
                print("========valid======")
                start_tokens = tf.ones([self.batch_size,],tf.int32)*self.start
                end_token = self.stop
                
                decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=embedding,
                                                                           start_tokens=start_tokens,end_token=end_token)
                inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,helper=decoding_helper,
                                                                    initial_state=decoder_initial_state,
                                                                    output_layer=output_layer)
                decoder_outputs,_,_ = tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder,
                                                                        maximum_iterations=self.max_target_sequence_length)
                
                self.decoder_logits_valid = tf.identity(decoder_outputs.rnn_output)
                self.decoder_predict_valid = tf.argmax(self.decoder_logits_valid,axis=-1,name="decoder_predict_valid")

                #?????
                self.loss_valid = tf.contrib.seq2seq.sequence_loss(logits=self.decoder_logits_valid,
                                                             targets=self.decoder_inputs,weights=self.mask)
                
                #training summary for the current batch_loss
                #tf.summary.scalar('loss_valid',self.loss_valid)
                #self.summary_op = tf.summary.merge_all()
                
                
        
        '''=======================4. save model'''
        self.saver = tf.train.Saver(tf.global_variables())
        
    def train(self,sess,x):
        feed_dict = {self.encoder_inputs:x['enc_in'],
                     self.encoder_length:x['enc_len'],
                     self.decoder_inputs:x['dec_in'],
                     self.decoder_length:x['dec_len'],
#                     self.decoder_outputs:['dec_out'],
                     self.keep_prob_placeholder:0.5,
                     self.batch_size:len(x['enc_in'])}
        
        _,loss,summary = sess.run([self.train_op,self.loss,self.summary_op],feed_dict=feed_dict)
        
        return loss,summary
    
    def validate(self,sess,x):
        feed_dict = {self.encoder_inputs:x['enc_in'],
                     self.encoder_length:x['enc_len'],
                     self.decoder_inputs:x['dec_in'],
                     self.decoder_length:x['dec_len'],
#                     self.decoder_outputs:['dec_out'],
                     self.keep_prob_placeholder:1.0,
                     self.batch_size:len(x['enc_in'])}
        loss,predicts = sess.run([self.loss_valid,self.decoder_predict_valid],feed_dict=feed_dict)
        return loss,predicts
    
        
    def infer(self,sess,x):
        feed_dict = {self.encoder_inputs:x['enc_in'],
                     self.encoder_length:x['enc_len'],
                     self.keep_prob_placeholder:1.0,
                     self.batch_size:len(x['enc_in'])}
        predict = sess.run([self.decoder_predict_decode],feed_dict=feed_dict)
        return predict
        
            
        