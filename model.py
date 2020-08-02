import tensorflow as tf
from Layers import EncoderLayer,DecoderLayer
from Embed import positional_encoding
from Batch import create_masks

class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (batch_size, 64, hidden_size)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

        # attention_weights shape == (batch_size, 64, 1)
        # you get 1 at the last axis because you are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

class ImageCaptioningEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                maximum_position_encoding, rate=0.1):
        super(ImageCaptioningEncoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        
        # self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        # self.pos_encoding = positional_encoding(maximum_position_encoding, 
        #                                         self.d_model)
        
        self.affine = tf.keras.layers.Dense(d_model)
        
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                        for _ in range(num_layers)]
    
        self.dropout = tf.keras.layers.Dropout(rate)
            
    
    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]
        
        # adding embedding and position encoding.
        # x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        # x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # x += self.pos_encoding[:, :seq_len, :]

        x = self.affine(x)

        x = self.dropout(x, training=training)
        
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        
        return x  # (batch_size, input_seq_len, d_model)

class ImageCaptioningDecoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                maximum_position_encoding, rate=0.1):
        super(ImageCaptioningDecoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                        for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
        
    def call(self, x, enc_output, training, 
            look_ahead_mask, padding_mask):

        seq_len = tf.shape(x)[1]
        attention_weights = {}
        
        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                look_ahead_mask, padding_mask)
        
        attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
        attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
        
        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights

class ImageCaptioningTransformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff,pe_target, vocab_size,rate=0.1):
        super(ImageCaptioningTransformer, self).__init__()

        self.encoder = ImageCaptioningEncoder(num_layers, d_model, num_heads, dff, 
                            vocab_size, rate)

        self.decoder = ImageCaptioningDecoder(num_layers, d_model, num_heads, dff, 
                            vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(vocab_size)
        

    def call(self, inp, tar, training, enc_padding_mask, 
            look_ahead_mask, dec_padding_mask):
        '''
        enc_padding_mask :  create_padding_mask(inp) 
        look_ahead_mask  :  
        dec_padding_mask :  create_padding_mask(inp) use to mask the encoder ouputs for attention block
        '''


        enc_output = self.encoder(inp, training, None)  # (batch_size, inp_seq_len, d_model)
        
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        
        # return final_output, attention_weights 
        return final_output


if __name__ == '__main__':
    # encoder = CNN_Encoder(256)
    # decoder = RNN_Decoder(256, 512, 5000)
    out_encoder_feature = tf.random.uniform((64, 64, 256))
    num_layers = 4
    d_model = 512
    num_heads = 8
    dff = 1024
    vocab_size = 5000
    maximum_position_encoding = 200
    # tar_inp = tar[:, :-1]
    # tar_real = tar[:, 1:]
    decoder = ImageCaptioningTransformer(num_layers,d_model,num_heads,dff,maximum_position_encoding,vocab_size)

    temp_target =  tf.random.uniform((64,1),maxval = vocab_size , dtype = tf.int32)
    x = tf.expand_dims([0] * temp_target.shape[0], 1)
    # print(temp_target.shape,x.shape)
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(x, temp_target)
    
    output = decoder(out_encoder_feature, temp_target, training=True, 
                               enc_padding_mask=enc_padding_mask, 
                               look_ahead_mask=combined_mask,
                               dec_padding_mask=dec_padding_mask)
    print(output.shape)

