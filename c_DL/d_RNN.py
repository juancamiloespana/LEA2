
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer ### este paquete ya no es muy usado el paquete keras_nlp tiene funciones de preprocesamiento pero no funciona para windows
from tensorflow.keras.preprocessing.sequence import pad_sequences ## estandarizar tamaños de textos
from tensorflow import keras

from datasets import load_dataset ### paquete con conjunto de datos
import random
# Cargar datos con comentarios de películas
imdb_data = load_dataset('imdb')





# Guar datos de diccionario en variables
random.seed(42)
train_texts_full = imdb_data['train']['text']
random.shuffle(train_texts_full) ## ordena la lista de manera aleatoria
train_texts=train_texts_full[:500]
train_texts[0] ### ejemplo de primer revisión

random.seed(42)
train_labels_full = imdb_data['train']['label']
random.shuffle(train_labels_full) ## ordena la lista de manera aleatoria
train_labels=train_labels_full[:500]
train_labels[0] ## ejemplo primer etiqueta



random.seed(42)
test_texts_full = imdb_data['test']['text']
random.shuffle(test_texts_full) ## ordena la lista de manera aleatoria
test_texts=test_texts_full[:500]
test_texts[0]

random.seed(42)
test_labels_full = imdb_data['test']['label']
random.shuffle(test_labels_full) ## ordena la lista de manera aleatoria
test_labels=test_labels_full[:500]
test_labels[0]

# Volver datos tokens(palabra o grupo de palabras) y generar onehot encoding

num_words = 10000 ## definir tamaño de la representación
maxlen = 500 ## máximo tamaño de textos

tokenizer = Tokenizer(num_words=num_words) ## configurar tokenizer
tokenizer.fit_on_texts(train_texts) ## entrenar en datos

x_train = tokenizer.texts_to_sequences(train_texts) ## convierte el texto en una lista de número, cada número representa la posición de cada palabra en el diccionario construido
x_train[0][0]
len(x_train[0])
len(x_train[1])

x_test = tokenizer.texts_to_sequences(test_texts)


# llenar de ceros las primeras posiciones para completar un vector de tamaño 500 y que la longitud de cada input sea estándar
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)
x_train[0]
len(x_train[0])
len(x_train[1])


# Convertir labels a formato tensor
y_train = tf.convert_to_tensor(train_labels)
y_test = tf.convert_to_tensor(test_labels)


###################################################
# Construir red recurrente simple#################
##################################################

model_rnn = keras.models.Sequential()
model_rnn.add(keras.layers.Embedding(input_dim=num_words, output_dim=64, input_length=maxlen))
model_rnn.add(keras.layers.SimpleRNN(64, activation='tanh'))
model_rnn.add(keras.layers.Dropout(0.5))
model_rnn.add(keras.layers.Dense(1, activation='sigmoid'))

# Compilar modelo
model_rnn.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
model_rnn.fit(x_train, y_train,epochs=5, validation_data=(x_test, y_test))

model_rnn.summary()




##################################################
# Construir red recurrente con GRU
##################################################
model_gru = keras.models.Sequential()
model_gru.add(keras.layers.Embedding(input_dim=num_words, output_dim=64, input_length=maxlen))
model_gru.add(keras.layers.GRU(64, activation='tanh'))
model_gru.add(keras.layers.Dropout(0.5))
model_gru.add(keras.layers.Dense(1, activation='sigmoid'))

# Compilar modelo
model_gru.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
model_gru.fit(x_train, y_train,epochs=5, validation_data=(x_test, y_test))

model_gru.summary()




##################################################
# Construir red recurrente con LSTM
##################################################
model_lstm = keras.models.Sequential()
model_lstm.add(keras.layers.Embedding(input_dim=num_words, output_dim=64, input_length=maxlen))
model_lstm.add(keras.layers.LSTM(64, activation='tanh'))
model_lstm.add(keras.layers.Dropout(0.5))
model_lstm.add(keras.layers.Dense(1, activation='sigmoid'))

# Compilar modelo
model_lstm.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
model_lstm.fit(x_train, y_train,epochs=5, validation_data=(x_test, y_test))

model_lstm.summary()