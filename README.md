# ts
1. Import
2. pd.read_csv kek biasa
3. check pake plot
4. from statsmodels.tsa.seasonal import seasonal_decompose
5. seasonal_decompose(df[''])
6. train test pake iloc
7. normalize pake minmaxscaler
8. from keras.preprocessing.sequence import TimeseriesGenerator
9. define generator
n_input = 3
n_features = 1
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)
10. X,y = generator[0]
print(f'Given the Array: \n{X.flatten()}')
print(f'Predict this y: \n {y}')
11. We do the same thing, but now instead for 12 months
n_input = 12
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)
12. import sequential dari keras.models, dense sama LSTM dari keras.layers
13. # define model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
14. model.fit parameter generatornya sama epoch 50 bebas
15. loss_per_epoch = model.history.history['loss']
plt.plot(range(len(loss_per_epoch)),loss_per_epoch)
16. untuk predict
last_train_batch = scaled_train[-12:]
last_train_batch = last_train_batch.reshape((1, n_input, n_features))
model.predict(last_train_batch)
scaled_test[0]
17. test_predictions = []

first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(len(test)):

  get the prediction value for the first batch
  current_pred = model.predict(current_batch)[0]

  append the prediction into the array
  test_predictions.append(current_pred)

  use the prediction to update the batch and remove the first value
  current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)

test_predictions

test.head()

true_predictions = scaler.inverse_transform(test_predictions)

test['Predictions'] = true_predictions

test.plot(figsize=(14,5))

from sklearn.metrics import mean_squared_error
from math import sqrt
rmse=sqrt(mean_squared_error(test['Production'],test['Predictions']))
print(rmse)

sa
1. import tambahan sns sama nltk, nltk.download('puntk')
2. from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
sns.set()
3. read dataset
4. imdb.sentiment.value_counts()
5. for loop dan diword tokenize yaitu dipisahin kata per kata
corpus = []
for text in imdb['review']:
  words = [word.lower() for word in word_tokenize(text)]
  corpus.append(words)
6. print len sama shape
7. train test split dari shapenya
train_size = int(imdb.shape[0] * 0.8)
X_train = imdb.review[:train_size]
y_train = imdb.sentiment[:train_size]

X_test = imdb.review[train_size:]
y_test = imdb.sentiment[train_size:]
8. Lakukan Tokenizer yaitu membuat textnya jadi array angka
tokenizer = Tokenizer(num_words)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_train = pad_sequences(X_train, maxlen=128, truncating='post', padding='post')

X_train[0], len(X_train[0])

X_test = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(X_test, maxlen=128, truncating='post', padding='post')

X_test[0], len(X_test[0])

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
9. diLabelEncoder()
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
10. buat modelnya, summary, dan fit
model = Sequential()

model.add(Embedding(input_dim=num_words, output_dim=100,
                    input_length=128, trainable=True))
model.add(LSTM(100, dropout=0.1, return_sequences=True))
model.add(LSTM(100, dropout=0.1))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

11. Plotting
plt.figure(figsize=(16,5))
epochs = range(1, len(history.history['accuracy'])+1)
plt.plot(epochs, history.history['loss'], 'b', label='Training Loss', color'red')
plt.plot(epochs, history.history['val_loss'], 'b', label='Validation Loss')
plt.legend()
plt.show()

satu lagi yang accuracy

12. predict masukin sentence
validation_sentence = ['']
validation_sentence_tokened = tokenizer.texts_to_sequences(validation_sentence)
validation_sentence_padded = pad_sequences(validation_sentence_tokened, maxlen=128,
                                           truncating='post', padding='post')
print(validation_sentence[0])
print("Probability of Positive: {}".format(model.predict(validation_sentence_padded)[0]))
