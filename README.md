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
