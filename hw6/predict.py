import sys
import pandas as pd
import numpy as np
from keras.models import load_model

df = pd.read_csv(sys.argv[1]).values
users = []
movies = []

for i in range(len(df)):
	users.append(int(df[i][1]))
	movies.append(int(df[i][2]))
users = np.array(users)
movies = np.array(movies)

model = load_model("ball3.h5")
prediction = model.predict([users, movies], batch_size = 512, verbose = 1)
index = np.arange(len(prediction)) + 1
prediction = prediction.reshape(-1)
prediction = np.clip(prediction, 1, 5)
df = pd.DataFrame.from_items([('TestDataID',index.tolist()),('Rating',prediction.tolist())])
df.to_csv(sys.argv[2], index = False)
# my implementation of mf is from last year's sample code: https://docs.google.com/presentation/d/1oR3JJz7wVd5GD78AX9qImk3k-uS7cGZTGyxXeYraRlY/edit#slide=id.g1f620502c7_0_36
# The method of writing csv file in this way is told by my classmate b05902043
