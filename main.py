import numpy as np
import pandas as pd

exo_train_df = pd.read_csv(
    'https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/whitehat-ds-datasets/kepler-exoplanets-dataset/exoTrain.csv')
exo_test_df = pd.read_csv(
    'https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/whitehat-ds-datasets/kepler-exoplanets-dataset/exoTest.csv')


def mean_normalise(series):
    norm_series = (series - series.mean()) / (series.max() - series.min())
    return norm_series


norm_train_df = exo_train_df.iloc[:, 1:].apply(mean_normalise, axis=1)
norm_train_df.insert(loc=0, column='LABEL', value=exo_train_df['LABEL'])
norm_test_df = exo_test_df.iloc[:, 1:].apply(mean_normalise, axis=1)
norm_test_df.insert(loc=0, column='LABEL', value=exo_test_df['LABEL'])
exo_train_df.T


def fast_fourier_transform(star):
    fft_star = np.fft.fft(star, n=len(star))
    return np.abs(fft_star)


freq = np.fft.fftfreq(len(exo_train_df.iloc[0, 1:]))
x_fft_train_T = norm_train_df.iloc[:, 1:].T.apply(fast_fourier_transform)
x_fft_train = x_fft_train_T.T

x_fft_test_T = norm_test_df.iloc[:, 1:].T.apply(fast_fourier_transform)
x_fft_test = x_fft_test_T.T
y_train = norm_train_df['LABEL']
y_test = norm_test_df['LABEL']
from imblearn.over_sampling import SMOTE

sm = SMOTE(ratio=1)

x_fft_train_res, y_fft_train_res = sm.fit_sample(x_fft_train, y_train)
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xg

xgb = xg.XGBClassifier()
xgb.fit(x_fft_train_res, y_fft_train_res)
y2_pred = xgb.predict(np.array(x_fft_test))
print('prediction -->', y2_pred)
print('acuracy stated by the module -->',xgb.score(x_fft_train_res, y_fft_train_res))
print('confusion matrix-->', confusion_matrix(y_test, y2_pred))
print('classification report -->', classification_report(y_test, y2_pred))
print('\n Very accurate but not the most. Target aquired!')