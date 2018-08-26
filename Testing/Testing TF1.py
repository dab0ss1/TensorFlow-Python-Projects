# dependencies
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf

# locations
IMPORT_PATH = '../../ShrunkData/AUD_CAD_H1.csv'

# main
def main():
    ''' Load in data '''
    global  IMPORT_PATH # path to data
    data = pd.read_csv(IMPORT_PATH) # load the data into a pandas dataframe
    for i in range(len(data['Open'])):
        total = data['Open'][i] + data['Close'][i]
        data.set_value(i, 'Open', data['Open'][i]/total)
        data.set_value(i, 'Close', data['Close'][i] / total)
    print(data.head()) # print a sample of the data

    ''' Turn classification column into 0 and 1 '''
    print(data['OpenGrtClose'].unique())
    data['OpenGrtClose'] = data['OpenGrtClose'].apply(label_fix)

    ''' Split the data into train and test '''
    x_data = data.drop('OpenGrtClose', axis=1)
    y_labels = data['OpenGrtClose']
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_labels, test_size=0.2, random_state=101)

    ''' Feature columns created for continuous values '''
    open = tf.feature_column.numeric_column("Open")
    close = tf.feature_column.numeric_column("Close")
    feat_cols = [open, close]

    ''' Create model and train '''
    input_func = tf.estimator.inputs.pandas_input_fn(x=x_train, y=y_train, batch_size=100, num_epochs=None, shuffle=True)
    model = tf.estimator.LinearClassifier(feature_columns=feat_cols)
    model.train(input_fn=input_func, steps=5000)

    ''' Evaluate model '''
    pred_fn = tf.estimator.inputs.pandas_input_fn(x=x_test, batch_size=len(x_test), shuffle=False)
    predictions = list(model.predict(input_fn=pred_fn))
    print(predictions[0])

    ''' List of predictions '''
    final_preds = []
    for pred in predictions:
        final_preds.append(pred['class_ids'][0])
    print(final_preds[:50])

    ''' Calculate model performance on test data '''
    print(classification_report(y_test, final_preds))

# end of main


def label_fix(label):
    if label == 0:
        return 0
    else:
        return 1

if __name__ == '__main__':
    main()