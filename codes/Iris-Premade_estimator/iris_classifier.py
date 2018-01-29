import pandas as pd
import tensorflow as tf
import urllib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=100, type=int, help="batch Size")
parser.add_argument("--train_steps", default=1000, type=int, help="number of training Steps")

train_url = "http://download.tensorflow.org/data/iris_training.csv"
test_url = "http://download.tensorflow.org/data/iris_test.csv"
column_name = ['SepalLength', 'SepalWidth','PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Sentosa', 'Versicolor', 'Virginica']
def download(path="dataset"):
    train_path = path + '/' + train_url.split('/')[-1]
    test_path = path + '/' + test_url.split('/')[-1]
    print "Downloading Train Data"
    urllib.urlretrieve(train_url, train_path)
    print "Downloading Test data"
    urllib.urlretrieve(test_url, test_path)
    return train_path, test_path
def load_data(y_name = 'Species'):
    train_path, test_path = download()
    train = pd.read_csv(train_path, names = column_name, header = 0)
    train_x, train_y = train, train.pop(y_name)
    test = pd.read_csv(test_path, names = column_name, header = 0)
    test_x, test_y = test, test.pop(y_name)
    return (train_x, train_y), (test_x, test_y)
def train_input_fun(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()
def eval_input_fn(features, labels, batch_size):
    features = dict(features)
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    assert batch_size is not None, "Batch Siz must not be None"
    dataset = dataset.batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()
CSV_TYPES = [[0.0],[0.0],[0.0],[0.0],[0]]

def _parse_line(line):
    fields = tf.decode_csv(line, record_defaults = CSV_TYPES)
    features = dict(zip(CSV_COLUMN_NAMES, fields))
    label = features.pop('Species')
    return features, label
def csv_input_fn(csv_path, batch_size):
    dataset = tf.data.TextLineDataset(csv_path).skip(1)
    dataset = dataset.map(_parse_line)
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()

def main(argv):
    args = parser.parse_args(argv[1:])
    (train_x, train_y), (test_x, test_y) = load_data()
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    classifier = tf.estimator.DNNClassifier(
    feature_columns = my_feature_columns,
    hidden_units = [10,10],
    model_dir = "/home/dhinesh/iris",
    n_classes = 3
    )
    classifier.train(
    input_fn = lambda:train_input_fun(train_x, train_y, args.batch_size),steps = args.train_steps)
    eval_result = classifier.evaluate(
    input_fn = lambda:eval_input_fn(test_x, test_y, args.batch_size))

    print ("\n Test set Accuracy: {accuracy: 0.3f}\n". format(**eval_result))
    expected = ["Sentosa","Versicolor", "Virginica"]
    predict_x = {
    'SepalLength':[5.1,5.9,6.9],
    'SepalWidth':[3.3, 3.0, 3.1],
    'PetalLength': [1.7, 4.2, 5.4],
    'PetalWidth': [0.5, 1.5, 2.1]
    }
    predictions = classifier.predict(input_fn = lambda:eval_input_fn(predict_x, labels=None,
    batch_size = args.batch_size))
    for pred_dict, expec in zip(predictions, expected):
        template = ('\n Prediction is "{}" ({:.1f}%), expected "{}"')
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]
        print(template.format(SPECIES[class_id], 100*probability,expec))


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
