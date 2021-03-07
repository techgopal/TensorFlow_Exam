import pandas as pd


def get_data():
    # Data From TF Datasets
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                    'Acceleration', 'Model Year', 'Origin']

    df = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)
    df = df.dropna()

    print(df.size)

    #Transform Origin
    df['Origin'] = df['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
    df = pd.get_dummies(df, prefix=' ', prefix_sep=' ')

    print(df.size)

    # Split data
    train_dataset = df.sample(frac=0.8, random_state=0)
    test_dataset = df.drop(train_dataset.index)

    print('Train D '+str(train_dataset.size))

    X_train = train_dataset.copy()
    X_test = test_dataset.copy()
    y_train = X_train.pop('MPG')
    y_test = X_test.pop('MPG')

    print(X_train.size)
    print(y_train.size)

    return X_train, y_train,X_test, y_test
