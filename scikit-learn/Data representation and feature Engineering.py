import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mglearn
from sklearn.model_selection import train_test_split


def in11():
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    x, y = mglearn.datasets.make_wave(n_samples=100)
    line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
    reg = DecisionTreeRegressor(min_samples_split=3).fit(x, y)
    plt.plot(line, reg.predict(line), label='DecisionTreeRegressor')
    reg = LinearRegression().fit(x, y)
    plt.plot(line, reg.predict(line), label='LinearRegression')
    plt.plot(x[:, 0], y, 'o', c='k')
    plt.ylabel('output')
    plt.xlabel('input')
    plt.legend(loc='best')
    plt.show()


def in12():
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    x, y = mglearn.datasets.make_wave(n_samples=100)

    bins = np.linspace(-3, 3, 11)
    which_bin = np.digitize(x, bins=bins)
    from sklearn.preprocessing import OneHotEncoder
    encode = OneHotEncoder(sparse=False)
    encode.fit(which_bin)
    x_bins = encode.transform(which_bin)

    line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
    line_bins = encode.transform(np.digitize(line, bins=bins))

    reg = DecisionTreeRegressor(min_samples_split=3).fit(x_bins, y)
    plt.plot(line, reg.predict(line_bins), label='DecisionTreeRegressor')

    reg = LinearRegression().fit(x_bins, y)
    plt.plot(line, reg.predict(line_bins), label='LinearRegression')

    plt.plot(x[:, 0], y, 'o', c='k')
    plt.ylabel('output')
    plt.xlabel('input')
    plt.legend(loc='best')
    plt.show()


def in17():
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    x, y = mglearn.datasets.make_wave(n_samples=100)

    bins = np.linspace(-3, 3, 11)
    which_bin = np.digitize(x, bins=bins)
    from sklearn.preprocessing import OneHotEncoder
    encode = OneHotEncoder(sparse=False)
    encode.fit(which_bin)
    x_bins = encode.transform(which_bin)

    line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
    line_bins = encode.transform(np.digitize(line, bins=bins))

    x_combined = np.hstack([x, x_bins])

    reg = LinearRegression().fit(x_combined, y)
    line_combined = np.hstack([line, line_bins])
    plt.plot(line, reg.predict(line_combined), label='LinearRegression combined')
    for bin in bins:
        plt.plot([bin, bin], [-3, 3], ':', c='k')
    plt.plot(x[:, 0], y, 'o', c='k')
    plt.ylabel('output')
    plt.xlabel('input')
    plt.legend(loc='best')
    print([bin, bin])
    print([-3, 3])
    plt.show()


def in19():
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    x, y = mglearn.datasets.make_wave(n_samples=100)

    bins = np.linspace(-3, 3, 11)
    which_bin = np.digitize(x, bins=bins)
    from sklearn.preprocessing import OneHotEncoder
    encode = OneHotEncoder(sparse=False)
    encode.fit(which_bin)
    x_bins = encode.transform(which_bin)

    line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
    line_bins = encode.transform(np.digitize(line, bins=bins))

    x_combined = np.hstack([x, x_bins])

    reg = LinearRegression().fit(x_combined, y)
    line_combined = np.hstack([line, line_bins])

    x_product = np.hstack([x_bins, x * x_bins])

    reg = LinearRegression().fit(x_product, y)
    line_product = np.hstack([line_bins, line * line_bins])
    plt.plot(line, reg.predict(line_product), label='LinearRegression product')
    for bin in bins:
        plt.plot([bin, bin], [-3, 3], ':', c='k')
    plt.plot(x[:, 0], y, 'o', c='k')
    plt.ylabel('output')
    plt.xlabel('input')
    plt.legend(loc='best')
    plt.show()


def in21():
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    x, y = mglearn.datasets.make_wave(n_samples=100)
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=10, include_bias=False)
    poly.fit(x)
    x_poly = poly.transform(x)
    reg = LinearRegression().fit(x_poly, y)
    line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
    line_poly = poly.transform(line)
    plt.plot(line, reg.predict(line_poly), label='PolynomialFeatures LinearRegression ')
    plt.plot(x[:, 0], y, 'o', c='k', label='samples', color='red')
    plt.ylabel('output')
    plt.xlabel('input')
    plt.legend(loc='best')
    plt.show()


def in26():
    from sklearn.svm import SVR
    x, y = mglearn.datasets.make_wave(n_samples=100)
    line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
    for i in [1, 5, 10]:
        svr = SVR(gamma=i).fit(x, y)
        plt.plot(line, svr.predict(line), label='SVR gamma={}'.format(i))

    plt.plot(x[:, 0], y, 'o', c='k', label='samples', color='red')
    plt.ylabel('output')
    plt.xlabel('input')
    plt.legend(loc='best')
    plt.show()


def in27():
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    boston = load_boston()
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=0)
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=2).fit(x_train_scaled)
    x_train_poly = poly.transform(x_train_scaled)
    x_test_poly = poly.transform(x_test_scaled)
    #   print(poly.get_feature_names())
    from sklearn.linear_model import Ridge
    ridge = Ridge().fit(x_train_poly, y_train)
    print('{}'.format(ridge.score(x_train_poly, y_train)))
    print('{}'.format(ridge.score(x_test_poly, y_test)))


def in32():
    rnd = np.random.RandomState(0)
    x_org = rnd.normal(size=(1000, 3))
    w = rnd.normal(size=3)
    x = rnd.poisson(10 * np.exp(x_org))
    y = np.dot(x_org, w)
    bins = np.bincount(x[:, 0])
    '''
    plt.bar(range(len(bins)),bins,color='red')
    plt.ylabel('appearance')
    plt.xlabel('value')
    plt.show()

    from sklearn.linear_model import Ridge
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
    ridge=Ridge().fit(x_train,y_train)
    print(ridge.score(x_train,y_train))
    print(ridge.score(x_test,y_test))


    '''
    from sklearn.linear_model import Ridge
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
    x_train_log = np.log(x_train + 1)
    x_test_log = np.log(x_test + 1)
    plt.hist(x_train_log[:, 0], bins=25, color='red')
    plt.xlabel('value')
    plt.ylabel('appearance')
    plt.show()
    print(x[:2])
    print(y[:2])


def in39():
    from sklearn.datasets import load_breast_cancer
    from sklearn.feature_selection import SelectPercentile
    from sklearn.model_selection import train_test_split
    cancer = load_breast_cancer()
    rng = np.random.RandomState(42)
    noise = rng.normal(size=(len(cancer.data), 50))
    #   print(cancer.data.shape) (596,30)
    x_w_noise = np.hstack([cancer.data, noise])
    x_train, x_test, y_train, y_test = train_test_split(x_w_noise, cancer.target, random_state=0, test_size=0.5)
    select = SelectPercentile(percentile=50).fit(x_train, y_train)
    x_train_s = select.transform(x_train)
    mask = select.get_support()

    plt.matshow(mask.reshape(1, -1), cmap='gray_r')
    plt.xlabel('sample index')
    plt.show()

    from sklearn.linear_model import LogisticRegression
    x_test_s = select.transform(x_test)
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    print(lr.score(x_test, y_test))
    lr.fit(x_train_s, y_train)
    print(lr.score(x_test_s, y_test))


def in42():
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import RandomForestClassifier
    select=SelectFromModel(RandomForestClassifier(n_estimators=100,random_state=42),threshold='median')

    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    cancer = load_breast_cancer()
    rng = np.random.RandomState(42)
    noise = rng.normal(size=(len(cancer.data), 50))
    #   print(cancer.data.shape) (596,30)
    x_w_noise = np.hstack([cancer.data, noise])
    x_train, x_test, y_train, y_test = train_test_split(x_w_noise, cancer.target, random_state=0, test_size=0.5)

    select.fit(x_train,y_train)
    x_train_l1=select.transform(x_train)

    mask = select.get_support()
    plt.matshow(mask.reshape(1, -1), cmap='gray_r')
    plt.xlabel('sample index')
    plt.show()

    x_test_l1=select.transform(x_test)
    from sklearn.linear_model import LogisticRegression
    print(LogisticRegression().fit(x_train,y_train).score(x_test,y_test))
    print(LogisticRegression().fit(x_train_l1, y_train).score(x_test_l1, y_test))


def in46():
    from sklearn.feature_selection import RFE
    from sklearn.ensemble import RandomForestClassifier
    select=RFE(RandomForestClassifier(n_estimators=100,random_state=42),n_features_to_select=40)

    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    cancer = load_breast_cancer()
    rng = np.random.RandomState(42)
    noise = rng.normal(size=(len(cancer.data), 50))
    #   print(cancer.data.shape) (596,30)
    x_w_noise = np.hstack([cancer.data, noise])
    x_train, x_test, y_train, y_test = train_test_split(x_w_noise, cancer.target, random_state=0, test_size=0.5)

    select.fit(x_train,y_train)

    mask = select.get_support()
    plt.matshow(mask.reshape(1, -1), cmap='gray_r')
    plt.xlabel('sample index')
    plt.show()

    x_train_rfe=select.transform(x_train)
    x_test_rfe=select.transform(x_test)

    from sklearn.linear_model import LogisticRegression
    print(LogisticRegression().fit(x_train, y_train).score(x_test, y_test))
    print(LogisticRegression().fit(x_train_rfe, y_train).score(x_test_rfe, y_test))



# use the first 184 data points for training, the rest for testing
n_train = 184

# function to evaluate and plot a regressor on a given feature set
def eval_on_features(features, target, regressor):
    # split the given features into a training and a test set
    X_train, X_test = features[:n_train], features[n_train:]
    # also split the target array
    y_train, y_test = target[:n_train], target[n_train:]
    regressor.fit(X_train, y_train)
    print("Test-set R^2: {:.2f}".format(regressor.score(X_test, y_test)))
    y_pred = regressor.predict(X_test)
    y_pred_train = regressor.predict(X_train)
    plt.figure(figsize=(10, 3))
    citibike = mglearn.datasets.load_citibike()
    # extract the target values (number of rentals)
    y = citibike.values
    # convert to POSIX time by dividing by 10**9
    X = citibike.index.astype("int64").values.reshape(-1, 1)

    plt.xticks(range(0, len(X), 8), rotation=90,
               ha="left")

    plt.plot(range(n_train), y_train, label="train")
    plt.plot(range(n_train, len(y_test) + n_train), y_test, '-', label="test")
    plt.plot(range(n_train), y_pred_train, '--', label="prediction train")

    plt.plot(range(n_train, len(y_test) + n_train), y_pred, '--',
             label="prediction test")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Rentals")
    plt.show()

def in49():
    citibike=mglearn.datasets.load_citibike()

    ''''
     xticks = pd.date_range(start=citibike.index.min(), end=citibike.index.max(),
                           freq='D')
    plt.xticks(xticks.astype("int"), xticks.strftime("%a %m-%d"), rotation=90, ha="left")
    plt.plot(citibike, linewidth=1)
    plt.xlabel("Date")
    plt.ylabel("Rentals")
    plt.show()
    
    '''

    # extract the target values (number of rentals)
    y = citibike.values
    # convert to POSIX time by dividing by 10**9
    X= citibike.index.astype("int64").values.reshape(-1, 1)
    from sklearn.ensemble import RandomForestRegressor
    re=RandomForestRegressor(n_estimators=100,random_state=0)

    eval_on_features(X,y,re)
    

in49()