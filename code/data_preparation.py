from math import sin, cos, pi

import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    accuracy_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


def preprocess_data(df):
    """Add in externally obtained location data and clean certain categorical field containing NaN"""
    df = df.dropna(subset=["RainToday", "RainTomorrow"])
    df["RainToday"] = df["RainToday"].replace("No", 0).replace("Yes", 1).astype(float)
    df["RainTomorrow"] = (
        df["RainTomorrow"].replace("No", 0).replace("Yes", 1).astype(float)
    )
    df["WindGustDir"] = df["WindGustDir"].fillna("NaN")
    df["WindDir9am"] = df["WindDir9am"].fillna("NaN")
    df["WindDir3pm"] = df["WindDir3pm"].fillna("NaN")

    df["Date"] = pd.to_datetime(df["Date"])

    latitude_longitude = {
        "Adelaide": {"Latitude": 34.9285, "Longitude": 138.6007},
        "Albany": {"Latitude": 35.0269, "Longitude": 117.8837},
        "Albury": {"Latitude": 36.0737, "Longitude": 146.9135},
        "AliceSprings": {"Latitude": 23.6980, "Longitude": 133.8807},
        "BadgerysCreek": {"Latitude": 33.8829, "Longitude": 150.7609},
        "Ballarat": {"Latitude": 37.5622, "Longitude": 143.8503},
        "Bendigo": {"Latitude": 36.7570, "Longitude": 144.2794},
        "Brisbane": {"Latitude": 27.4705, "Longitude": 153.0260},
        "Cairns": {"Latitude": 16.9186, "Longitude": 145.7781},
        "Canberra": {"Latitude": 35.2809, "Longitude": 149.1300},
        "Cobar": {"Latitude": 31.4958, "Longitude": 145.8389},
        "CoffsHarbour": {"Latitude": 30.2986, "Longitude": 153.1094},
        "Dartmoor": {"Latitude": 37.9144, "Longitude": 141.2730},
        "Darwin": {"Latitude": 12.4637, "Longitude": 130.8444},
        "GoldCoast": {"Latitude": 28.0167, "Longitude": 153.4000},
        "Hobart": {"Latitude": 42.8826, "Longitude": 147.3257},
        "Katherine": {"Latitude": 14.4520, "Longitude": 132.2699},
        "Launceston": {"Latitude": 41.4391, "Longitude": 147.1358},
        "Melbourne": {"Latitude": 37.8136, "Longitude": 144.9631},
        "MelbourneAirport": {"Latitude": 37.6690, "Longitude": 144.8410},
        "Mildura": {"Latitude": 34.2080, "Longitude": 142.1246},
        "Moree": {"Latitude": 29.4658, "Longitude": 149.8339},
        "MountGambier": {"Latitude": 37.8284, "Longitude": 140.7804},
        "MountGinini": {"Latitude": 35.5294, "Longitude": 148.7723},
        "Newcastle": {"Latitude": 32.9283, "Longitude": 151.7817},
        "Nhil": {"Latitude": 36.3328, "Longitude": 141.6503},
        "NorahHead": {"Latitude": 33.2833, "Longitude": 151.5667},
        "NorfolkIsland": {"Latitude": 29.0408, "Longitude": 167.9547},
        "Nuriootpa": {"Latitude": 34.4666, "Longitude": 138.9917},
        "PearceRAAF": {"Latitude": 31.6676, "Longitude": 116.0292},
        "Penrith": {"Latitude": 33.7507, "Longitude": 150.6877},
        "Perth": {"Latitude": 31.9523, "Longitude": 115.8613},
        "PerthAirport": {"Latitude": 31.9484, "Longitude": 115.9726},
        "Portland": {"Latitude": 38.3609, "Longitude": 141.6041},
        "Richmond": {"Latitude": 37.8230, "Longitude": 144.9980},
        "Sale": {"Latitude": 38.1026, "Longitude": 147.0730},
        "SalmonGums": {"Latitude": 32.9815, "Longitude": 121.6438},
        "Sydney": {"Latitude": 33.8688, "Longitude": 151.2093},
        "SydneyAirport": {"Latitude": 33.9399, "Longitude": 151.1753},
        "Townsville": {"Latitude": 19.2590, "Longitude": 146.8169},
        "Tuggeranong": {"Latitude": 35.4244, "Longitude": 149.0888},
        "Uluru": {"Latitude": 25.3444, "Longitude": 131.0369},
        "WaggaWagga": {"Latitude": 35.1082, "Longitude": 147.3598},
        "Walpole": {"Latitude": 34.9777, "Longitude": 116.7338},
        "Watsonia": {"Latitude": 37.7080, "Longitude": 145.0830},
        "Williamtown": {"Latitude": 32.8150, "Longitude": 151.8428},
        "Witchcliffe": {"Latitude": 34.0261, "Longitude": 115.1003},
        "Wollongong": {"Latitude": 34.4278, "Longitude": 150.8931},
        "Woomera": {"Latitude": 31.1656, "Longitude": 136.8193},
    }
    for key in ["Latitude", "Longitude"]:
        df[key] = df["Location"].map(lambda loc: latitude_longitude[loc][key])
    return df


def preprocess_non_time_series(df):
    """Create time based cyclic encoded columns to help model learn cyclical behavior more easily"""
    day_of_year = df["Date"].dt.day_of_year
    df["DayOfYear_Sin"] = day_of_year.map(lambda x: sin(2.0 * pi * x / 365))
    df["DayOfYear_Cos"] = day_of_year.map(lambda x: cos(2.0 * pi * x / 365))
    df["Year"] = df["Date"].dt.year
    df.drop("Date", axis=1, inplace=True)

    return df


def preprocess_time_series(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    period_index = df.index.to_period("D")
    df.index = df.index.to_period("D")
    return df


def fit_predict(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    confusion = confusion_matrix(y_test, y_pred, normalize="all")
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred)
    summary = {
        "accuracy": accuracy_score(y_test, y_pred),
        "classifier": clf,
        "y_pred": y_pred,
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
        "support": support,
        "confusion_matrix": confusion,
    }
    try:
        y_pred_proba = clf.predict_proba(X_test)
        summary["pred_proba"] = y_pred_proba
        summary["average_precision_score"] = average_precision_score(
            y_test, y_pred_proba[:, 1]
        )
        summary["precision_recall_curve"] = precision_recall_curve(
            y_test, y_pred_proba[:, 1]
        )
        summary["roc_auc_score"] = roc_auc_score(y_test, y_pred_proba[:, 1])
        summary["roc_curve"] = roc_curve(y_test, y_pred_proba[:, 1])
    except AttributeError:
        y_decision_function = clf.decision_function(X_test)
        summary["decision_function"] = y_decision_function
        summary["average_precision_score"] = average_precision_score(
            y_test, y_decision_function
        )
        summary["precision_recall_curve"] = precision_recall_curve(
            y_test, y_decision_function
        )
        summary["roc_auc_score"] = roc_auc_score(y_test, y_decision_function)
        summary["roc_curve"] = roc_curve(y_test, y_decision_function)
    return summary
