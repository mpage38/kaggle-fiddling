import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression

class PredictionError(Exception):
    pass

def clean_data(tdf):
    tdf['gender'] = tdf['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    # Missing values will be filled with medians
    tdf = tdf.fillna(tdf.median())

    # Drop all the features that will not be used in logistic regression
    tdf = tdf.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId',
                    'Embarked'], axis=1)
    return tdf


def score_results(df, gt, pred):
    """ Score the results in a DataFrame that has both the ground
        truth and the prediction. gt is the name of the ground truth
        column etc.  Return the raw accuracy, the precision and the recall.
        Note that precision and recall are most useful for evaluating
        predictions on rare events. """
    correct = df.ix[df[gt] == df[pred]]
    incorrect = df.ix[df[gt] != df[pred]]
    true_pos = len(correct.ix[correct[gt] == 1])
    true_neg = len(correct.ix[correct[gt] == 0])
    false_neg = len(incorrect.ix[incorrect[pred] == 0])
    false_pos = len(incorrect.ix[incorrect[pred] == 1])

    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    num_correct = len(correct)
    tot_pred = len(df)
    accuracy = num_correct / tot_pred
    return accuracy, precision, recall


if __name__ == '__main__':
    trdf = pd.read_csv("train_mod.csv", header=0)
    trdf = clean_data(trdf)

    train = trdf.values
    logit = LogisticRegression()
    logit.fit(train[:, 1:], train[:, 0])

    testdf = pd.read_csv("test_mod.csv", header=0)
    pass_survival = testdf.copy()
    testdf = clean_data(testdf)

    c = logit.predict(testdf.values)

    # add the survival prediction column
    pass_survival['pred_survival'] = c

    # Get the ground truth dataset
    gtdf = pd.read_csv("titanic3_mod.csv", header=0)

    # Now merge the predictions with the ground truth.  This
    # is like a table join in the database world.
    ansdf = pd.merge(gtdf, pass_survival)

    # check to see if we got all the answers
    if len(pass_survival) != len(ansdf):
        raise PredictionError("Did not match all predictions with ground truth")

    accuracy, precision, recall = score_results(ansdf, 'Survived', 
                                                'pred_survival')

    print("accuracy: {}   precision: {}   recall: {}".format(accuracy,
           precision, recall))

    
