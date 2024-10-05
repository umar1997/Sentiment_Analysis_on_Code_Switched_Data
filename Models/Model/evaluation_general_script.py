import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import hamming_loss, accuracy_score 
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report


###############################################################################
# https://stats.stackexchange.com/questions/233275/multilabel-classification-metrics-on-scikit
def get_exact_match_ratio(expected, predicted): # aka Subset Accuracy
    """
    This is the total examples predicted correctly/ Total examples
    total examples predicted correctly: Means that all the labels in that 1 example are correct
    For e.g have 3 examples with 4 labels then then the Total examples = 3
    """
    return accuracy_score(expected, predicted, normalize=True, sample_weight=None)

def get_accuracy_score(expected, predicted):
    """
    This is the total correct predictions (0 & 1) / Total prediction
    For e.g have 3 examples with 4 labels then then the Total predictions = 12
    """
    accuracy_score = sum(np.where(expected==predicted, True, False).reshape(-1))/predicted.size
    return accuracy_score

def get_classification_report(expected, predicted, labels):
    return classification_report(expected, predicted, target_names=labels, zero_division=1)

def get_hamming_score(expected, predicted):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    https://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(expected.shape[0]):
        set_true = set( np.where(expected[i])[0] )
        set_pred = set( np.where(predicted[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
    return np.mean(acc_list)

if __name__ == "__main__":

    ################################### Example 1 ###################################
    # y_true = np.array([[0,1,0],
    #                [0,1,1],
    #                [1,0,1],
    #                [0,0,1]])

    # y_pred = np.array([[0,1,1],
    #                 [0,1,1],
    #                 [0,1,0],
    #                 [0,0,1]])

    # print(get_hamming_score(y_true, y_pred))
    # print(get_accuracy_score(y_true, y_pred))
    # print(get_exact_match_ratio(y_true, y_pred))
    # breakpoint()




    ################################### Example 2 ###################################
    # https://towardsdatascience.com/evaluating-multi-label-classifiers-a31be83da6ea

    # A = np.array([[1, 0, 1], [0, 0, 1], [1, 1, 1]])
    # B = np.array([[1, 1, 0], [0, 0, 1], [0, 1, 1]])

    # confusion_matrix_A = multilabel_confusion_matrix(A, B)[0] # Confusion Matrix for 1 class
    # confusion_matrix_ALL = multilabel_confusion_matrix(A, B) # Confusion Matrix for ALL classes

    # accuracy = sum(np.where(A==B, True, False).reshape(-1))/A.size
    # print(accuracy)
    # print(classification_report(A, B, output_dict=False,target_names=['class A', 'class B', 'class C']))
    # breakpoint()

    ################################### Example 3 ###################################
    # Example Multilabel Binarizer
    expected = [['A', 'C'],['C'],['A', 'B', 'C']]
    predicted = [['A', 'B'], ['C'], ['B', 'C']]
    # predicted = [['A', 'B'], ['C'], ['A', 'B', 'C']]

    classes = ['A','B','C','D']
    mlb = MultiLabelBinarizer()
    mlb.fit([classes])
    gold = mlb.transform(expected)
    pred = mlb.transform(predicted)
    breakpoint()
    # Output:
    # gold
    # array([[1, 0, 1, 0],
    #     [0, 0, 1, 0],
    #     [1, 1, 1, 0]])
    # pred
    # array([[1, 1, 0, 0],
    #     [0, 0, 1, 0],
    #     [0, 1, 1, 0]])

    # macro_f1 = f1_score(gold, pred, average="macro", zero_division=1)
    # micro_f1 = f1_score(gold, pred, average="micro", zero_division=1)

    # Accuracy
    # accuracy = accuracy_score(gold, pred)*100
    # This is the same as the exact match ratio we are calculating above
    #  """
    # This is the total examples predicted correctly/ Total examples
    # total examples predicted correctly: Means that all the labels in that 1 example are correct
    # For e.g have 3 examples with 4 labels then then the Total examples = 3
    # """

    # Classification Report
    # classification_report(gold, pred, zero_division=1)
