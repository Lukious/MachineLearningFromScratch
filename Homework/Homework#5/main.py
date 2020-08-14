import numpy as np
import csv
import time

np.random.seed(20200223)
def randomize(): np.random.seed(time.time())

RND_MEAN = 0
RND_STD = 0.0030

LEARNING_RATE = 0.001

def abalone_exec(epoch_count=10, mb_size=10, report=1):
    load_abalone_dataset()
    init_model()
    train_and_test(epoch_count, mb_size, report)

def load_abalone_dataset():
    with open('./data.csv') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader, None)
        rows = []
        for row in csvreader:
            rows.append(row)
            
    global data, input_cnt, output_cnt
    input_cnt, output_cnt = 10, 1
    data = np.zeros([len(rows), input_cnt+output_cnt])

    for n, row in enumerate(rows):
        #one Hot encode
        data[n, 3:] = row[1:]    

def init_model():
    global weight, bias, input_cnt, output_cnt
    weight = np.random.normal(RND_MEAN, RND_STD,[input_cnt, output_cnt])
    bias = np.zeros([output_cnt])

def train_and_test(epoch_count, mb_size, report):
    step_count = arrange_data(mb_size)
    test_x, test_y = get_test_data()
    
    # inplement epoch learning algorithm

    final_acc = run_test(test_x, test_y)
    print('\nFinal Test: final accuracy = {:5.3f}'.format(final_acc))

def arrange_data(mb_size):
    global data, shuffle_map, test_begin_idx
    #inplement suffle
    #set step count
    test_begin_idx = step_count * mb_size
    return step_count

def get_test_data():
    #import test_data splitor
    return test_data[:, :-output_cnt], test_data[:, -output_cnt:]

def get_train_data(mb_size, nth):
    #import train_data splitor
    return train_data[:, :-output_cnt], train_data[:, -output_cnt:]

def run_train(x, y):
    output, aux_nn = forward_neuralnet(x)
    loss, aux_pp = forward_postproc(output, y)
    accuracy = eval_accuracy(output, y)
    
    G_loss = 1.0
    G_output = backprop_postproc(G_loss, aux_pp)
    backprop_neuralnet(G_output, aux_nn)
    
    return loss, accuracy

def run_test(x, y):
    output, _ = forward_neuralnet(x)
    accuracy = eval_accuracy(output, y)
    return accuracy


def forward_neuralnet(x):
    #implement forward processing

def backprop_neuralnet(G_output, x):
    #implement backward processing


def forward_postproc(output, y):
    #implement forward processing by using MSE
    return loss, diff
def backprop_postproc(G_loss, diff):
    #implement backward processing by using MSE
    return G_output

def eval_accuracy(output, y):
    mdiff = np.mean(np.abs((output - y)/y))
    return 1 - mdiff


def backprop_postproc_oneline(G_loss, diff):  # alternative method of backprop_postproc()
    return 2 * diff / np.prod(diff.shape)

abalone_exec(epoch_count = 1000, mb_size = 50, report = 20)