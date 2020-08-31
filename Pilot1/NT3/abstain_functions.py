import keras
from keras import backend as K

abs_definitions = [
    {'name': 'add_class',
     'nargs': '+',
     'type': int,
     'help': 'flag to add abstention (per task)'},
    {'name': 'alpha',
     'nargs': '+',
     'type': float,
     'help': 'abstention penalty coefficient (per task)'},
    {'name': 'min_acc',
     'nargs': '+',
     'type': float,
     'help': 'minimum accuracy required (per task)'},
    {'name': 'max_abs',
     'nargs': '+',
     'type': float,
     'help': 'maximum abstention fraction allowed (per task)'},
    {'name': 'alpha_scale_factor',
     'nargs': '+',
     'type': float,
     'help': 'scaling factor for modifying alpha (per task)'},
    {'name': 'init_abs_epoch',
     'action': 'store',
     'type': int,
     'help': 'number of epochs to skip before modifying alpha'},
    {'name': 'n_iters',
     'action': 'store',
     'type': int,
     'help': 'number of iterations to iterate alpha'},
    {'name': 'acc_gain',
     'type': float,
     'default': 5.0,
     'help': 'factor to weight accuracy when determining new alpha scale'},
    {'name': 'abs_gain',
     'type': float,
     'default': 1.0,
     'help': 'factor to weight abstention fraction when determining new alpha scale'},
    {'name': 'task_list',
     'nargs': '+',
     'type': int,
     'help': 'list of task indices to use'},
    {'name': 'task_names',
     'nargs': '+',
     'type': int,
     'help': 'list of names corresponding to each task to use'},
]

def adjust_alpha(gParameters, X_test, truths_test, labels_val, model, alpha, add_index):

    task_names = gParameters['task_names']
    task_list = gParameters['task_list']
    # retrieve truth-pred pair
    avg_loss = 0.0
    ret = []
    ret_k = []

    # set abstaining classifier parameters
    max_abs = gParameters['max_abs']
    min_acc = gParameters['min_acc']
    alpha_scale_factor = gParameters['alpha_scale_factor']

    #print('labels_test', labels_test)
    #print('Add_index', add_index)

    feature_test = X_test
    label_test = keras.utils.to_categorical(truths_test)

    #loss = model.evaluate(feature_test, [label_test[0], label_test[1],label_test[2], label_test[3]])
    loss = model.evaluate(feature_test, labels_val)
    avg_loss = avg_loss + loss[0]

    pred = model.predict(feature_test)
    #print('pred',pred.shape, pred)

    abs_gain = gParameters['abs_gain']
    acc_gain = gParameters['acc_gain']

    accs = []
    abst = []

    for k in range((alpha.shape[0])):
        if k in task_list:
            truth_test = truths_test[:, k]
            alpha_k = K.eval(alpha[k])
            pred_classes = pred[k].argmax(axis=-1)
            #true_classes = labels_test[k].argmax(axis=-1)
            true_classes = truth_test

            #print('pred_classes',pred_classes.shape, pred_classes)
            #print('true_classes',true_classes.shape, true_classes)
            #print('labels',label_test.shape, label_test)

            true = K.eval(K.sum(K.cast(K.equal(pred_classes, true_classes), 'int64')))
            false = K.eval(K.sum(K.cast(K.not_equal(pred_classes, true_classes), 'int64')))
            abstain = K.eval(K.sum(K.cast(K.equal(pred_classes, add_index[k] - 1), 'int64')))

            print(true, false, abstain)

            total = false + true
            tot_pred = total - abstain
            abs_acc = 0.0
            abs_frac = abstain / total

            if tot_pred > 0:
                abs_acc = true / tot_pred

            scale_k = alpha_scale_factor[k]
            min_scale = scale_k
            max_scale = 1. / scale_k

            acc_error = abs_acc - min_acc[k]
            acc_error = min(acc_error, 0.0)
            abs_error = abs_frac - max_abs[k]
            abs_error = max(abs_error, 0.0)
            new_scale = 1.0 + acc_gain * acc_error + abs_gain * abs_error

            # threshold to avoid huge swings
            new_scale = min(new_scale, max_scale)
            new_scale = max(new_scale, min_scale)

            print('Scaling factor: ', new_scale)
            K.set_value(alpha[k], new_scale * alpha_k)

            print_abs_stats(task_names[k], new_scale*alpha_k, true, false, abstain, max_abs[k])

            ret_k.append(truth_test)
            ret_k.append(pred)

            ret.append(ret_k)

            accs.append(abs_acc)
            abst.append(abs_frac)
        else:
            accs.append(1.0)
            accs.append(0.0)

    write_abs_stats(gParameters['output_dir']+'abs_stats.csv', alpha, accs, abst)

    return ret, alpha

def loss_param(alpha, mask):
    def loss(y_true, y_pred):

        cost = 0

        base_pred = (1 - mask) * y_pred
        #base_true = (1 - mask) * y_true
        base_true = y_true

        base_cost = K.sparse_categorical_crossentropy(base_true, base_pred)

        abs_pred = K.mean(mask * (y_pred), axis=-1)
        # add some small value to prevent NaN when prediction is abstained
        abs_pred = K.clip(abs_pred, K.epsilon(), 1. - K.epsilon())
        cost = (1. - abs_pred) * base_cost - (alpha) * K.log(1. - abs_pred)

        return cost
    return loss


def print_abs_stats(
        task_name,
        alpha,
        num_true,
        num_false,
        num_abstain,
        max_abs):

    # Compute interesting values
    total = num_true + num_false
    tot_pred = total - num_abstain
    abs_frac = num_abstain / total
    abs_acc = 1.0
    if tot_pred > 0:
        abs_acc = num_true / tot_pred

    print('        task,       alpha,     true,    false,  abstain,    total, tot_pred,   abs_frac,    max_abs,    abs_acc')
    print('{:>12s}, {:10.5e}, {:8d}, {:8d}, {:8d}, {:8d}, {:8d}, {:10.5f}, {:10.5f}, {:10.5f}'
          .format(task_name, alpha,
                  num_true, num_false - num_abstain, num_abstain, total,
                  tot_pred, abs_frac, max_abs, abs_acc))


def write_abs_stats(stats_file, alphas, accs, abst):

    # Open file for appending
    abs_file = open(stats_file, 'a')

    # we write all the results
    for k in range((alphas.shape[0])):
        abs_file.write("%10.5e," % K.get_value(alphas[k]))
    for k in range((alphas.shape[0])):
        abs_file.write("%10.5e," % accs[k])
    for k in range((alphas.shape[0])):
        abs_file.write("%10.5e," % abst[k])
    abs_file.write("\n")
