import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def plot_history(out, history, metric='loss', title=None, width=8, height=6):
    title = title or 'model {}'.format(metric)
    val_metric = 'val_{}'.format(metric)
    plt.figure(figsize=(width, height))
    plt.plot(history.history[metric], marker='o')
    plt.plot(history.history[val_metric], marker='d')
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(['train_{}'.format(metric), 'val_{}'.format(metric)], loc='upper center')
    png = '{}.plot.{}.png'.format(out, metric)
    plt.savefig(png, bbox_inches='tight')

def plot_scatter(data, classes, out, width=10, height=8):
    cmap = plt.cm.get_cmap('gist_rainbow')
    plt.figure(figsize=(width, height))
    plt.scatter(data[:, 0], data[:, 1], c=classes, cmap=cmap, lw=0.5, edgecolor='black', alpha=0.7)
    plt.colorbar()
    png = '{}.png'.format(out)
    plt.savefig(png, bbox_inches='tight')

def plot_error(y_true, y_pred, batch, file_ext, file_pre='output_dir', subsample=1000):
    if batch % 10:
        return

    total = len(y_true)
    if subsample and subsample < total:
        usecols = np.random.choice(total, size=subsample, replace=False)
        y_true = y_true[usecols]
        y_pred = y_pred[usecols]

    y_true = y_true * 100
    y_pred = y_pred * 100
    diffs = y_pred - y_true

    bins = np.linspace(-200, 200, 100)
    if batch == 0:
        y_shuf = np.random.permutation(y_true)
        plt.hist(y_shuf - y_true, bins, alpha=0.5, label='Random')

    #plt.hist(diffs, bins, alpha=0.35-batch/100., label='Epoch {}'.format(batch+1))
    plt.hist(diffs, bins, alpha=0.3, label='Epoch {}'.format(batch+1))
    plt.title("Histogram of errors in percentage growth")
    plt.legend(loc='upper right')
    plt.savefig(file_pre+'.histogram'+file_ext+'.b'+str(batch)+'.png')
    plt.close()

    # Plot measured vs. predicted values
    fig, ax = plt.subplots()
    plt.grid('on')
    ax.scatter(y_true, y_pred, color='red', s=10)
    ax.plot([y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.savefig(file_pre+'.diff'+file_ext+'.b'+str(batch)+'.png')
    plt.close()

