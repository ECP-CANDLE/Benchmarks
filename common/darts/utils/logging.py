import logging


logger = logging.getLogger('DARTS')
fh = logging.FileHandler('darts_accuracy.log')
logger.addHandler(fh)


def log_accuracy(accuracy, split: str = 'train'):
    """ Log the average accuracy

    Parameters
    ----------
    accuracy: darts.MultitaskAccuracyMeter
        Current accuracy meter state

    split: str
        Either training of testing
    """
    acc_info = (
        f">>> {split.upper()} Accuracy - Subsite: {accuracy.get_avg_accuracy('subsite'):.4f}, "
        f"Laterality: {accuracy.get_avg_accuracy('laterality'):.4f}, "
        f"Behavior: {accuracy.get_avg_accuracy('behavior'):.4f}, "
        f"Grade: {accuracy.get_avg_accuracy('grade'):.4f}"
    )

    logger.info(acc_info)


def log_single_accuracy(accuracy, split: str = 'train'):
    """ Log the average accuracy for a single task

    Parameters
    ----------
    accuracy: darts.MultitaskAccuracyMeter
        Current accuracy meter state

    split: str
        Either training of testing
    """
    acc_info = (
        f">>> {split.upper()} Accuracy - Response: {accuracy.get_avg_accuracy('response'):.4f}, "
    )

    logger.info(acc_info)
