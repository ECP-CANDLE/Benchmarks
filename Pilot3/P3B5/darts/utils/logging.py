from loguru import logger


logger.add("darts_p3b3.log", format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}", level="INFO")


def log_accuracy(accuracy, split: str='train'):
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