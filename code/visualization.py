from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    ConfusionMatrixDisplay,
)


def get_precision_recall_display(summary, plot=True):
    disp = PrecisionRecallDisplay(
        precision=summary["precision_recall_curve"][0],
        recall=summary["precision_recall_curve"][1],
        average_precision=summary["average_precision_score"],
    )
    if plot:
        disp.plot()
    return disp


def get_roc_curve_display(summary, plot=True):
    disp = RocCurveDisplay(
        fpr=summary["roc_curve"][0],
        tpr=summary["roc_curve"][1],
        roc_auc=summary["roc_auc_score"],
    )
    if plot:
        disp.plot()
    return disp


def get_confusion_matrix_display(summary, plot=True):
    disp = ConfusionMatrixDisplay(summary["confusion_matrix"])
    if plot:
        disp.plot()
    return disp


def get_all_displays(summary, plot=True):
    return (
        get_precision_recall_display(summary),
        get_roc_curve_display(summary),
        get_confusion_matrix_display(summary),
    )
