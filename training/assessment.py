"""Classification diagnostics and empirical targets, never deployment approval."""
import math
import numpy as np


def wilson_interval(successes, trials, z=1.959963984540054):
    """Two-sided 95% binomial interval; assumes independent Bernoulli trials."""
    if trials == 0:
        return None
    p = successes / trials
    denominator = 1 + z*z/trials
    center = (p + z*z/(2*trials))/denominator
    radius = z*math.sqrt(p*(1-p)/trials + z*z/(4*trials*trials))/denominator
    return [max(0.,center-radius),min(1.,center+radius)]


def additional_metrics(labels, probabilities, tp, fp, tn, fn):
    n = len(labels)
    p = np.clip(np.asarray(probabilities,dtype=np.float64),1e-15,1-1e-15)
    denominator = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    bins = np.minimum((np.asarray(probabilities)*10).astype(int),9)
    reliability = []
    for index in range(10):
        mask = bins == index
        if mask.any():
            reliability.append(dict(bin=index,count=int(mask.sum()),
                                    mean_score=float(np.mean(probabilities[mask])),
                                    attack_fraction=float(np.mean(labels[mask]))))
    return dict(accuracy=(tp+tn)/n,attack_prevalence=(tp+fn)/n,
        confusion_matrix=[[tn,fp],[fn,tp]],confusion_matrix_labels=['benign','attack'],
        false_negative_rate=fn/(tp+fn) if tp+fn else None,
        specificity=tn/(tn+fp) if tn+fp else None,
        balanced_accuracy=.5*(tp/(tp+fn)+tn/(tn+fp)) if tp+fn and tn+fp else None,
        f2=5*tp/max(5*tp+4*fn+fp,1),mcc=(tp*tn-fp*fn)/denominator if denominator else 0.,
        log_loss=float(-np.mean(labels*np.log(p)+(1-labels)*np.log1p(-p))),
        precision_defined=bool(tp+fp),recall_defined=bool(tp+fn),fpr_defined=bool(tn+fp),
        intervals_95_iid=dict(precision=wilson_interval(tp,tp+fp),recall=wilson_interval(tp,tp+fn),
                              false_positive_rate=wilson_interval(fp,fp+tn)),
        interval_assumption='IID binomial illustration only; correlated flows can understate uncertainty. Use independent capture evidence.',
        ece_10_bins=sum(r['count']/n*abs(r['mean_score']-r['attack_fraction']) for r in reliability),
        reliability_bins=reliability)


def target_check(report, min_precision, min_recall, max_fpr):
    failures = []
    if report['tp']+report['fn']:
        if not report['tp']+report['fp'] or report['precision'] < min_precision:
            failures.append('precision')
        if report['recall'] < min_recall:
            failures.append('recall')
    if report['tn']+report['fp'] and report['false_positive_rate'] > max_fpr:
        failures.append('false_positive_rate')
    return dict(empirical_targets_met=not failures,failed_metrics=failures)


def assess_protocol(report, min_precision, min_recall, max_fpr):
    aggregate = target_check(report,min_precision,min_recall,max_fpr)
    captures = {g:target_check(r,min_precision,min_recall,max_fpr) for g,r in report['per_capture'].items()}
    return dict(aggregate=aggregate,per_capture=captures,
        empirical_targets_met=aggregate['empirical_targets_met'] and all(r['empirical_targets_met'] for r in captures.values()),
        note='Checks observed binary metrics only. Not statistical certification or approval for automatic blocking.')
