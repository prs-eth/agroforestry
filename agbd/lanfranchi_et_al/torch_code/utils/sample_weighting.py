# inspired from https://medium.com/gumgum-tech/handling-class-imbalance-by-introducing-sample-weighting-in-the-loss-function-3bdebd8203b4
import numpy as np

def get_weights_effective_num_of_samples(beta, frequencies):
    effective_nums = list(map(lambda x: (1-beta)/(1-beta**x), frequencies))
    normalization_cst = np.sum(effective_nums)
    return effective_nums/normalization_cst

def get_weights_inverse_num_of_samples(frequencies, power=1):
    inverse_freq = list(map(lambda x: 1/x**power if x>0 else 0, frequencies))
    normalization_cst = np.sum(inverse_freq)
    return inverse_freq/normalization_cst

def get_weights_for_sample(sample_weighting_method, frequencies, use_nb_of_classes, beta):
    """
    This function applies the given Sample Weighting Scheme and returns the sample weights normalized over the training set.
    Args:
        sample_weighting_methode: str, options available: 'ens', 'ins', 'isns'
        classes: array of size [no_of_bins] that indicates how to bin the agbd values
        frequencies: array of size [no_of_bins] that stores the number of samples for each bin
        beta: float
    
    Returns:
        weights_for_samples: array of size [no_of_bins]     
    """
    if sample_weighting_method == 'ens':
        weights_for_samples = get_weights_effective_num_of_samples(beta, frequencies)
    elif sample_weighting_method == 'ins':
        weights_for_samples = get_weights_inverse_num_of_samples(frequencies)
    elif sample_weighting_method == 'isns': # squared root
        weights_for_samples = get_weights_inverse_num_of_samples(frequencies, 0.5)
    elif sample_weighting_method == 'ifns': # fourth root
        weights_for_samples = get_weights_inverse_num_of_samples(frequencies, 0.25)
    else:
        raise NotImplementedError(f'unknown sample weighting method {sample_weighting_method}')
    
    #classes = np.digitize(labels, bins)-1 # subtract 1 because bins start at 1
    #weights_for_samples = weights_for_samples[classes]
    if use_nb_of_classes:
        return weights_for_samples*len(frequencies)
    else:
        return weights_for_samples