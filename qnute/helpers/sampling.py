import numpy as np

#----------#
# Sampling #
#----------#
def sample_from_a(a):
    '''
    Given a vector of real numbers a,
    returns a vector where all elements are 0 except for a random
    index i with probability:
        P(i) = |a[i]|/sum(|a|)
    and that index has the original a[i] in the output
    '''

    cdf = np.abs(a.copy())  # load |a|
    cdf = np.cumsum(cdf)    # Convert to CDF
    
    # Generate a random number
    y = np.random.uniform(0.0,cdf[-1])

    # Flag to see if the index i was found
    found_flag = False 

    for i in range(len(cdf)):
        if y <= cdf[i] and not found_flag:
            found_flag = True
            cdf[i] = a[i]
        else:
            cdf[i] = 0.0
    
    return cdf

def generate_random_complex_vector(n):
    amps = np.random.normal(0.0, 1.0, n)
    phases = np.random.uniform(0.0,np.pi, n)*1j
    amps /= np.linalg.norm(amps)
    return amps * np.exp(phases)
