import math
import EWMA
import numpy

def smooth(signal, span):
    final = []

    l = len(signal)
    neighbors = span // 2
    print(neighbors)

    for i in range(l):
        sum = 0
        if (i < neighbors):
            sp = i

        elif ((neighbors+i) >  l-1):
            sp = (l-1)-i

        else:
            sp = neighbors

        for j in range((i-sp), (i+sp+1)):
            sum+=signal[j]

        final.append(sum/((sp*2)+1))

    return final

def process_function(signalin, t_0, t_1, t_2, samplingfreq):

    F_0 = []

    Fs = samplingfreq

    t_0_s = math.floor(t_0 * Fs)
    t_1_s = math.floor(t_1 * Fs)
    t_2_s = math.floor(t_2 * Fs)

    F_sm = smooth(signalin, t_1_s)

    for i in range((t_2_s), len(signalin)):
        F_0.append(min(F_sm[i-t_2_s:i]))

    R_0 = numpy.division((F_sm[t_2_s:] - numpy.transpose(F_0)), numpy.transpose(F_0))

    R_0_sm = EWMA.smoothed_z(R_0, t_0_s)

    return R_0_sm

F = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
G = smooth(F, 3)
print(G)