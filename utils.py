import json

def upper(vt, v):
    l, r = 0, len(vt)
    while l < r:
        m = (l+r)//2
        if vt[m] <= v:
            l = m+1
        else:
            r = m
    return r

def lower(vt, v):
    l, r = 0, len(vt)
    while l < r:
        m = (l+r)//2
        if vt[m] < v:
            l = m+1
        else:
            r = m
    return r

def chatterjee(X, Y, rand):
    samples = list(zip(X,Y))
    n = len(samples)

    tmp = [(v[0], rand.random(), v[1]) for v in samples] #needed to break ties uniformly at random
    tmp = sorted(tmp)
    vals = [(v[0], v[2]) for v in tmp]

    for i in range(n-1):
        assert vals[i][0] <= vals[i+1][0], f'{vals[i][0]} {vals[i+1][0]}'
    
    v2 = sorted([v[1] for v in vals])
    r = [upper(v2, vals[i][1]) for i in range(n)]
    l = [n - lower(v2, vals[i][1]) for i in range(n)]
    
    num = 0
    den = 0
    for i in range(n):
        assert l[i] >= 1 and r[i] >= 1
        if i < n-1:
            num += abs(r[i+1] - r[i])
        den += l[i] * (n - l[i])  
    if den < 0.00001:
        assert num < 0.00001 #all values are the same
        return 1
    chatterjee = 1 - n * num / (2 * den)
    return chatterjee

def d_sym_max(X,Y, rand):
    return 1 - max(chatterjee(X, Y, rand), chatterjee(Y, X, rand))

def d_sym_max(chatterjee_XY, chatterjee_YX):
    return 1 - max(chatterjee_XY, chatterjee_YX)

def d_sym_min(chatterjee_XY, chatterjee_YX):
    return 1 - min(chatterjee_XY, chatterjee_YX)

def d_sym_avg(chatterjee_XY, chatterjee_YX):
    return 1 - (chatterjee_XY + chatterjee_YX)/2

def d_asym(chatterjee):
    return 1 - chatterjee

def dict_str_format(d):
    return {str(k): v for (k,v) in d.items()}

def dict_to_tuple_format(d):
    return {tuple(map(int, k[1:-1].split(', '))):v for (k, v) in d.items()}

def get_available_uci_datasets():
    #get datasets available through python library
    with open('data/uci_python_list.json', 'r') as f:
        l = json.load(f)
    return l

def get_uci_datasets_id():
    # all datasets we use in our paper
    return [1, 9, 10, 16, 17, 27, 50, 52, 53, 60, 78, 94, 107, 109, 110, 143, 145, 147, 151, 155, 159, 162, 165, 174, 183, 186, 189, 193, 198, 211, 212, 225, 242, 247, 257, 264, 267, 275, 291, 294, 329, 332, 342, 360, 372, 373, 374, 390, 396, 409, 451, 464, 468, 471, 477, 484, 492, 536, 544, 545, 551, 560, 571, 572, 601, 602, 697, 755, 763, 799, 848, 849, 850, 851, 864, 880, 887, 913, 925, 967]

def get_uci_datasets_id_more_than_median():
    # datasets with #features > median
    return [10, 16, 17, 50, 52, 94, 107, 109, 110, 147, 151, 159, 174, 183, 186, 189, 193, 198, 211, 247, 264, 329, 332, 342, 372, 373, 374, 396, 464, 471, 484, 551, 572, 602, 799, 864, 913, 925]