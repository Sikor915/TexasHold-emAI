
def normalize(narray):
    return [x/sum(x) for x in narray]

def get_def_probability():
    return [
        [0.7, 0.2, 0.0],
        [0.6, 0.3, 0.1],
        [0.1, 0.8, 0.1],
        [0.0, 0.5, 0.5],
        [0.0, 0.2, 0.8]
    ]

def add_lists(list_of_lists):
    result = []
    for i in range(len(list_of_lists[0])):
        result.append(sum([lst[i] for lst in list_of_lists]))
    return result