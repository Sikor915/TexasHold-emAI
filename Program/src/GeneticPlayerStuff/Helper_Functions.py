
def normalize(narray):
    return [x/sum(x) for x in narray]

def get_def_probability():
    return [
        [0.8, 0.2, 0.0],
        [0.6, 0.3, 0.1],
        [0.1, 0.8, 0.1],
        [0.0, 0.5, 0.5],
        [0.0, 0.2, 0.8]
    ]
# Based on player 1 from 16-hour training session (choice=1)
# Based on player 6 from 19-hour training session (choice=2)
def get_best_probability(choice=1):
    match(choice):
        case 1:
            return [
                [ 0.7124873590059478, 0.28751264099405205, 0.0 ],
                [ 0.6119830080550757, 0.2803372624344745, 0.10767972951044984 ],
                [ 0.07213065168987474, 0.6388192008654975, 0.28905014744462787 ],
                [ 0.0, 0.7885663563926363, 0.21143364360736366 ],
                [ 0.0, 0.6549456079982731, 0.34505439200172683 ]
            ]
        case 2:
            return [
                [ 0.9244754019783054, 0.07552459802169476, 0.0 ],
                [ 0.5336575816487088, 0.29377268459812783, 0.1725697337531633 ],
                [ 0.12781208210465617, 0.6830414660218055, 0.1891464518735384 ],
                [ 0.0, 0.354346608453182, 0.645653391546818 ],
                [ 0.0, 0.4788500658304157, 0.5211499341695843 ]
            ]

def get_best_aggresion(choice=1):
    match(choice):
        case 1:
            return 0.736942979615231
        case 2:
            return 0.3279154041436131

def add_lists(list_of_lists):
    result = []
    for i in range(len(list_of_lists[0])):
        result.append(sum([lst[i] for lst in list_of_lists]))
    return result