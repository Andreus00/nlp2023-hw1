'''
Classes for the task
'''

class2int = {
    "B-ACTION": 0,
    "B-CHANGE": 1,
    "B-POSSESSION": 2,
    "B-SCENARIO": 3,
    "B-SENTIMENT": 4,
    "I-ACTION": 5,
    "I-CHANGE": 6,
    "I-POSSESSION": 7,
    "I-SCENARIO": 8,
    "I-SENTIMENT": 9,
    "O": 10,
}

int2class = {v: k for k, v in class2int.items()}

pos2int = {
    "ADJ": 0,
    "ADP": 1,
    "ADV": 2,
    "CONJ": 3,
    "DET": 4,
    "NOUN": 5,
    "NUM": 6,
    "PRT": 7,
    "PRON": 8,
    "VERB": 9,
    ".": 10,
    "X": 11,
    "PAD": 12,
}