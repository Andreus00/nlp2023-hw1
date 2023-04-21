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