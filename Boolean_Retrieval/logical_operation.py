def AND(all_list):
    return [all(column) for column in zip(*all_list)]

def OR(all_list):
    return [any(column) for column in zip(*all_list)]
    
def NOT(single_list):
    return [not x for x in single_list]
