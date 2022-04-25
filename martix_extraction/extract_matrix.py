import numpy as np


def read_file(path, ncols=100, nrows=100):
    conn = open(path, 'r')
    text = conn.readlines()
    conn.close()
    start_line = None
    stop_line = None
    for i, line in enumerate(text):
        if 'RDM1: ' in line:
            start_line = i + 2
        if '1RDM Norm' in line:
            stop_line = i - 2 + 1  # + 1 because of the range
    if start_line is None or stop_line is None:
        raise Exception("I didn't find words 'RDM1: ' and '1RDM Norm' in the file")
    current_line = start_line
    export_table = []
    while current_line < stop_line:
        data_line = []
        # print(current_line)
        while len(data_line) < ncols:
            # print(current_line, len(data_line), ncols)
            float_line = [float(i) for i in text[current_line].strip().split(' ') if i != '']
            data_line.extend(float_line)
            current_line += 1
        if len(data_line) != ncols:
            raise Exception(f'Problem with number of features in the line: {len(data_line)} != {ncols}')
        export_table.append(data_line)
    if len(export_table) != nrows:
        raise Exception(f'Problem with number of lines: {len(export_table)} != {nrows}')

    return np.array(export_table)


if __name__ == '__main__':
    a = read_file('example.out')
    print( a.shape)
    aaaa = 1
