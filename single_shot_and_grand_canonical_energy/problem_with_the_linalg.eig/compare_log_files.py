"""
This file tries to make eigenvectors from fortran and numpy log files more similar
"""
import numpy as np
from single_shot import print_matrix

def read_file(path1):
    with open(path1) as conn:
        text = conn.readlines()
        text = [i.strip() for i in text]
        return text

def correct_text_python(text):
    start_ei = 0
    in_table = False
    end_ei = 0
    ei_vec = []
    for i, line in enumerate(text):
        if in_table:
            if line == '':
                end_ei = i
                in_table = False
                continue
            splitted = [float(i) for i in line.split(' ') if i!='']
            ei_vec.append(splitted)

        if "Eigenvectors" in line:
            start_ei = i + 1
            in_table = True
    return (start_ei, end_ei), np.array(ei_vec)

if __name__ == '__main__':
    text_py = read_file('Procedure_log-U-8_N-10_reference.txt')
    text_fr = read_file('Procedure_log-U-8_N-10_Ne-18_Sajan.txt')
    result_py = correct_text_python(text_py)
    eivec_py = result_py[1]
    result_fr = correct_text_python(text_fr)
    eivec_fr = result_fr[1]

    print("Python")
    print_matrix(eivec_py, 'python before')
    print("Fortran")
    print_matrix(eivec_fr, 'Fortran before')

    for i in range(eivec_fr.shape[1]):
        eivec_py[:,i] = np.sign(eivec_py[0, i]) * eivec_py[:,i]
        eivec_fr[:, i] = np.sign(eivec_fr[0, i]) * eivec_fr[:, i]

    for i in range(eivec_fr.shape[1]-1):
        delta1 = np.sum(np.square(eivec_py[:, i] - eivec_fr[:, i])) + np.sum(np.square(eivec_py[:, i+1] - eivec_fr[:, i+1]))
        delta2 = np.sum(np.square(eivec_py[:, i] - eivec_fr[:, i+1])) + np.sum(np.square(eivec_py[:, i+1] - eivec_fr[:, i]))
        if delta1 > delta2:
            temp1 = eivec_py[:, i].copy()
            eivec_py[:, i] = eivec_py[:, i + 1]
            eivec_py[:, i + 1] = temp1
            print(f"Chaging {i} and {i+1}")
    print("Python after")
    print_matrix(eivec_py, 'Python after')
    print("Fortran after")
    print_matrix(eivec_fr, 'Fortran after')
    print("Difference")
    print_matrix(eivec_py - eivec_fr, False)
