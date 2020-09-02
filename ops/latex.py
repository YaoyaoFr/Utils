'''
Author: your name
Date: 2019-12-23 09:22:44
LastEditTime: 2020-08-25 15:04:55
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /DeepLearning2.0/home/ai/data/yaoyao/Program/Python/Utils/ops/latex.py
'''

import numpy as np
import collections


def matrix(A: np.ndarray,
           dtype: str = '',
           name: str = ''):
    if not name == '':
        name += '='
    latex_matrix_prefix = '{:s}\\left[ \r\n' \
                          '\\begin{matrix} \r\n'.format(name)
    latex_matrix_postfix = '\\end{matrix} \r\n' \
                           '\\right]'
    latex_matrix_content = ''
    for row_index, row in enumerate(A):
        for col_index, col in enumerate(row):
            if col == 0:
                final_format = 'd'
            else:
                final_format = dtype
            latex_matrix_content += ('{:' +
                                     '{:s}'.format(final_format) + '}').format(col)
            if col_index < np.size(A, 1) - 1:
                latex_matrix_content += ' & '
        if row_index < np.size(A, 0) - 1:
            latex_matrix_content += ' \\\\ \r\n'
        else:
            latex_matrix_content += '\r\n'
    return latex_matrix_prefix + latex_matrix_content + latex_matrix_postfix


def matmul(A: np.ndarray,
           B: np.ndarray,
           sum_items: list = None,
           printf: bool = True,
           ):
    shape1 = np.shape(A)
    shape2 = np.shape(B)
    assert shape1[1] == shape2[0], 'Dimension must be equal but {:d} and {:d}.'.format(
        shape1[1], shape2[0])

    if sum_items is None:
        sum_items = [[[] for _ in range(shape2[1])] for _ in range(shape1[0])]
    for i in range(shape1[0]):
        for j in range(shape2[1]):
            for p in range(shape1[1]):
                if A[i, p] and B[p, j]:
                    A_str = '{:d}{:d}'.format(i + 1, p + 1)
                    B_str = '{:d}{:d}'.format(p + 1, j + 1)
                    sum_items[i][j].append(
                        'A_{' + A_str + '} * B_{' + B_str + '}')

    if printf:
        print(matrix(sum_items, dtype='s'))

    return sum_items


def graph_matmul(graph: np.ndarray,
                 adj_matrix: np.ndarray,
                 sum_items: list = None,
                 printf: bool = True,
                 ):
    v_num = np.size(graph, 0)
    assert (v_num, v_num) == np.shape(graph)
    assert (v_num, v_num, v_num) == np.shape(adj_matrix)

    if sum_items is None:
        sum_items = [['' for _ in range(v_num)] for _ in range(v_num)]

    new_sum_items = [[[] for _ in range(v_num)] for _ in range(v_num)]

    for i in range(v_num):
        for j in range(v_num):
            for p in range(v_num):
                if graph[i, p] and adj_matrix[p, j, i]:
                    if sum_items[i][p] != '':
                        if '+' in sum_items[i][p]:
                            format_str = '({:s})'
                        else:
                            format_str = '{:s}'
                        A_str = format_str.format(sum_items[i][p])
                    else:
                        A_str = 'A_{' + '{:d}{:d}'.format(i, p) + '}'

                    B_str = 'A_{' + '{:d}{:d}'.format(p, j) + '}'
                    new_sum_items[i][j].append(
                        '{:s} * {:s}'.format(A_str, B_str))
            if len(new_sum_items[i][j]) > 0:
                new_sum_items[i][j] = ' + '.join(new_sum_items[i][j])
            else:
                new_sum_items[i][j] = 0

    graph_latex = (matrix(new_sum_items, dtype='s'))

    return new_sum_items, graph_latex


def table(contents: list, title: str = '', cols: int = 0, label: str = ''):
    length_c = ''.join(['c' for _ in range(cols)])
    begin_tex = '\\renewcommand\\arraystretch{{2}} \n' +\
        '\\newcommand{\\tabincell}[2]{\\begin{tabular}{@{}#1@{}}#2\\end{tabular}} \n' +\
        '\\begin{table*}[htbp] \n' +\
        '\\centering \n' +\
        '\\setlength{\\abovecaptionskip}{0.cm} \n' +\
        '\\setlength{\\belowcaptionskip}{0.4cm} \n' +\
        '\\caption{{{:s}}} \n'.format(title) +\
        '\\begin{{tabular}}{{{:s}}} \n'.format(length_c)

    content_tex = ''
    for line in contents:
        max_row_num = np.max([len(cell) for cell in line])

        line_tex = ''
        for row in range(max_row_num):
            row_tex = ''

            for index, cell in enumerate(line):
                length = len(cell)

                try:
                    cell_content = str(cell[row])
                    if length < max_row_num:
                        multi_row = int(max_row_num / length)
                        cell_content = '\\multirow{{{:d}}}{{*}}{{\\tabincell{{c}}{{{:s}}}}}'.format(
                            multi_row, cell_content)
                except:
                    cell_content = ''
                row_tex += cell_content

                if index == len(line) - 1:
                    row_tex += '\t\\\\ \n'
                else:
                    row_tex += '&\t'
            line_tex += row_tex 
        line_tex += '\\hline \n'

        content_tex += line_tex

    end_tex = '\\end{{tabular}}\\label{{{:s}}} \n'.format(label) +\
        '\\end{table*}'

    # return begin_tex + content_tex + end_tex
    return content_tex


if __name__ == "__main__":
    pass