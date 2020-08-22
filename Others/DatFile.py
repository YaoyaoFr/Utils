import numpy as np
import pandas as pd
def load_dat_file(file_path: str,
                  sep: str = '\t',
                  header: str = None,
                  names: str = None,
                  engine: str = 'python',
                  ):
    data = pd.read_table(file_path, sep=sep, header=header, engine=engine)
    return data

if __name__ == '__main__':
    file_path = 'F:\OneDrive\Program\LaTeX\pgfplots_1.16.tds\doc\latex\pgfplots\pgfplots.doc.src\plotdata\pgfplots_scatterdata4.dat'
    data = load_dat_file(file_path)