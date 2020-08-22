from pdf.PDF import PDF, write_excel
from pdf.files import get_file_paths_within_directory

if __name__ == '__main__':
    directory_path = 'F:\OneDrive - emails.bjut.edu.cn\Paper'
    file_paths = get_file_paths_within_directory(directory_path=directory_path, post_fix='.pdf')

    pdf_files = []
    for idx, file_path in enumerate(file_paths):
        print('Analyse {:d}/{:d}'.format(idx+1, len(file_paths)))
        pdf_file = PDF(file_path=file_path)
        pdf_file.analyse_annotations()
        if pdf_file.flag:
            pdf_files.append(pdf_file)

    write_excel(pdf_files, directory_path=directory_path)


