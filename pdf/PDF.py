import re
import os
import xlwt
import PyPDF2
from PyPDF2 import PdfFileReader


class PDF:
    annotation_keywords = ['title', 'keywords', 'authors', 'author', 'type', 'host', 'Source', 'year', 'volume', 'page',
                           'ref', 'disadv', 'adv', 'arg', 'que', 'rem', 'ins', 'iden', 'label', 'com', 'trans', 'mot',
                           ]

    def __init__(self, file_path):
        self.annotation_dict = {name: [] for name in self.annotation_keywords}
        self.annotations = []
        self.flag = False
        try:
            file = PdfFileReader(open(file_path, 'rb'))
            page_num = file.getNumPages()
        except Exception:
            print('File {:s} open failed.'.format(file_path))
            return
        for page_idx in range(page_num):
            try:
                current_page = file.getPage(page_idx)
                if '/Annots' not in current_page:
                    continue
                annots = [annot.getObject() for annot in current_page['/Annots']]
            except Exception:
                continue
            for annot in annots:
                try:
                    if '/Contents' not in annot:
                        continue
                    annotation = str(annot['/Contents'])
                    self.annotations.append(annotation)
                except Exception:
                    continue

    def analyse_annotations(self):
        for annotation in self.annotations:
            for annotation_line in annotation.split('\\'):
                regex_keyword = r'#.+#'
                pattern = re.compile(regex_keyword)
                keyword = pattern.findall(annotation_line)
                if len(keyword) == 1:
                    keyword = keyword[0][1:-1]
                else:
                    continue

                if keyword in self.annotation_dict:
                    regex_contents = r'<[^<]+>'
                    pattern = re.compile(regex_contents)
                    contents = pattern.findall(annotation_line)

                    contents = [content[1:-1] for content in contents]

                    self.flag = True
                    self.annotation_dict[keyword].append(contents)


def write_excel(pdf_files, directory_path, keywords: list = None):
    annotation_keywords = {'title': 'Title',
                           'keywords': 'Keywords',
                           'authors': 'Author',
                           'author': 'Author',
                           'type': 'Document class',
                           'host': 'Host',
                           'Source': 'Document source',
                           'year': 'Year',
                           'volume': 'Volume',
                           'page': 'Pages',
                           'ref': 'Reference',
                           'disadv': 'Disadvantage',
                           'adv': 'Advantage',
                           'arg': 'Arguments',
                           'que': 'Question',
                           'rem': 'Remaining',
                           'ins': 'Inspire',
                           'iden': 'Identification',
                           'label': 'Labeling of paper',
                           'com': 'Comments of paragraph',
                           'trans': 'Translation',
                           'mot': 'Motivation',
                           }

    if keywords is None:
        keywords = annotation_keywords

    annotation_dict = {name: [] for name in keywords}
    for pdf_file in pdf_files:
        for keyword in keywords:
            annotation_dict[keyword].extend(pdf_file.annotation_dict[keyword])

    excel_file = xlwt.Workbook()
    for sheet_name in keywords:
        sheet = excel_file.add_sheet(sheet_name, cell_overwrite_ok=True)
        for row, annotation_line in enumerate(annotation_dict[sheet_name]):
            for column, annotation in enumerate(annotation_line):
                sheet.write(row, column, annotation)

    save_path = os.path.join(directory_path, 'annotation.xls')
    excel_file.save(save_path)
    return annotation_dict
