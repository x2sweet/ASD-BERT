import re
import os
import re


def extract_rdf_triples(input_dir, output_file):
    pattern = re.compile(r'^<[^>]+>\s+<[^>]+>\s+(".*"@\w+|<[^>]+>)\s+\.$')

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for root, _, files in os.walk(input_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as in_f:
                        for line in in_f:
                            line = line.strip()
                            if pattern.match(line):
                                out_f.write(line + '\n')
                except Exception as e:
                    print(f'处理文件 {file_path} 时出错: {str(e)}')


if __name__ == '__main__':
    input_directory = 'instance'
    output_path = 'data/kg/rdf_triples.txt'
    extract_rdf_triples(input_directory, output_path)
    with open('data/kg/rdf_triples.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    pattern = r'<([^>]+)> <([^>]+)> "(.+?)"@zh .'

    with open('data/kg/kg.txt', 'w', encoding='utf-8') as f:
        for line in lines:
            match = re.match(pattern, line)
            if match:
                subject = match.group(1).split('/')[-1]
                predicate = match.group(2).split('/')[-1]
                obj = match.group(3)
                f.write(f'{subject} {predicate} {obj}\n')