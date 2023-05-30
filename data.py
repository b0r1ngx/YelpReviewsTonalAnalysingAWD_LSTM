import json

from typing import List

path = r'C:\Users\perke\Downloads\yelp_dataset\yelp_academic_dataset_review.json'


def read_file(file: str, write_to: List, read_first_n_lines: int = 1000, encoding: str = 'utf-8'):
    with open(file, encoding=encoding) as reviews_json_file:
        for _ in range(read_first_n_lines):
            read_line = next(reviews_json_file)
            to_dict = json.loads(read_line)
            write_to.append(to_dict)


reviews: List[dict] = []
read_file(path, reviews, 5000)
print('Reviews size:', len(reviews))
