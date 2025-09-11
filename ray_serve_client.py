import csv
import time
from pathlib import Path

import requests
from tqdm import tqdm

TABLE_NAME = "foodreview"

CLEAR_TABLE_QUERY = f"""DROP TABLE IF EXISTS {TABLE_NAME};"""
CREATE_TABLE_QUERY = (
    f"""CREATE TABLE {TABLE_NAME} (name TEXT(10), rating INTEGER, review TEXT(1000));"""
)
INSERT_QUERY = """INSERT INTO foodreview (name, rating, review) VALUES ('{name}', {rating}, '{review}');"""
ROW_COUNT_QUERY = f"SELECT COUNT(name) FROM foodreview;"
LLM_QUERY = f"""SELECT name FROM {TABLE_NAME}
WHERE ChatGPT('What is the following review about ? Only choose \"food\" or \"service\"', review) = 'food'
AND rating <= 1;"""
RATING_QUERY = f"""SELECT name FROM {TABLE_NAME} WHERE rating > 3;"""


def _execute_query(query: str) -> str:
    response = requests.post("http://127.0.0.1:8000/", json=query)
    return response.text


def main():
    print(_execute_query(CLEAR_TABLE_QUERY))
    print(CREATE_TABLE_QUERY)
    print(_execute_query(CREATE_TABLE_QUERY))

    print("Reading normal.txt and populating review_table ...")

    with Path("data", "normal.txt").open() as f:
        reader = csv.reader(f, delimiter="|")
        next(reader, None)
        for i, row in enumerate(tqdm(reader)):
            values = {
                "name": f"Customer{i+1}",
                "rating": int(row[0]),
                "review": str(row[1]).replace("'", "\\'").replace('"', '\\"'),
            }
            query = INSERT_QUERY.format(**values)
            _execute_query(query)

        print(_execute_query(ROW_COUNT_QUERY))
        start_time = time.perf_counter()
        print(_execute_query(LLM_QUERY))
        end_time = time.perf_counter()

        print(f"Time elapsed: {end_time - start_time}s")


if __name__ == "__main__":
    main()
