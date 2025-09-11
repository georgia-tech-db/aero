import requests
from pathlib import Path
import csv
from tqdm import tqdm

TABLE_NAME = "foodreview"

CLEAR_TABLE_QUERY = f"""DROP TABLE IF EXISTS {TABLE_NAME};"""
CREATE_TABLE_QUERY = (
    f"""CREATE TABLE {TABLE_NAME} (name TEXT(10), rating INTEGER, review TEXT(1000));"""
)
INSERT_QUERY = """INSERT INTO foodreview (name, rating, review) VALUES ('{name}', {rating}, '{review}');"""
ROW_COUNT_QUERY = f"SELECT COUNT(name) FROM foodreview;"
LLM_QUERY = f"""SELECT name FROM {TABLE_NAME}
WHERE ChatGPT('What is the following review about ? Only
choose \"food\" or \"service\"', review) = 'food' AND rating <= 1;"""
RATING_QUERY = f"""SELECT name FROM {TABLE_NAME} WHERE rating > 3;"""


def _execute_query(query: str) -> str:
    response = requests.post("http://127.0.0.1:8001/", json=query)
    return response.text


def main():
    print(_execute_query("SHOW TABLES;"))
    print(_execute_query(CLEAR_TABLE_QUERY))
    print(CREATE_TABLE_QUERY)
    print(_execute_query(CREATE_TABLE_QUERY))


if __name__ == "__main__":
    main()
