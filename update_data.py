import requests


def main():
    key = "a23afe9bce62d311a8948609e347d76a"
    url = "https://members.midasnetwork.us/midascontacts/query/people/visualizer/all?apiKey=a23afe9bce62d311a8948609e347d76a"

    with open('data_sources/people.json', 'wb') as out_file:
        content = requests.get(url, stream=True).content
        out_file.write(content)

    url = "https://members.midasnetwork.us/midascontacts/query/papers/visualizer/all?apiKey=a23afe9bce62d311a8948609e347d76a"

    with open('data_sources/papers.json', 'wb') as out_file:
        content = requests.get(url, stream=True).content
        out_file.write(content)


if __name__ == "__main__":
    main()
    quit()
