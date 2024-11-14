import hashlib
import os

import requests

# a dict that holds urls
urls = {
    "people.json": "https://members.midasnetwork.us/midascontacts/query/people/visualizer/all",
    "papers.json": "https://members.midasnetwork.us/midascontacts/query/papers/visualizer/all"}


def read_api_key_from_file():
    with open("api_key.txt", "r") as file:
        return file.read().strip()


def update_file(file_name, url, key):
    print("Updating " + file_name)
    file_path = "data_sources" + os.path.sep + file_name
    with open(file_path, 'wb') as out_file:
        content = requests.get(url + "?apiKey=" + key, stream=True).content
        out_file.write(content)

    print("  Size of " + file_name + " is " + str(len(content)) + " bytes")
    # write md5 checksum of file
    print("  Writing md5 checksum for " + file_name)
    with open(file_path + ".md5", 'w') as out_file:
        out_file.write(hashlib.md5(content).hexdigest())


def delete_cache_if_needed(file_name):
    print("Checking if " + file_name + " cache is still valid.")
    try:
        with open("cache/" + file_name + ".md5", 'r') as in_file:
            old_md5 = in_file.read()
    except FileNotFoundError:
        old_md5 = ""

    # if the md5 checksums are different, delete the cache file
    file_path = "data_sources" + os.path.sep + file_name
    with open(file_path + ".md5", 'r') as in_file:
        new_md5 = in_file.read()
        if old_md5 != new_md5:
            print("  Change detected, invalidating cache " + file_name)
            for file in os.listdir("cache"):
                os.remove("cache/" + file)
        else:
            print("  No change detected, " + file_name + "cache is still valid.")


def move_md5_checksum_to_cache_directory(file_name):
    print("Updating cache directory with " + file_name + " checksum.")
    ##make sure the cache directory exists
    os.makedirs("cache", exist_ok=True)
    file_path = "data_sources" + os.path.sep + file_name
    os.rename(file_path + ".md5", "cache/" + file_name + ".md5")


def update_data():
    key = read_api_key_from_file()

    # for each item in the urls dict, update the file
    for file_name, url in urls.items():
        update_file(file_name, url, key)
        delete_cache_if_needed(file_name)

    for file_name, url in urls.items():
        move_md5_checksum_to_cache_directory(file_name)


def main():
    update_data()


if __name__ == "__main__":
    main()
    quit()
