import os
import requests
from bs4 import BeautifulSoup
from utils import Utils

class BaseDownloader:
    def __init__(self, min_th_version = 3, max_th_version = 15):
        self.base_url = 'https://www.clasher.us/clash-of-clans/layouts/'
        self.min_th_version = min_th_version
        self.max_th_version = max_th_version
        self.out_folder = 'layouts'
        self.create_folder(self.out_folder)
        self.current_th = None
        self.utils = Utils()

    def __format_params(self, params):
        string = []
        for each_key, value in params.items():
            string.append("{}={}".format(each_key, value))
        return "&".join(string)

    def get_request(self, url, params={"page": 94}):
        print('Getting Url {}?{}'.format(url, self.__format_params(params)))
        response = requests.get(url, params=params)
        return self.get_base_content(response.content)

    def create_folder(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    def download_image(self, url):
        response = requests.get(url)
        if response.status_code == 200:
            with open('layouts/{}/{}'.format(self.current_th, os.path.basename(url)), 'wb') as fptr:
                fptr.write(response.content)

    def get_base_content(self, content):
        soup = BeautifulSoup(content, features="lxml")
        check = soup.find('div', {'class' : ['alert', 'alert-dismissible', 'alert-danger']})
        if check:
            return False
        images = soup.find_all('img', {'data-src' != None})
        invalid_images = 0
        image_array = []
        for image in images:
            try:
                if self.current_th in image['alt'].split(' ')[0].lower():
                    image_array.append(image['data-src'].replace('thumb', 'full'))
                else:
                    invalid_images += 1
            except:
                invalid_images += 1
        self.utils.run_parallel(self.download_image, 20, image_array)
        return (invalid_images != len(images))

    def solve_for_th(self, th_level):
        url_for_th = "{}town-hall-{}".format(self.base_url, th_level)
        params={"page": 1}
        self.current_th = 'th{}'.format(th_level)
        self.create_folder(os.path.join(self.out_folder, self.current_th))
        while self.get_request(url_for_th, params):
            params["page"] += 1

    def run_downloader(self):
        for i in range(self.min_th_version, self.max_th_version + 1):
            self.solve_for_th(i)


if __name__ == "__main__":
    base_downloader = BaseDownloader()
    base_downloader.run_downloader()