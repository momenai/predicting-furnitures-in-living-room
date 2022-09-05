import re
import requests
from bs4 import BeautifulSoup

site = 'http://pixabay.com'

response = requests.get(site)

soup = BeautifulSoup(response.text, 'html.parser')
img_tags = soup.find_all('img')

urls = [img['src'] for img in img_tags]


for url in urls:
    filename = re.search(r'/([\w_-]+[.](jpg|gif|png))$', url)
    try:
        with open(filename.group(1), 'wb') as f:
            if 'http' not in url:
                # sometimes an image source can be relative 
                # if it is provide the base url which also happens 
                # to be the site variable atm. 
                url = '{}{}'.format(site, url)
            response = requests.get(url)
            f.write(response.content)
    except:
        pass
		
#reference:
#https://stackoverflow.com/questions/18408307/how-to-extract-and-download-all-images-from-a-website-using-beautifulsoup