

url = 'https://www.sports-reference.com/cbb/schools/#all_NCAAW_schools'
session = requests.session()
df = pd.read_html(url)
df = df[1]
response = session.get(url)
page_html = BeautifulSoup(response.text, 'html5lib')
page_html = str(page_html).split("SRS back to 2001-02")
w_html = page_html[1]
w_html = BeautifulSoup(w_html, 'html5lib')
url_dict = utils.get_url_dict(w_html)

new_dict = {}
for key in url_dict.keys():
    if ('/women/' in url_dict[key]) & ('/cbb/schools/' in url_dict[key]):
        new_dict[key] = url_dict[key]