# import requests

# claim = "Jackie (2016 film) was directed by Peter Jackson."
# evidence = "Jackie is a 2016 biographical drama film directed by Pablo Larraín and written by Noah Oppenheim ."
# web = requests.get(f"http://140.115.54.36/?claim={claim}&evidence={evidence}")
 
# print(web.text)

import requests

# # 英文
claim = "Jackie (2016 film) was directed by Peter Jackson."
url = "https://en.wikipedia.org/wiki/Jackie_(2016_film)"

# # 中文
# claim = "新疆棉花生產已高度機械化，不需要強迫勞動。"
# url = "https://www.piyao.org.cn/2021-03/25/c_1211081883.htm"

# # equery
web = requests.get(f"http://140.115.54.36/equery/?claim={claim}")
print(web.text)
# # result -> {"equery":"Jackie_(2016_film)","state":"1"}

# everify
web = requests.get(f"http://140.115.54.36/everify/?claim={claim}&url={url}")
print(web.text)
# result -> {"label":1,"evidence":
#   ["Jackie grossed $14 million in the United States and Canada and $22.",
#   " Thus, Jackie is the first film that he could approach from a woman's perspective." ...],
#   "state":"1"}

# cverify
web = requests.get(f"http://140.115.54.36/cverify/?claim={claim}&url={url}")
print(web.text)