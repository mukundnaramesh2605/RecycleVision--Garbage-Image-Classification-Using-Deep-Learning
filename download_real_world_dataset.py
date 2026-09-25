# pip install icrawler
import os
from icrawler.builtin import BingImageCrawler

# descriptive search terms (better results than bare class names)
QUERIES = ['battery', 'rotten fruit vegetable', 'brown glass bottle', 'cardboard box',
           'folded clothes', 'green glass bottle', 'metal can', 'paper sheet',
           'plastic bottle', 'shoes', 'trash garbage waste', 'white glass jar']

# your actual class folder names
FOLDERS = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
           'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']

OUT = "real_world_test"
for query, folder in zip(QUERIES, FOLDERS):
    d = os.path.join(OUT, folder)
    os.makedirs(d, exist_ok=True)
    BingImageCrawler(storage={"root_dir": d}).crawl(keyword=query, max_num=10)
    print(f"done: {folder}")