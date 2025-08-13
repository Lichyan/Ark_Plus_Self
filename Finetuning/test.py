import csv, os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

root="./advCheX_test"; csvp="dataset/advCheX/test.csv"
bad=[]
good=0
with open(csvp) as f:
    r=csv.reader(f); header=next(r)
    for i,row in enumerate(r, start=1):
        p=os.path.join(root, row[0])
        try:
            with Image.open(p) as im:
                im = im.convert('RGB')    # 触发颜色转换
                im.load()                 # 强制解码像素
                good += 1
                print("GOOD!")
        except Exception as e:
            print(f"[BAD #{i}] {p} :: {e}")
            bad.append((i,p))
print("BAD =", len(bad))
print("GOOD = ", good)
