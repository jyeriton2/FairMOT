import cv2
import json


color_point = {'black' : (80,120), 'white' : (80,420), 'red1' : (80,730), 'red2' : (80,1030), 'yellow1' : (80,1350), 'green1' : (80,1660),
        'green2' : (310,120), 'blue1' : (310,420), 'blue2' : (310,730), 'blue3' : (310,1030), 'purple' : (310,1350), 'gray1' : (310,1660),
        'blue4' : (560,120), 'orange' : (560,420), 'gray2' : (560,730), 'yellow2' : (560,1030), 'blue5' : (560,1350), 'green3' : (560,1660)}

img = cv2.imread('./bbox_color.jpg',cv2.IMREAD_COLOR)
out = {}
with open('./bbox_color.json','w', encoding='utf-8') as f:
    print(color_point)
    for c in color_point.keys():
        print(c)
        print(color_point[c])
        print(img[color_point[c]])
        out[c] = img[color_point[c]].tolist()
    print(out)
    json.dump(out,f,ensure_ascii=False,indent='\t')
