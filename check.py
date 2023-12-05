'''list = ['apple','bananas','orange','eggs']
cos = {'apples' : 50,
               'bananas' : 10,
               'oranges' : 40}
total = 0
item = {'apples': 3, 'bananas': 2, 'oranges': 3}
for key,itme in item.items():
            print(key)
            total = total+(cos[key])*(item[key])

print(total)
'''

import torch


model = torch.load('yolov5s.pt')

num_classes = model.fc.out_features
print(f"The model can predict {num_classes} classes.")
