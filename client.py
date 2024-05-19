# client.py
import requests
import base64
import os


filename = "./diplom/test_image/Rubble_5-20_mm_0.jpeg"
files = {'file': (filename, open(filename, 'rb'))}

response = requests.post(
    'http://127.0.0.1:8090/YOLOv8m_cls_inferences',
    files=files)

response_content = response.json()
decoded_img = base64.b64decode(response_content['image_with_bbox'])
with open('received_image.jpeg', 'wb') as f:
    f.write(decoded_img)

print(response_content['image_dimensions'])
print(response_content['predcited_labels'])


# response = requests.post(
#     'http://127.0.0.1:8090/YOLOv8m_inferences',
#     files=files,
# )