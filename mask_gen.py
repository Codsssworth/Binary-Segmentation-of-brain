import json
from PIL import Image, ImageDraw
import urllib.request
import os

# url = "https://api.labelbox.com/graphql"
# query = """
# query ($datasetId: ID!, $first: Int) {
#     dataset(id: $datasetId) {
#         dataRows(first: $first) {
#             id
#             rowData
#         }
#     }
# }
# """
# variables = {
#     "datasetId": "<YOUR_DATASET_ID>",
#     "first": 10
# }
# headers = {
#     "Content-Type": "application/json"
# }
# req = urllib.request.Request(url, json.dumps({"query": query, "variables": variables}).encode("utf-8"), headers)
# response = urllib.request.urlopen(req)
# data = json.loads(response.read())
mask_path = 'Unet Data/masks/'
with open('label_data.json', 'r') as f:
    data = json.load(f)

for item in data:
        name= item['ID']
        mask_filename =f'{mask_path}{name}.png'

        if os.path.isfile( mask_filename ):
           print( f"Mask file '{mask_filename}' already exists. Skipping..." )
           image_url = item['Labeled Data']
           image = Image.open( urllib.request.urlopen( image_url ) )
           image.save(f'{name}.png')
        else:
            if 'objects' in item['Label'] and len(item['Label']['objects']) > 0:
                region = item['Label']['objects'][0]['polygon']

                # Extract labeled region details
                regions = item['Label']['objects'][0]['polygon']
                x = []
                y = []
                for point in regions:
                    x.append( point['x'] )
                    y.append( point['y'] )
                vertices = [(xi, yi) for xi, yi in zip( x, y )]
                image_url = item['Labeled Data']
                image = Image.open( urllib.request.urlopen( image_url ) )
                print(type(Image))
                mask = Image.new( 'L', image.size, 0 )
                ImageDraw.Draw( mask ).polygon( vertices, outline=1, fill=255 )
                mask.save( f'{name}.png' )
                regions = []
            else:
                print("no mask for: ",name)