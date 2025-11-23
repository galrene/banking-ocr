import pytesseract
from PIL import Image
import re
import pandas as pd
from pathlib import Path
import cv2

def preprocess(image: Path):
    img = cv2.imread(image)

    img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to greyscale

    x0 = 170
    x1 = 170 + 850
    y0 = 500
    y1 = 500 + 220 * 8
    crop = img_gs[y0:y1, 
                  x0:x1]
    
    return crop

def ocr_images(directory: str, extension: str = ".PNG"):
    img_texts = []
    
    for image in Path(directory).glob(f'*{extension}'):
        img = Image.fromarray(preprocess(image))
        
        text = pytesseract.image_to_string(img)
        t_arr = text.split('\n')
        img_texts.append(t_arr)

    return img_texts

def parse_ocr_text(img_texts) -> dict:
    # Parse data from ocr text
    data = {}

    p = r'\w+\s.*CZK'
    for text_arr in img_texts:
        for line in text_arr:
            m = re.search(p, line)
            
            if m is not None:
                line = m.group()
                
                split = line.split('-') # [category, spent_CZK]
                cat = split[0].strip()
                val = split[1].strip('CZK').strip().replace(',', '')[:-3]
                data[cat] = val
    
    # Create dict digestible by pandas
    data_for_pd = {}
    for key, val in data.items():
        data_for_pd[key] = [val]

    return data_for_pd


def main():
    img_texts = ocr_images('src_img')
    data = parse_ocr_text(img_texts)

    rename_map = {
            # 'Savings'           : '',
            'Household'         : 'Housing',
            # 'Flexible spending' : '',
            # 'Groceries'         : '',
            'Food and Drink'    : 'Restaurants',
            'Travel'            : 'Travel [Flexible spending]',
            # 'Shopping'          : '',
            # 'Finance'           : '',
            # 'Freetime'          : '',
            # 'Phone'             : '',
            # 'Drinks'            : '',
            'Health and beauty' : 'Drugstore/Home'

        }

    df = pd.DataFrame.from_dict(data).T
    df.rename(index=rename_map, inplace=True)

    df.sort_index(ascending=False)
    print(df)
    df.to_csv('ocr.csv', header=False)
    df.to_clipboard(excel=True, header=False)

if __name__ == '__main__':
    main()