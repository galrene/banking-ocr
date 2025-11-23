import pytesseract
from PIL import Image
import re
import pandas as pd
from pathlib import Path
import cv2
from typing import Any, Dict, List, Optional

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
    
    for image in Path(directory).glob(f'*{extension}', case_sensitive=False):
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


def dict_to_categorical_dataframe(data_dict: Dict[str, Any], rename_map, custom_order: List[str]
                                 ) -> pd.DataFrame:
    data_to_process = data_dict.copy()

    df = pd.DataFrame.from_dict(
        data_to_process, 
        orient='index', 
        columns=['Value']
    )
    
    df.rename(index=rename_map, inplace=True)
    df = df.reset_index()
    df = df.rename(columns={'index': 'Category'})

    # 1. Add categories that are in custom order but are missing in data
    order_set = set(custom_order)
    dict_keys = set(df['Category'].tolist())
    missing_data_keys = order_set - dict_keys
    
    for key in missing_data_keys:
        df.loc[len(df)] = [key, 0]
    
    # 2. Identify categories in the data (now including the zeros) that are NOT in custom_order
    df_keys = set(df['Category'].tolist())
    missing_categories_from_order = list(df_keys - order_set)
    
    # 3. Define the FINAL categories list
    categories_list = custom_order + missing_categories_from_order
    
    # 4. Apply categorical type and sort immediately
    df['Category'] = pd.Categorical(
        df['Category'], 
        categories=categories_list, 
        ordered=True
    )
    df = df.sort_values(by='Category').reset_index(drop=True)

    return df


def main():
    img_texts = ocr_images('src_img')
    if len(img_texts) == 0:
        raise ValueError("OCR hasn't found anything")
    data = parse_ocr_text(img_texts)
    
    # rename categories
    rename_map = {
            # 'Savings'           : '',
            'Household'         : 'Housing',
            # 'Flexible spending' : '',
            # 'Groceries'         : '',
            'Food and Drink'    : 'Restaurants',
            'Travel'            : 'Travel [Flexible spending]',
            # 'Shopping'          : '',
            'Finance'           : 'Finance [Cash]',
            # 'Freetime'          : '',
            # 'Phone'             : '',
            # 'Drinks'            : '',
            'Health and beauty' : 'Drugstore/Home'

        }

    custom_order = ['Housing','Restaurants','Groceries','Drugstore/Home','Drinks','Shopping',
                    'Transport','Phone','Freetime','Flexible spending','Savings']
    df = dict_to_categorical_dataframe(data, rename_map=rename_map, custom_order=custom_order)
    print(df)

    
    df.to_csv('ocr.csv', header=False, index=False)
    df.to_clipboard(excel=True, header=False, index=False)

if __name__ == '__main__':
    main()