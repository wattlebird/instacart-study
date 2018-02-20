import pandas as pd
import numpy as np
from setting import *


log=pd.read_csv(DATA + '/log.prior.small.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8}, engine='c' )
log.sort_values(by=['order_id', 'product_id'], inplace=True)
log = log.reset_index(drop=True)

log.groupby('product_id')['add_to_cart_order'].agg(['mean', 'sum', 'min', 'max', 'median', 'std']).add_prefix('item_priority_').to_csv(DATA+'/item_priority.csv')