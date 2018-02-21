import pandas as pd
import numpy as np
from setting import *

log=pd.read_csv(DATA + '/log.prior.small.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8}, usecols=['order_id', 'product_id', 'reordered'], engine='c' )
log.sort_values(by=['order_id', 'product_id'], inplace=True)
log = log.reset_index(drop=True)
log.set_index('order_id', inplace=True)
log['product_cnt'] = log.groupby('order_id')['product_id'].count()
product_appearance = log.set_index('product_id').drop(columns='reordered')
product_appearance = product_appearance.groupby('product_id')['product_cnt'].agg(['mean', 'std', 'min', 'max', 'median'])
product_appearance.rename(columns={'mean':'avg'}).add_prefix('item_context_')
product_appearance.to_csv(DATA+'/item_context.csv')