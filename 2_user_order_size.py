import pandas as pd
import numpy as np
from setting import *

orders = pd.read_csv(DATA + '/orders.small.csv', dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'eval_set': 'category',
        'order_number': np.int16,
        'order_dow': np.int8,
        'order_hour_of_day': np.int8,
        'days_since_prior_order': np.float32}, engine='c').\
        rename(columns = {'order_dow': 'order_week', 'order_hour_of_day': 'order_hour', 'days_since_prior_order': 'order_gap'})

log=pd.concat([
    pd.read_csv(DATA + '/log.train.small.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8}, usecols=['order_id', 'product_id', 'reordered'], engine='c'),
    pd.read_csv(DATA + '/log.prior.small.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8}, usecols=['order_id', 'product_id', 'reordered'], engine='c')],
    ignore_index=True)
log.sort_values(by=['order_id', 'product_id'], inplace=True)
log = log.reset_index(drop=True)

order_product = pd.merge(orders[orders.eval_set=='prior'][['order_id', 'user_id']], log, on='order_id')

user_order_size = pd.concat([
    order_product.groupby('order_id')['product_id'].agg('count').rename('order_cnt'),
    orders[orders.eval_set=='prior'][['order_id', 'user_id']].set_index('order_id')
], axis=1)

u4 = user_order_size.reset_index().groupby('user_id')['order_cnt'].agg(['mean', 'std', 'median', 'min', 'max']).\
add_prefix('user_order_size_')

u4.to_csv(DATA+'/user_order_size.csv')
