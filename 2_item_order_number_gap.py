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

log=pd.read_csv(DATA + '/log.prior.small.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8}, usecols=['order_id', 'product_id', 'reordered'], engine='c' )
log.sort_values(by=['order_id', 'product_id'], inplace=True)
log = log.reset_index(drop=True)

log_extended = pd.merge(orders[['order_id', 'user_id', 'order_number']],
                        log[['order_id', 'product_id']],
                        on='order_id', how='inner')
log_extended['order_number_t-1'] = log_extended.groupby(['product_id', 'user_id'])['order_number'].shift(1)

product_order_number = log_extended[['product_id', 'order_number', 'order_number_t-1']].copy()
product_order_number['item_order_number_gap'] = product_order_number['order_number']-product_order_number['order_number_t-1']
product_order_number.groupby('product_id')['item_order_number_gap'].agg(['mean', 'std', 'min', 'max', 'median']).\
rename(columns={'mean': 'avg'}).add_prefix('item_order_number_gap_').to_csv(DATA+'/item_order_number_gap.csv')