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

product_reordered = \
orders[orders.eval_set=='prior'][['order_id', 'user_id']].merge(
    log,
    on='order_id'
).drop(columns='order_id').drop_duplicates().reset_index(drop=True)

product_reordered = product_reordered.groupby('product_id')['reordered'].agg(['count', 'sum'])
product_reordered['item_first_order_cnt'] = product_reordered['count']-product_reordered['sum']
product_reordered.rename(columns={'sum': 'item_reorder_cnt'}, inplace=True)
product_reordered.drop(columns = 'count', inplace=True)
product_reordered['item_reorder_rate'] = product_reordered.item_reorder_cnt / product_reordered.item_first_order_cnt

product_reordered.to_csv(DATA+'/item_first_reorder_rate.csv')