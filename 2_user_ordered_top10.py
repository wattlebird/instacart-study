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

order_prior = orders[orders.eval_set=='prior'].order_id
log_prior = log[log.order_id.isin(order_prior)]
log_prior_top10 = log_prior.groupby('product_id')['order_id'].count().sort_values(ascending=False)[:10]
normal_orders = log_prior[log_prior.product_id.isin(log_prior_top10.index)]
u5 = orders[orders.eval_set=='prior'][['order_id', 'user_id']].merge(normal_orders, on='order_id', how='left')
u5 = u5.groupby('user_id')['product_id'].any()
u5.to_csv(DATA+'/user_ordered_top10.csv', header=True)