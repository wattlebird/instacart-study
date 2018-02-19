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
order_none = log[log.order_id.isin(order_prior)].groupby('order_id')['reordered'].any().apply(lambda x: not x)

order_prior = orders[orders.eval_set=='prior'][['order_id', 'user_id']]
for i in range(1, 21):
    order_prior['order_id_t-{0}'.format(i)] = order_prior.groupby('user_id')['order_id'].shift(i)

order_none = order_none.to_frame()
for i in range(1, 21):
    order_prior = order_prior.merge(order_none.rename(columns={'reordered': 'order_none_t-{0}'.format(i)}), how='left', left_on='order_id_t-{0}'.format(i), right_index=True, )
o3 = order_prior.drop(columns=['user_id']+['order_id_t-{0}'.format(i) for i in range(1, 21)])
o3.set_index('order_id')
o3.to_csv(DATA+'/order_none_history.csv')