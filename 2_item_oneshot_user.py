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

log_extended = log.merge(orders[['order_id', 'user_id']], on='order_id', how='left')
item_user = log_extended.groupby(['product_id', 'user_id'])['order_id'].count().\
rename('item_user_cnt').to_frame().reset_index()

item_user['item_user_oneshot'] = (item_user.item_user_cnt==1)*1
item_user = item_user.groupby('product_id')['item_user_oneshot'].agg(['count', 'sum']).\
rename(columns = {'sum': 'item_oneshot_user_cnt'})
item_user['item_oneshot_user_percent'] = item_user['item_oneshot_user_cnt']/item_user['count']
item_user.drop(columns='count', inplace=True)
item_user.to_csv(DATA+'/item_oneshot_user.csv')