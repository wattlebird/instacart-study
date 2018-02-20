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
orders['order_gap_cumsum']=orders.sort_values(['user_id', 'order_number']).groupby('user_id')['order_gap'].cumsum().fillna(0)

log=pd.read_csv(DATA + '/log.prior.small.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8}, usecols=['order_id', 'product_id', 'reordered'], engine='c' )
log.sort_values(by=['order_id', 'product_id'], inplace=True)
log = log.reset_index(drop=True)

log_extended = log.merge(orders[['order_id', 'user_id', 'order_gap_cumsum']], on='order_id').\
sort_values(['user_id', 'product_id', 'order_gap_cumsum']).reset_index(drop=True)
log_extended['order_gap_cumsum_t-1'] = log_extended.groupby(['user_id', 'product_id'])['order_gap_cumsum'].shift(1)
log_extended['item_order_gap'] = log_extended['order_gap_cumsum']-log_extended['order_gap_cumsum_t-1']
item_order = log_extended.drop(columns=['order_gap_cumsum', 'order_gap_cumsum_t-1'])
i1 = item_order.groupby('product_id')['item_order_gap'].agg(['mean', 'std', 'min', 'max', 'median']).\
rename(columns={'mean': 'avg'}).add_prefix('item_order_gap_')
i2 = item_order.groupby('product_id')['reordered'].agg(['count', 'sum']).\
rename(columns={'count': 'item_order_cnt', 'sum': 'item_reorder_cnt'})
i2['item_reorder_rate'] = i2['item_reorder_cnt']/i2['item_order_cnt']
i3 = item_order[['product_id', 'user_id']].drop_duplicates().groupby('product_id')['user_id'].count().rename('item_customer_cnt')
i0 = pd.concat([i1, i2, i3], axis=1, join='inner')
i0['item_order_per_user'] = i0['item_order_cnt']/i0['item_customer_cnt']
i0.to_csv(DATA+'/item_order.csv')