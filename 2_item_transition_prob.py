import pandas as pd
import numpy as np
from setting import *
from itertools import product

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

order_prodlist = log.groupby('order_id')['product_id'].apply(list).rename('product_list')
order_detail = orders[['order_id', 'user_id']].copy()
order_detail['order_id_t-1'] = order_detail.groupby('user_id')['order_id'].shift(1)
order_detail.dropna(inplace=True)
order_detail['order_id_t-1'] = order_detail['order_id_t-1'].astype(np.int32)
order_detail = order_detail.merge(order_prodlist.to_frame(), left_on='order_id', right_index=True, how='left').\
merge(order_prodlist.to_frame().add_suffix('_t-1'), left_on='order_id_t-1', right_index=True, how='left').dropna()

cur = []
pre = []
for curlst, prelst in order_detail[['product_list', 'product_list_t-1']].values:
    for itm in product(curlst, prelst):
        cur.append(itm[0])
        pre.append(itm[1])
product_trans = pd.DataFrame(data = {
    'pre': pre,
    'cur': cur,
    'cnt': 1
})
product_trans = product_trans.groupby(['pre', 'cur'])['cnt'].sum().rename('cnt').to_frame().reset_index()
product_trans.set_index('pre', inplace=True)
product_trans['pre_cnt'] = product_trans.groupby('pre')['cnt'].sum()
product_trans['prob'] = product_trans.cnt/product_trans.pre_cnt
product_trans.drop(columns=['cnt', 'pre_cnt'], inplace=True)
product_trans.to_csv(DATA+'/item_transition_prob.csv')