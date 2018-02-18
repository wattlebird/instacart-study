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

order_sorted = orders[orders.eval_set=='prior'].sort_values(by=['user_id', 'order_number']).\
               set_index('order_id')
order_sorted['order_gap_cum'] = order_sorted.groupby('user_id').order_gap.cumsum().fillna(0)

log['fresh'] = 1-log.reordered

order_prod_summary = pd.concat([log.groupby('order_id')['reordered'].agg(['mean', 'sum']).\
            rename(columns={'mean': 'order_reorder_rate', 'sum': 'order_reorder_cnt'}),
          log.groupby('order_id')['product_id'].agg('count').\
            rename('order_overall_cnt'),
          log.groupby('order_id')['fresh'].agg('sum').\
            rename('order_nouveau_cnt')], axis=1)
order_prod = pd.merge(order_sorted[['user_id', 'order_number']], order_prod_summary, 
                      left_index=True, right_index=True).\
             sort_values(['user_id', 'order_number'])
order_prod['order_nouveau_cum'] = order_prod.groupby(by='user_id')['order_nouveau_cnt'].cumsum()
order_prod['order_overall_cum'] = order_prod.groupby(by='user_id')['order_overall_cnt'].cumsum()
order_prod['order_nouveau_avg'] = order_prod.order_nouveau_cum/order_prod.order_number
order_prod['order_overall_avg'] = order_prod.order_overall_cum/order_prod.order_number

order_prodlist = log.groupby(by='order_id')['product_id'].apply(list).rename('product_list')
order_prodlist = pd.concat([order_sorted, order_prodlist], axis=1, join='inner')
order_prodlist['product_t-1'] = order_prodlist.groupby('user_id').product_list.shift(1)
order_prodlist['product_t-2'] = order_prodlist.groupby('user_id').product_list.shift(2)
order_prodlist['product_t-3'] = order_prodlist.groupby('user_id').product_list.shift(3)
order_prodlist['product_t-4'] = order_prodlist.groupby('user_id').product_list.shift(4)
order_prodlist['product_t-5'] = order_prodlist.groupby('user_id').product_list.shift(5)
order_prodlist['order_repeat_rate_window_1'] = order_prodlist.apply(
    lambda x: np.nan if(type(x['product_t-1'])!=list) else len(set(x['product_list']).intersection(x['product_t-1']))
              /float(len(x['product_list'])),
    axis=1
)
order_prodlist['order_repeat_rate_window_2'] = order_prodlist.apply(
    lambda x: np.nan if(type(x['product_t-2'])!=list) else len(set(x['product_list']).intersection(x['product_t-1']+x['product_t-2']))
              /float(len(x['product_list'])),
    axis=1
)
order_prodlist['order_repeat_rate_window_3'] = order_prodlist.apply(
    lambda x: np.nan if(type(x['product_t-3'])!=list) else len(set(x['product_list']).intersection(x['product_t-1']+x['product_t-2']+x['product_t-3']))
              /float(len(x['product_list'])),
    axis=1
)
order_prodlist['order_repeat_rate_window_4'] = order_prodlist.apply(
    lambda x: np.nan if(type(x['product_t-4'])!=list) else len(set(x['product_list']).intersection(x['product_t-1']+x['product_t-2']+x['product_t-3']+x['product_t-4']))
              /float(len(x['product_list'])),
    axis=1
)
order_prodlist['order_repeat_rate_window_5'] = order_prodlist.apply(
    lambda x: np.nan if(type(x['product_t-5'])!=list) else len(set(x['product_list']).intersection(x['product_t-1']+x['product_t-2']+x['product_t-3']+x['product_t-4']+x['product_t-5']))
              /float(len(x['product_list'])),
    axis=1
)
o1 = pd.merge(order_prodlist, order_prod, left_index=True, right_index=True)
order_prod.drop(columns=['user_id', 'order_number', 'order_reorder_cnt'], inplace=True)
order_prodlist.drop(columns=['user_id', 'eval_set', 'order_number', 'product_list', 'product_t-1', 
                    'product_t-2', 'product_t-3', 'product_t-4', 'product_t-5'], inplace=True)
o1 = pd.concat([order_prod, order_prodlist], axis=1, join='inner')
o1.to_csv(DATA+'/order_repeat_previous_ratio.csv')