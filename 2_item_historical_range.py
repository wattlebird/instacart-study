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
order_amount = log_extended.groupby('user_id')['order_number'].max().rename('order_number_max').to_frame().reset_index()
product_order_series = log_extended.groupby(['product_id', 'user_id'])['order_number'].agg({
    'order_history': lambda x: tuple(x)
}).reset_index()
product_order_series = product_order_series.merge(order_amount, on='user_id').sort_values(by=['product_id', 'user_id'])
product_order_series.reset_index(drop=True, inplace=True)

cur = -1
prod_list = []
cnt_list = []
hit_list = []
hitprob_list = []
cnt = np.zeros(4)
hit = np.zeros(4)
for row in product_order_series.itertuples():
    if cur==-1:
        cur = row.product_id
    if cur!=row.product_id:
        cnt_list.append(cnt)
        hit_list.append(hit)
        hitprob_list.append(hit/cnt)
        prod_list.append(cur)
        cnt = np.zeros(4)
        hit = np.zeros(4)
        cur = row.product_id
    
    for i, n in enumerate(row.order_history):
        itm = row.order_history
        maxl = row.order_number_max
        if itm[i]+3<maxl:
            cnt[0]+=1
            if i+1<len(itm) and itm[i+1]-itm[i]<3:
                hit[0]+=1
        if itm[i]+4<maxl:
            cnt[1]+=1
            if i+1<len(itm) and itm[i+1]-itm[i]<4:
                hit[1]+=1
        if itm[i]+5<maxl:
            cnt[2]+=1
            if i+1<len(itm) and itm[i+1]-itm[i]<5:
                hit[2]+=1
        if itm[i]+6<maxl:
            cnt[3]+=1
            if i+1<len(itm) and itm[i+1]-itm[i]<6:
                hit[3]+=1
        
cnt_list.append(cnt)
hit_list.append(hit)
hitprob_list.append(hit/cnt)
prod_list.append(cur)

hit_df = pd.DataFrame(data = np.vstack(hit_list), index=prod_list, columns=['3', '4', '5', '6']).add_prefix('item_range_hit_')
cnt_df = pd.DataFrame(data = np.vstack(cnt_list), index=prod_list, columns=['3', '4', '5', '6']).add_prefix('item_range_cnt_')
hitprob_df = pd.DataFrame(data = np.vstack(hitprob_list), index=prod_list, columns=['3', '4', '5', '6']).fillna(0).add_prefix('item_range_hitrate_')
i11 = pd.concat([hit_df, cnt_df, hitprob_df], axis=1, join='inner')

i11.to_csv(DATA+'/item_range.csv')