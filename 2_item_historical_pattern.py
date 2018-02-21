import pandas as pd
import numpy as np
from setting import *

def history_processor_1(itm, limit):
    cnt = 0
    hit = 0
    for i, idx in enumerate(itm):
        if idx+1<=limit:
            cnt+=1
            if i+1<len(itm) and itm[i+1]==itm[i]+1:
                hit+=1
    return cnt, hit

def history_processor_10(itm, limit):
    cnt = 0
    hit = 0
    for i, idx in enumerate(itm):
        if idx+2<=limit:
            cnt+=1
            if i+1<len(itm) and itm[i+1]==itm[i]+2:
                hit+=1
    return cnt, hit

def history_processor_11(itm, limit):
    cnt = 0
    hit = 0
    for i, idx in enumerate(itm):
        if idx+2<=limit and i+1<len(itm) and itm[i+1]==itm[i]+1:
            cnt+=1
            if i+2<len(itm) and itm[i+2]==itm[i]+2:
                hit+=1
    return cnt, hit

def history_processor_100(itm, limit):
    cnt = 0
    hit = 0
    for i, idx in enumerate(itm):
        if idx+3<=limit:
            cnt+=1
            if i+1<len(itm) and itm[i+1]==itm[i]+3:
                hit+=1
    return cnt, hit

def history_processor_101(itm, limit):
    cnt = 0
    hit = 0
    for i, idx in enumerate(itm):
        if idx+3<=limit and i+1<len(itm) and itm[i+1]==itm[i]+2:
            cnt+=1
            if i+2<len(itm) and itm[i+2]==itm[i]+3:
                hit+=1
    return cnt, hit

def history_processor_110(itm, limit):
    cnt = 0
    hit = 0
    for i, idx in enumerate(itm):
        if idx+3<=limit and i+1<len(itm) and itm[i+1]==itm[i]+1:
            cnt+=1
            if i+3<len(itm) and itm[i+2]==itm[i]+3:
                hit+=1
    return cnt, hit

def history_processor_111(itm, limit):
    cnt = 0
    hit = 0
    for i, idx in enumerate(itm):
        if idx+3<=limit and i+2<len(itm) and itm[i+1]==itm[i]+1 and itm[i+2]==itm[i]+2:
            cnt+=1
            if i+3<len(itm) and itm[i+3]==itm[i]+3:
                hit+=1
    return cnt, hit

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
cnt = np.zeros(7)
hit = np.zeros(7)
for row in product_order_series.itertuples():
    if cur==-1:
        cur = row.product_id
    if cur!=row.product_id:
        cnt_list.append(cnt)
        hit_list.append(hit)
        hitprob_list.append(hit/cnt)
        prod_list.append(cur)
        cnt = np.zeros(7)
        hit = np.zeros(7)
        cur = row.product_id
    
    tmp = history_processor_1(row.order_history, row.order_number_max)
    cnt[0]+=tmp[0]
    hit[0]+=tmp[1]
    
    tmp = history_processor_10(row.order_history, row.order_number_max)
    cnt[1]+=tmp[0]
    hit[1]+=tmp[1]
    
    tmp = history_processor_11(row.order_history, row.order_number_max)
    cnt[2]+=tmp[0]
    hit[2]+=tmp[1]
    
    tmp = history_processor_100(row.order_history, row.order_number_max)
    cnt[3]+=tmp[0]
    hit[3]+=tmp[1]
    
    tmp = history_processor_101(row.order_history, row.order_number_max)
    cnt[4]+=tmp[0]
    hit[4]+=tmp[1]
    
    tmp = history_processor_110(row.order_history, row.order_number_max)
    cnt[5]+=tmp[0]
    hit[5]+=tmp[1]
    
    tmp = history_processor_111(row.order_history, row.order_number_max)
    cnt[6]+=tmp[0]
    hit[6]+=tmp[1]
    
cnt_list.append(cnt)
hit_list.append(hit)
hitprob_list.append(hit/cnt)
prod_list.append(cur)

hit_df = pd.DataFrame(data = np.vstack(hit_list), index=prod_list).add_prefix('item_pattern_hit_')
cnt_df = pd.DataFrame(data = np.vstack(cnt_list), index=prod_list).add_prefix('item_pattern_cnt_')
hitprob_df = pd.DataFrame(data = np.vstack(hitprob_list), index=prod_list).fillna(0).add_prefix('item_pattern_hitrate_')
i10 = pd.concat([hit_df, cnt_df, hitprob_df], axis=1, join='inner')

i10.to_csv(DATA+'/item_pattern.csv')