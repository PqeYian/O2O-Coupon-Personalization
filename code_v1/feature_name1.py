# -*- coding: utf-8 -*-

def feature_name(train_quality=None):
    # 用户特征
    # 用户使用优惠券消费次数
    import pandas as pd
    data_user = pd.DataFrame()
    data_user['user_use_coupon_times'] = (train_quality[['date_received','date']].count(axis=1) == 2).groupby(train_quality['user_id']).sum()
    
    # 用户总消费次数
    data_user['user_consume_times'] = (train_quality.loc[:,'date':'date'].count(axis=1)).groupby(train_quality['user_id']).sum()
    
    # 用户使用优惠券消费次数与总消费次数的比值
    data_user['user_use_coupon_rate'] = data_user['user_use_coupon_times']/data_user['user_consume_times']
    data_user['user_use_coupon_rate'] = data_user['user_use_coupon_rate'].fillna(0)
    
    # 用户领取优惠券而未使用的数量
    data_user['user_receive_coupon_unused_times'] = ((train_quality.loc[:, 'coupon_id'].notnull())&(train_quality.loc[:, 'date'].isnull())).groupby(train_quality['user_id']).sum()
    
    # 用户使用优惠券的日期平均与领取日期相隔多少天
    data_user['user_mean_use_coupon_interval'] = (train_quality['date']-train_quality['date_received']).dt.days.groupby(train_quality['user_id']).mean()
    data_user['user_mean_use_coupon_interval'] = data_user['user_mean_use_coupon_interval'].fillna(data_user['user_mean_use_coupon_interval'].max() + 1)
    
    
    # 商户特征
    # 商户发放的优惠券被使用的数量
    data_merchant = pd.DataFrame()
    data_merchant['merchant_launch_coupon_used_count'] = (train_quality[['date_received','date']].count(axis=1) == 2).groupby(train_quality["merchant_id"]).sum()
    
    # 商户发放的优惠券被使用数与商户总消费次数的比值
    merchant_consume_times = (train_quality.loc[:, 'date':'date'].count(axis=1)).groupby(train_quality['merchant_id']).sum()
    data_merchant['merchant_launch_coupon_used_rate'] = data_merchant['merchant_launch_coupon_used_count']/merchant_consume_times
    data_merchant['merchant_launch_coupon_used_rate'] = data_merchant['merchant_launch_coupon_used_rate'].fillna(0)
    
    # 商户发放优惠券的数量
    data_merchant['merchant_launch_coupon_count'] = (train_quality.loc[:, 'coupon_id':'coupon_id'].count(axis=1)).groupby(train_quality["merchant_id"]).sum()
    
    # 商户发放优惠券而未被使用的数量
    data_merchant['user_receive_coupon_unused_times'] = ((train_quality.loc[:, 'coupon_id'].notnull()) & (train_quality.loc[:, 'date'].isnull())).groupby(train_quality['merchant_id']).sum()
    
    # 商户发放的优惠券平均相隔多少天会被使用
    data_merchant['merchant_mean_launch_coupon_interval'] = (train_quality['date']-train_quality['date_received']).dt.days.groupby(train_quality['merchant_id']).mean()
    data_merchant['merchant_mean_launch_coupon_interval'] = data_merchant["merchant_mean_launch_coupon_interval"].fillna(
            data_merchant['merchant_mean_launch_coupon_interval'].max() + 1)
    
    # 优惠券
    # 优惠券流行度=被使用优惠券/发放优惠券总数 (9364,)
    data_coupon = pd.DataFrame()
    day_fifteen = (train_quality['date'] - train_quality['date_received']).dt.days<=15
    data_coupon['coupon_fifteen_used_count'] = (train_quality.loc[day_fifteen, ['date_received', 'date']].count(axis=1)==2).groupby(train_quality["coupon_id"]).sum()
    coupon_consume_times = (train_quality.loc[day_fifteen, 'date':'date'].count(axis=1)).groupby(train_quality["coupon_id"]).sum()
    
    data_coupon['coupon_used_rate'] = data_coupon['coupon_fifteen_used_count']/coupon_consume_times
    data_coupon['coupon_used_rate'] = data_coupon['coupon_used_rate'].fillna(0)
    return data_user, data_merchant, data_coupon
