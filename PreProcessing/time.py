# Created by [Yuexiong Ding] at 2018/3/16
# 提取时间特征
#
import pandas as pd


class TimeFeature:
    t_datetime = []

    def __init__(self, datetime):
        self.t_datetime = datetime

    def get_time_feature(self):
        month = self.get_month_feature()
        day = self.get_day_feature()
        hour = self.get_hour_feature()
        return month, day, hour

    def get_month_feature(self):
        return [i.month for i in self.t_datetime]

    def get_day_feature(self):
        return [i.day for i in self.t_datetime]

    def get_hour_feature(self):
        return [i.hour for i in self.t_datetime]
