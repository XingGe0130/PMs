import datetime


def hours_to_time_stamp(hours, date_begin="1900-01-01 00:00:00"):
    begin = datetime.datetime.strptime(date_begin, "%Y-%m-%d %H:%M:%S")
    return (begin + datetime.timedelta(hours=hours)).strftime("%Y%m%d %H:%M:%S")


def subtract_day(date_begin, days=1):
    begin = datetime.datetime.strptime(date_begin, "%Y%m%d")
    return (begin - datetime.timedelta(days=days)).strftime("%Y%m%d")


def getBetweenDay(begin_date, end_date):
    date_list = list()
    begin_date = datetime.datetime.strptime(begin_date, "%Y%m%d")
    end_date = datetime.datetime.strptime(end_date, "%Y%m%d")
    while begin_date <= end_date:
        date_str = begin_date.strftime("%Y%m%d")
        date_list.append(date_str)
        begin_date += datetime.timedelta(days=1)
    return date_list
