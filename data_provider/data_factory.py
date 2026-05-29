from data_provider.data_loader import Dataset_DuoLun_Minute,Dataset_YuCheng_Minute,Dataset_XiaoLangDi_Minute,Dataset_ChangLing_Minute,Dataset_DaXingAnLing_Minute, Dataset_HaiBei_Minute,Dataset_Xilin_Minute, Dataset_AiLaoShan_Minute, Dataset_QianYanZhou_Minute, Dataset_XiShuangBanNa_Minute, Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader, Dataset_Meteorology,Dataset_Weather_Minute,Dataset_Electricity_Minute,Dataset_BeijingAirQuality,Dataset_ExchangeRate
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader
from data_provider.data_loader import Dataset_XiaoLangDi_Minute, Dataset_YuCheng_Minute,Dataset_MaoWuSu_Minute,Dataset_DangXiong_Minute,Dataset_BaoTianMan_Minute

data_dict = {
    'DuoLun': Dataset_DuoLun_Minute,
    'XiLin': Dataset_Xilin_Minute,
    'MaoWuSu': Dataset_MaoWuSu_Minute,
    'DangXiong':Dataset_DangXiong_Minute,
    'BaoTianMan':Dataset_BaoTianMan_Minute,
    'AiLaoShan': Dataset_AiLaoShan_Minute,
    'XiShuangBanNa': Dataset_XiShuangBanNa_Minute,
    'QianYanZhou': Dataset_QianYanZhou_Minute,
    'HaiBei': Dataset_HaiBei_Minute,
    'DaXingAnLing': Dataset_DaXingAnLing_Minute,
    'ChangLing': Dataset_ChangLing_Minute,
    'XiaoLangDi': Dataset_XiaoLangDi_Minute,
    'YuCheng': Dataset_YuCheng_Minute,
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Weather':Dataset_Weather_Minute,
    'Electricity':Dataset_Electricity_Minute,
    'BeijingAirQuality':Dataset_BeijingAirQuality,
    'ExchangeRate':Dataset_ExchangeRate,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    # 'PSM': PSMSegLoader,
    # 'MSL': MSLSegLoader,
    # 'SMAP': SMAPSegLoader,
    # 'SMD': SMDSegLoader,
    # 'SWAT': SWATSegLoader,
    # 'UEA': UEAloader,
    # 'Meteorology' : Dataset_Meteorology
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
