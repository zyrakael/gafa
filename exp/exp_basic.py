import os
import torch
from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
    Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, MambaSimple, TemporalFusionTransformer, SCINet, TimeXer, \
    TimeBridge, MyModel,PatchMLP,MyModel1,MyModel2,MyModel3,MyModel4


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            # 'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            # 'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            # 'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            # 'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'iTransformer': iTransformer,
            # 'Koopa': Koopa,
            'TiDE': TiDE,
            'FreTS': FreTS,
            # 'MambaSimple': MambaSimple,
            'TimeMixer': TimeMixer,
            'TSMixer': TSMixer,
            'SegRNN': SegRNN,
            'TemporalFusionTransformer': TemporalFusionTransformer,
            "SCINet": SCINet,
            'TimeXer': TimeXer,
            'TimeBridge':TimeBridge,
            'MyModel': MyModel,
            'PatchMLP':PatchMLP,
            'MyModel1':MyModel1,
            'MyModel2':MyModel2,
            'MyModel3':MyModel3, 
            'MyModel4':MyModel4,
        }
        if args.model == 'Mamba':
            print('Please make sure you have successfully installed mamba_ssm')
            from models import Mamba
            self.model_dict['Mamba'] = Mamba

        self.device = self._acquire_device()
        # Build model and move to primary device first, then wrap DP if enabled
        base_model = self._build_model()
        self.model = base_model.to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            if self.args.use_multi_gpu:
                print(f"Use Multi-GPU: primary cuda:{self.args.gpu}, devices [{self.args.devices}]")
            else:
                print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
