from .evaluate_vqav2 import eval_vqav2
from .evaluate_vizwiz import eval_vizwiz
from .evaluate_gqa import eval_gqa
from .evaluate_textvqa import eval_textvqa
from .evaluate_ocrvqa import eval_ocrvqa
from .evaluate_vcr import eval_vcr
from .evaluate_vsr import eval_vsr
from .evaluate_okvqa import eval_okvqa
from .evaluate_aokvqa import eval_aokvqa
from .evaluate_sqa import eval_sqa
from .evaluate_mme import eval_mme
from .evaluate_mmbench import eval_mmbench
from .evaluate_seedbench import eval_seedbench
from .evaluate_haloquest import eval_haloquest 
from .evaluate_mmhalbench import eval_mmhalbench


def load_evaluate_fn(dataset):
    eval_fns = {
        'vqav2': eval_vqav2,
        'vizwiz': eval_vizwiz,
        'gqa': eval_gqa,
        'textvqa': eval_textvqa,
        'ocrvqa': eval_ocrvqa,
        'vcr': eval_vcr,
        'vsr': eval_vsr,
        'okvqa': eval_okvqa,
        'aokvqa': eval_aokvqa,
        'sqa': eval_sqa,
        'mme': eval_mme,
        'mmbench': eval_mmbench,
        'seedbench': eval_seedbench,
        'haloquest': eval_haloquest,
        'mmhalbench': eval_mmhalbench,
    }
    if dataset not in eval_fns:
        raise ValueError(f"Unsupported evaluation task: {dataset}")
    return eval_fns[dataset]
