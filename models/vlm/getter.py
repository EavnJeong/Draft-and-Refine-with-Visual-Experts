from .loader import (
    load_idefics,
    load_paligemma,
    load_instructblip,
    load_llava,
    load_qwen25_vl,
    load_cogvlm,
    load_minigptv2,
)

from .idefics_predict import (
    infer_idefics,
    infer_idefics_vizwiz,
    infer_idefics_gqa,
    infer_idefics_textvqa,
    infer_idefics_ocrvqa,

    infer_idefics_cococaption,
    infer_idefics_nocaps,
    infer_idefics_flickr,

    infer_idefics_vcr,
    infer_idefics_vsr,

    infer_idefics_okvqa,
    infer_idefics_aokvqa,
    infer_idefics_sqa,

    infer_idefics_mme,
    infer_idefics_mmbench,
    infer_idefics_seedbench,

    infer_idefics_haloquest,
    infer_idefics_mmhalbench,
)
from .pali_predict import (
    infer_paligemma,
    infer_paligemma_vizwiz,
    infer_paligemma_gqa,
    infer_paligemma_textvqa,
    infer_paligemma_ocrvqa,

    infer_paligemma_cococaption,
    infer_paligemma_nocaps,
    infer_paligemma_flickr,

    infer_paligemma_vcr,
    infer_paligemma_vsr,
    
    infer_paligemma_okvqa,
    infer_paligemma_aokvqa,
    infer_paligemma_sqa,

    infer_paligemma_mme,
    infer_paligemma_mmbench,
    infer_pali_seedbench,

    infer_pali_haloquest,
    infer_pali_mmhalbench,
)
from .instructblip_predict import (
    infer_instructblip,
    infer_instructblip_vizwiz,
    infer_instructblip_gqa,
    infer_instructblip_textvqa,
    infer_instructblip_ocrvqa,

    infer_instructblip_cococaption,
    infer_instructblip_nocaps,
    infer_instructblip_flickr,

    infer_instructblip_vcr,
    infer_instructblip_vsr,

    infer_instructblip_okvqa,
    infer_instructblip_aokvqa,
    infer_instructblip_sqa,

    infer_instructblip_mme,
    infer_instructblip_mmbench,
    infer_instructblip_seedbench,

    infer_instructblip_haloquest,
    infer_instructblip_mmhalbench,
)
from .llava_predict import (
    infer_llava, 
    infer_llava_vizwiz,
    infer_llava_gqa,

    infer_llava_cococaption,
    infer_llava_nocaps,
    infer_llava_flickr,
    infer_llava_textvqa,
    infer_llava_ocrvqa,

    infer_llava_vcr,
    infer_llava_vsr,

    infer_llava_okvqa,
    infer_llava_aokvqa,
    infer_llava_sqa,

    infer_llava_mme,
    infer_llava_mmbench,
    infer_llava_seedbench,

    infer_llava_haloquest,
    infer_llava_mmhalbench,
)
from .qwen_vl_predict import (
    infer_qwen,
    infer_qwen_vizwiz,
    infer_qwen_gqa,
    infer_qwen_textvqa,
    infer_qwen_ocrvqa,

    infer_qwen_cococaption,
    infer_qwen_nocaps,
    infer_qwen_flickr,

    infer_qwen_vcr,
    infer_qwen_vsr,

    infer_qwen_okvqa,
    infer_qwenvl_aokvqa,
    infer_qwen_sqa,

    infer_qwen_mme,
    infer_qwen_mmbench,
    infer_qwen_seedbench,

    infer_qwen_haloquest,
    infer_qwen_mmhalbench,
)
from .cogvlm_predict import (
    infer_cogvlm,
    infer_cogvlm_vizwiz,
    infer_cogvlm_gqa,
    infer_cogvlm_textvqa,
    infer_cogvlm_ocrvqa,

    infer_cogvlm_cococaption,
    infer_cogvlm_nocaps,
    infer_cogvlm_flickr,

    infer_cogvlm_vcr,
    infer_cogvlm_vsr,
    
    infer_cogvlm_okvqa,
    infer_cogvlm_aokvqa,
    infer_cogvlm_sqa,

    infer_cogvlm_mme,
    infer_cogvlm_mmbench,
    infer_cogvlm_seedbench,
    
    infer_cogvlm_haloquest,
    infer_cogvlm_mmhalbench,
)
from .minigptv2_predict import (
    infer_minigptv2,
    infer_minigptv2_vizwiz,
    infer_minigptv2_gqa,
    infer_minigptv2_textvqa,
    infer_minigptv2_ocrvqa,

    infer_minigptv2_cococaption,
    infer_minigptv2_nocaps,
    infer_minigptv2_flickr,

    infer_minigptv2_vcr,
    infer_minigptv2_vsr,

    infer_minigptv2_okvqa,
    infer_minigptv2_aokvqa,
    infer_minigptv2_sqa,

    infer_minigptv2_mme,
    infer_minigptv2_mmbench,
    infer_minigptv2_seedbench,

    infer_minigptv2_mmhalbench,
    infer_minigptv2_haloquest,
)


vlm_loader_map = {
    'idefics': load_idefics,
    'pali': load_paligemma,
    'instructblip': load_instructblip,
    'llava': load_llava,
    'qwen_vl': load_qwen25_vl,
    'cogvlm': load_cogvlm,
    'minigptv2': load_minigptv2,
}
vlm_run_map = {
    'idefics': {
        'vqav2': infer_idefics,
        'vizwiz': infer_idefics_vizwiz,
        'gqa': infer_idefics_gqa,
        'textvqa': infer_idefics_textvqa,
        'ocrvqa': infer_idefics_ocrvqa,

        'cococaption': infer_idefics_cococaption,
        'nocaps': infer_idefics_nocaps,
        'flickr': infer_idefics_flickr,

        'vcr': infer_idefics_vcr,
        'vsr': infer_idefics_vsr,

        'okvqa': infer_idefics_okvqa,
        'aokvqa': infer_idefics_aokvqa,
        'sqa': infer_idefics_sqa,
        
        'mme': infer_idefics_mme,
        'mmbench': infer_idefics_mmbench,
        'seedbench': infer_idefics_seedbench,

        'haloquest': infer_idefics_haloquest,
        'mmhalbench': infer_idefics_mmhalbench,
    },
    'pali': {
        'vqav2': infer_paligemma,
        'vizwiz': infer_paligemma_vizwiz,
        'gqa': infer_paligemma_gqa,
        'textvqa': infer_paligemma_textvqa,
        'ocrvqa': infer_paligemma_ocrvqa,

        'cococaption': infer_paligemma_cococaption,
        'nocaps': infer_paligemma_nocaps,
        'flickr': infer_paligemma_flickr,

        'vcr': infer_paligemma_vcr,
        'vsr': infer_paligemma_vsr,
        
        'okvqa': infer_paligemma_okvqa,
        'aokvqa': infer_paligemma_aokvqa,
        'sqa': infer_paligemma_sqa,
        
        'mme': infer_paligemma_mme,
        'mmbench': infer_paligemma_mmbench,
        'seedbench': infer_pali_seedbench,

        'haloquest': infer_pali_haloquest,
        'mmhalbench': infer_pali_mmhalbench,
    },
    'instructblip': {
        'vqav2': infer_instructblip,
        'vizwiz': infer_instructblip_vizwiz,
        'gqa': infer_instructblip_gqa,
        'textvqa': infer_instructblip_textvqa,
        'ocrvqa': infer_instructblip_ocrvqa,
        
        'cococaption': infer_instructblip_cococaption,
        'nocaps': infer_instructblip_nocaps,
        'flickr': infer_instructblip_flickr,

        'vcr': infer_instructblip_vcr,
        'vsr': infer_instructblip_vsr,
        
        'okvqa': infer_instructblip_okvqa,
        'aokvqa': infer_instructblip_aokvqa,
        'sqa': infer_instructblip_sqa,

        'mme': infer_instructblip_mme,
        'mmbench': infer_instructblip_mmbench,
        'seedbench': infer_instructblip_seedbench,

        'haloquest': infer_instructblip_haloquest,
        'mmhalbench': infer_instructblip_mmhalbench,
    },
    'llava': {
        'vqav2': infer_llava,
        'vizwiz': infer_llava_vizwiz,
        'gqa': infer_llava_gqa,
        'textvqa': infer_llava_textvqa,
        'ocrvqa': infer_llava_ocrvqa,

        'cococaption': infer_llava_cococaption,
        'nocaps': infer_llava_nocaps,
        'flickr': infer_llava_flickr,

        'vcr': infer_llava_vcr,
        'vsr': infer_llava_vsr,
        
        'okvqa': infer_llava_okvqa,
        'aokvqa': infer_llava_aokvqa,
        'sqa': infer_llava_sqa,
        
        'mme': infer_llava_mme,
        'mmbench': infer_llava_mmbench,
        'seedbench': infer_llava_seedbench,

        'haloquest': infer_llava_haloquest,
        'mmhalbench': infer_llava_mmhalbench,
    },
    'qwen_vl': {
        'vqav2': infer_qwen,
        'vizwiz': infer_qwen_vizwiz,
        'gqa': infer_qwen_gqa,
        'textvqa': infer_qwen_textvqa,
        'ocrvqa': infer_qwen_ocrvqa,

        'cococaption': infer_qwen_cococaption,
        'nocaps': infer_qwen_nocaps,
        'flickr': infer_qwen_flickr,

        'vcr': infer_qwen_vcr,
        'vsr': infer_qwen_vsr,

        'okvqa': infer_qwen_okvqa,
        'aokvqa': infer_qwenvl_aokvqa,
        'sqa': infer_qwen_sqa,
        
        'mme': infer_qwen_mme,
        'mmbench': infer_qwen_mmbench,
        'seedbench': infer_qwen_seedbench,

        'haloquest': infer_qwen_haloquest,
        'mmhalbench': infer_qwen_mmhalbench,
    },
    'cogvlm': {
        'vqav2': infer_cogvlm,
        'vizwiz': infer_cogvlm_vizwiz,
        'gqa': infer_cogvlm_gqa,
        'textvqa': infer_cogvlm_textvqa,
        'ocrvqa': infer_cogvlm_ocrvqa,

        'cococaption': infer_cogvlm_cococaption,
        'nocaps': infer_cogvlm_nocaps,
        'flickr': infer_cogvlm_flickr,

        'vcr': infer_cogvlm_vcr,
        'vsr': infer_cogvlm_vsr,

        'okvqa': infer_cogvlm_okvqa,
        'aokvqa': infer_cogvlm_aokvqa,
        'sqa': infer_cogvlm_sqa,
        
        'mme': infer_cogvlm_mme,
        'mmbench': infer_cogvlm_mmbench,
        'seedbench': infer_cogvlm_seedbench,

        'haloquest': infer_cogvlm_haloquest,
        'mmhalbench': infer_cogvlm_mmhalbench,
    },
    'minigptv2': {
        'vqav2': infer_minigptv2,
        'vizwiz': infer_minigptv2_vizwiz,
        'gqa': infer_minigptv2_gqa,
        'textvqa': infer_minigptv2_textvqa,
        'ocrvqa': infer_minigptv2_ocrvqa,

        'cococaption': infer_minigptv2_cococaption,
        'nocaps': infer_minigptv2_nocaps,
        'flickr': infer_minigptv2_flickr,
        
        'vcr': infer_minigptv2_vcr,
        'vsr': infer_minigptv2_vsr,
        
        'okvqa': infer_minigptv2_okvqa,
        'aokvqa': infer_minigptv2_aokvqa,
        'sqa': infer_minigptv2_sqa,

        'mme': infer_minigptv2_mme,
        'mmbench': infer_minigptv2_mmbench,
        'seedbench': infer_minigptv2_seedbench,

        'mmhalbench': infer_minigptv2_mmhalbench,
        'haloquest': infer_minigptv2_haloquest,
    },
}


def vlm_getter(vlm_name, device, dataset, cache_dir=None):
    model, processor = vlm_loader_map[vlm_name](device=device, cache_dir=cache_dir)
    infer_vlm = vlm_run_map[vlm_name][dataset]
    return model, processor, infer_vlm
