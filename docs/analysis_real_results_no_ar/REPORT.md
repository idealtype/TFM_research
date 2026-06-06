# Non-AR Real-Data Result Summary


Excluded: `/home/sia2/project/6.1AR_temp` AR h96 experiments. Metrics are real LOTSA+ETT only.


## Overall Leaderboard


| family | short | setting | mean_mae | median_mae | n |
| --- | --- | --- | --- | --- | --- |
| baseline | TimesFM | timesfm_zeroshot | 0.409603 | 0.367596 | 32 |
| soft_mask | soft warm s10 old | soft_warm_s10_oldloss_best | 0.504205 | 0.468409 | 32 |
| soft_mask | soft base | soft_syn_all_real | 0.506094 | 0.547303 | 32 |
| soft_mask | soft cycle b1024 | soft_inter_s13_res13_b1024 | 0.512634 | 0.534876 | 32 |
| nogate_soft_mask | nogate warm | nogate_warm_s13_b1024_coeffl1 | 0.515410 | 0.505850 | 32 |
| soft_mask | soft warm s50 old | soft_warm_s50_oldloss | 0.516099 | 0.492707 | 32 |
| nogate_soft_mask | nogate cycle | nogate_inter_s13_res13_b1024_coeffl1 | 0.517711 | 0.497878 | 32 |
| soft_mask | soft warm s50 TS | soft_warm_s50_ts_loss | 0.518661 | 0.511611 | 32 |
| soft_mask | soft warm m5000 | soft_warm_s13_b1024_m5000 | 0.520006 | 0.527170 | 32 |
| hard_mask | hard warm old | hard_warm_mix_old | 0.520657 | 0.506257 | 32 |
| soft_mask | soft cycle s50 TS | soft_inter_s50_res50_ts_loss | 0.522895 | 0.563502 | 32 |
| soft_mask | soft cycle s50 old | soft_inter_s50_res50_oldloss | 0.525622 | 0.559641 | 32 |
| soft_mask | soft warm ckpt err | soft_warm_checkpoint_error | 0.527971 | 0.515028 | 32 |
| hard_mask | hard base | hard_syn_and_alldata | 0.555722 | 0.578488 | 32 |
| soft_mask | soft cycle old | soft_interleaved_old | 0.565629 | 0.580948 | 32 |
| soft_mask | soft cycle m5000 | soft_inter_s13_res13_b1024_m5000 | 0.575103 | 0.588290 | 32 |


## Focused Key Runs


| family | short | setting | mean_mae | median_mae | n |
| --- | --- | --- | --- | --- | --- |
| baseline | TimesFM | timesfm_zeroshot | 0.409603 | 0.367596 | 32 |
| soft_mask | soft warm s10 old | soft_warm_s10_oldloss_best | 0.504205 | 0.468409 | 32 |
| soft_mask | soft base | soft_syn_all_real | 0.506094 | 0.547303 | 32 |
| soft_mask | soft cycle b1024 | soft_inter_s13_res13_b1024 | 0.512634 | 0.534876 | 32 |
| nogate_soft_mask | nogate warm | nogate_warm_s13_b1024_coeffl1 | 0.515410 | 0.505850 | 32 |
| nogate_soft_mask | nogate cycle | nogate_inter_s13_res13_b1024_coeffl1 | 0.517711 | 0.497878 | 32 |
| hard_mask | hard warm old | hard_warm_mix_old | 0.520657 | 0.506257 | 32 |
| hard_mask | hard base | hard_syn_and_alldata | 0.555722 | 0.578488 | 32 |


## Residual Metrics Available


| family | short | setting | mean_mae | residual_gain | residual_std | residual_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| soft_mask | soft cycle b1024 | soft_inter_s13_res13_b1024 | 0.512634 | 0.141580 | 0.414732 | 0.490577 |
| nogate_soft_mask | nogate warm | nogate_warm_s13_b1024_coeffl1 | 0.515410 | 0.100725 | 0.422267 | 0.514404 |
| nogate_soft_mask | nogate cycle | nogate_inter_s13_res13_b1024_coeffl1 | 0.517711 | 0.132608 | 0.402310 | 0.461263 |
| soft_mask | soft warm m5000 | soft_warm_s13_b1024_m5000 | 0.520006 | 0.089786 | 0.420649 | 0.493486 |
| soft_mask | soft cycle m5000 | soft_inter_s13_res13_b1024_m5000 | 0.575103 | 0.088216 | 0.376815 | 0.502928 |


## Files


- Tables: `tables/`
- Plots: `plots/`
