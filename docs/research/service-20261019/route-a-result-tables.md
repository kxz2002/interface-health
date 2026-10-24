## 逐seed与阈值

|路线|seed|AUROC|AP|q99阈值|TP/FP/FN/TN|
|---|---|---|---|---|---|
|prior_gru|1|0.824682|0.690464|2.245197|251/30/194/2822|
|prior_gru|2|0.821857|0.681168|2.227993|227/34/218/2818|
|prior_gru|42|0.825372|0.668550|2.226117|220/42/225/2810|
|bert|1|0.807610|0.671879|2.210598|226/44/219/2808|
|bert|2|0.812706|0.659399|2.310635|217/33/228/2819|
|bert|42|0.819390|0.669782|2.335319|221/35/224/2817|
|bert_tuned|1|0.810836|0.671382|2.240443|219/35/226/2817|
|bert_tuned|2|0.812562|0.668408|2.273468|215/33/230/2819|
|bert_tuned|42|0.813865|0.670403|2.294920|221/33/224/2819|
|no_prior|1|0.644470|0.489690|2.346629|153/44/292/2808|
|no_prior|2|0.652116|0.486229|2.525796|123/38/322/2814|
|no_prior|42|0.626979|0.467183|2.442880|142/48/303/2804|
|global_prior|1|0.641608|0.478302|2.385224|137/45/308/2807|
|global_prior|2|0.648374|0.477435|2.563428|113/37/332/2815|
|global_prior|42|0.651479|0.510171|2.555640|157/37/288/2815|
|bert_zero|1|0.823596|0.676904|2.214150|232/34/213/2818|
|bert_zero|2|0.823956|0.680095|2.273297|231/35/214/2817|
|bert_zero|42|0.823197|0.666724|2.357471|222/34/223/2818|
|bert_reverse|1|0.809089|0.672449|2.205220|229/44/216/2808|
|bert_reverse|2|0.812738|0.659246|2.296450|217/35/228/2817|
|bert_reverse|42|0.818929|0.668636|2.334021|219/36/226/2816|

## 按Case分解

下列recall为三seed均值；MSE/BCE为各seed组内逐窗分量均值的均值。各组ID完全一致。单类Case的AUROC/AP均为null，故不以Case内AUC排名。正常Case仅报告误报率。

### prior_gru

|Case|窗数|q99 recall|正常误报率|value MSE|availability BCE|
|---|---|---|---|---|---|
|Lv_D_CONNECTION_POOL_exhaustion_20261017T102058Z_7b28ab23f0694e7f_em|117|0.0028|null|0.2163|0.1848|
|Lv_D_TRANSACTION_timeout_20261017T042021Z_3fe5039544954db2_em|138|0.9952|null|3.3660|2.3869|
|Lv_P_CPU_preserve_20261017T154808Z_021eaa51bbcb4b60_em|39|0.4274|null|1.9437|0.1308|
|Lv_P_DISKIO_preserve_20261018T013644Z_707c577ddacf4703_em|38|0.1140|null|1.6883|0.1015|
|Lv_P_NETLOSS_preserve_20261017T145542Z_8a5e5b4c67c64664_em|34|1.0000|null|7.0403|2.0815|
|Lv_S_DNSFAIL_preserve_no_order_20261018T140118Z_81f642e966444fe0_em|39|0.0000|null|0.3656|0.1372|
|Lv_S_HTTPABORT_preserve_20261017T072058Z_daa34643ee434ee4_em|34|1.0000|null|0.0203|3.3653|
|Lv_S_KILLPOD_gateway_20261017T093533Z_94814e0f78844917_em|1|1.0000|null|2.0147|1.7807|
|Lv_S_KILLPOD_order_20261018T144636Z_74933cf4bb0d48cc_em|3|1.0000|null|8.1796|2.7210|
|Lv_S_KILLPOD_preserve_20261018T022220Z_aa928eae7af646e3_em|2|1.0000|null|14.2012|1.7116|
|Normal_20261018T153211Z_e448a74ed7104ebd|2852|null|0.0124|0.3280|0.1500|
### bert

|Case|窗数|q99 recall|正常误报率|value MSE|availability BCE|
|---|---|---|---|---|---|
|Lv_D_CONNECTION_POOL_exhaustion_20261017T102058Z_7b28ab23f0694e7f_em|117|0.0000|null|0.2339|0.1927|
|Lv_D_TRANSACTION_timeout_20261017T042021Z_3fe5039544954db2_em|138|0.9928|null|3.5268|2.4795|
|Lv_P_CPU_preserve_20261017T154808Z_021eaa51bbcb4b60_em|39|0.2393|null|1.8344|0.1363|
|Lv_P_DISKIO_preserve_20261018T013644Z_707c577ddacf4703_em|38|0.0263|null|1.5037|0.0939|
|Lv_P_NETLOSS_preserve_20261017T145542Z_8a5e5b4c67c64664_em|34|1.0000|null|7.4855|2.2466|
|Lv_S_DNSFAIL_preserve_no_order_20261018T140118Z_81f642e966444fe0_em|39|0.0000|null|0.3325|0.1323|
|Lv_S_HTTPABORT_preserve_20261017T072058Z_daa34643ee434ee4_em|34|1.0000|null|0.0254|3.3243|
|Lv_S_KILLPOD_gateway_20261017T093533Z_94814e0f78844917_em|1|1.0000|null|2.2780|1.5863|
|Lv_S_KILLPOD_order_20261018T144636Z_74933cf4bb0d48cc_em|3|1.0000|null|8.5672|2.8268|
|Lv_S_KILLPOD_preserve_20261018T022220Z_aa928eae7af646e3_em|2|1.0000|null|12.8740|1.8125|
|Normal_20261018T153211Z_e448a74ed7104ebd|2852|null|0.0131|0.3632|0.1510|
### bert_tuned

|Case|窗数|q99 recall|正常误报率|value MSE|availability BCE|
|---|---|---|---|---|---|
|Lv_D_CONNECTION_POOL_exhaustion_20261017T102058Z_7b28ab23f0694e7f_em|117|0.0000|null|0.2134|0.1862|
|Lv_D_TRANSACTION_timeout_20261017T042021Z_3fe5039544954db2_em|138|0.9976|null|3.4742|2.5027|
|Lv_P_CPU_preserve_20261017T154808Z_021eaa51bbcb4b60_em|39|0.1453|null|1.7287|0.1296|
|Lv_P_DISKIO_preserve_20261018T013644Z_707c577ddacf4703_em|38|0.0263|null|1.4729|0.0948|
|Lv_P_NETLOSS_preserve_20261017T145542Z_8a5e5b4c67c64664_em|34|1.0000|null|7.1752|2.2508|
|Lv_S_DNSFAIL_preserve_no_order_20261018T140118Z_81f642e966444fe0_em|39|0.0000|null|0.3239|0.1289|
|Lv_S_HTTPABORT_preserve_20261017T072058Z_daa34643ee434ee4_em|34|1.0000|null|0.0265|3.4503|
|Lv_S_KILLPOD_gateway_20261017T093533Z_94814e0f78844917_em|1|1.0000|null|2.3271|1.9485|
|Lv_S_KILLPOD_order_20261018T144636Z_74933cf4bb0d48cc_em|3|1.0000|null|8.0040|2.7978|
|Lv_S_KILLPOD_preserve_20261018T022220Z_aa928eae7af646e3_em|2|1.0000|null|13.3715|2.0460|
|Normal_20261018T153211Z_e448a74ed7104ebd|2852|null|0.0118|0.3422|0.1496|
### no_prior

|Case|窗数|q99 recall|正常误报率|value MSE|availability BCE|
|---|---|---|---|---|---|
|Lv_D_CONNECTION_POOL_exhaustion_20261017T102058Z_7b28ab23f0694e7f_em|117|0.0000|null|0.2215|0.1914|
|Lv_D_TRANSACTION_timeout_20261017T042021Z_3fe5039544954db2_em|138|0.5290|null|3.3296|0.7968|
|Lv_P_CPU_preserve_20261017T154808Z_021eaa51bbcb4b60_em|39|0.6239|null|2.0225|0.3743|
|Lv_P_DISKIO_preserve_20261018T013644Z_707c577ddacf4703_em|38|0.1228|null|1.7379|0.2895|
|Lv_P_NETLOSS_preserve_20261017T145542Z_8a5e5b4c67c64664_em|34|0.8922|null|7.0995|1.0291|
|Lv_S_DNSFAIL_preserve_no_order_20261018T140118Z_81f642e966444fe0_em|39|0.0000|null|0.3569|0.1487|
|Lv_S_HTTPABORT_preserve_20261017T072058Z_daa34643ee434ee4_em|34|0.0294|null|0.0361|0.1776|
|Lv_S_KILLPOD_gateway_20261017T093533Z_94814e0f78844917_em|1|1.0000|null|2.5622|0.8840|
|Lv_S_KILLPOD_order_20261018T144636Z_74933cf4bb0d48cc_em|3|1.0000|null|8.3445|0.6756|
|Lv_S_KILLPOD_preserve_20261018T022220Z_aa928eae7af646e3_em|2|1.0000|null|13.3614|0.9718|
|Normal_20261018T153211Z_e448a74ed7104ebd|2852|null|0.0152|0.3660|0.1647|
### global_prior

|Case|窗数|q99 recall|正常误报率|value MSE|availability BCE|
|---|---|---|---|---|---|
|Lv_D_CONNECTION_POOL_exhaustion_20261017T102058Z_7b28ab23f0694e7f_em|117|0.0000|null|0.2218|0.1925|
|Lv_D_TRANSACTION_timeout_20261017T042021Z_3fe5039544954db2_em|138|0.5217|null|3.4437|0.8194|
|Lv_P_CPU_preserve_20261017T154808Z_021eaa51bbcb4b60_em|39|0.5641|null|2.0583|0.4194|
|Lv_P_DISKIO_preserve_20261018T013644Z_707c577ddacf4703_em|38|0.0965|null|1.6831|0.3399|
|Lv_P_NETLOSS_preserve_20261017T145542Z_8a5e5b4c67c64664_em|34|0.9118|null|6.7177|0.7797|
|Lv_S_DNSFAIL_preserve_no_order_20261018T140118Z_81f642e966444fe0_em|39|0.0000|null|0.3616|0.1431|
|Lv_S_HTTPABORT_preserve_20261017T072058Z_daa34643ee434ee4_em|34|0.0294|null|0.0258|0.1699|
|Lv_S_KILLPOD_gateway_20261017T093533Z_94814e0f78844917_em|1|1.0000|null|2.6228|0.8256|
|Lv_S_KILLPOD_order_20261018T144636Z_74933cf4bb0d48cc_em|3|1.0000|null|7.9534|0.4906|
|Lv_S_KILLPOD_preserve_20261018T022220Z_aa928eae7af646e3_em|2|1.0000|null|12.7387|0.6234|
|Normal_20261018T153211Z_e448a74ed7104ebd|2852|null|0.0139|0.3458|0.1646|
### bert_zero

|Case|窗数|q99 recall|正常误报率|value MSE|availability BCE|
|---|---|---|---|---|---|
|Lv_D_CONNECTION_POOL_exhaustion_20261017T102058Z_7b28ab23f0694e7f_em|117|0.0000|null|0.2144|0.1888|
|Lv_D_TRANSACTION_timeout_20261017T042021Z_3fe5039544954db2_em|138|0.9903|null|3.3906|2.3169|
|Lv_P_CPU_preserve_20261017T154808Z_021eaa51bbcb4b60_em|39|0.3932|null|1.9377|0.1364|
|Lv_P_DISKIO_preserve_20261018T013644Z_707c577ddacf4703_em|38|0.0614|null|1.5135|0.1057|
|Lv_P_NETLOSS_preserve_20261017T145542Z_8a5e5b4c67c64664_em|34|1.0000|null|6.6688|2.0493|
|Lv_S_DNSFAIL_preserve_no_order_20261018T140118Z_81f642e966444fe0_em|39|0.0000|null|0.3843|0.1321|
|Lv_S_HTTPABORT_preserve_20261017T072058Z_daa34643ee434ee4_em|34|1.0000|null|0.0220|3.2339|
|Lv_S_KILLPOD_gateway_20261017T093533Z_94814e0f78844917_em|1|1.0000|null|2.1984|1.7976|
|Lv_S_KILLPOD_order_20261018T144636Z_74933cf4bb0d48cc_em|3|1.0000|null|7.3602|2.6700|
|Lv_S_KILLPOD_preserve_20261018T022220Z_aa928eae7af646e3_em|2|1.0000|null|12.9875|1.7058|
|Normal_20261018T153211Z_e448a74ed7104ebd|2852|null|0.0120|0.3296|0.1492|
### bert_reverse

|Case|窗数|q99 recall|正常误报率|value MSE|availability BCE|
|---|---|---|---|---|---|
|Lv_D_CONNECTION_POOL_exhaustion_20261017T102058Z_7b28ab23f0694e7f_em|117|0.0000|null|0.2339|0.1926|
|Lv_D_TRANSACTION_timeout_20261017T042021Z_3fe5039544954db2_em|138|0.9928|null|3.5095|2.4383|
|Lv_P_CPU_preserve_20261017T154808Z_021eaa51bbcb4b60_em|39|0.2479|null|1.8596|0.1364|
|Lv_P_DISKIO_preserve_20261018T013644Z_707c577ddacf4703_em|38|0.0263|null|1.5199|0.0940|
|Lv_P_NETLOSS_preserve_20261017T145542Z_8a5e5b4c67c64664_em|34|1.0000|null|7.4745|2.1311|
|Lv_S_DNSFAIL_preserve_no_order_20261018T140118Z_81f642e966444fe0_em|39|0.0000|null|0.3290|0.1328|
|Lv_S_HTTPABORT_preserve_20261017T072058Z_daa34643ee434ee4_em|34|1.0000|null|0.0247|3.3176|
|Lv_S_KILLPOD_gateway_20261017T093533Z_94814e0f78844917_em|1|1.0000|null|2.5342|1.6055|
|Lv_S_KILLPOD_order_20261018T144636Z_74933cf4bb0d48cc_em|3|1.0000|null|8.5689|2.9823|
|Lv_S_KILLPOD_preserve_20261018T022220Z_aa928eae7af646e3_em|2|1.0000|null|13.5667|1.6041|
|Normal_20261018T153211Z_e448a74ed7104ebd|2852|null|0.0134|0.3632|0.1512|

## 按服务排序

各列为三seed AUROC均值；单类服务保留null。服务内排序与使用全局阈值的召回不可等同。

|服务|窗数|prior_gru|bert|bert_tuned|no_prior|global_prior|bert_zero|bert_reverse|
|---|---|---|---|---|---|---|---|---|
|ts-assurance-service|118|null|null|null|null|null|null|null|
|ts-auth-service|105|null|null|null|null|null|null|null|
|ts-basic-service|119|null|null|null|null|null|null|null|
|ts-cancel-service|99|null|null|null|null|null|null|null|
|ts-config-service|119|null|null|null|null|null|null|null|
|ts-contacts-service|119|null|null|null|null|null|null|null|
|ts-execute-service|95|null|null|null|null|null|null|null|
|ts-gateway-service|120|0.9944|0.9972|1.0000|0.9944|0.9888|0.9972|1.0000|
|ts-inside-payment-service|118|null|null|null|null|null|null|null|
|ts-notification-service|96|null|null|null|null|null|null|null|
|ts-order-other-service|119|null|null|null|null|null|null|null|
|ts-order-service|197|0.6580|0.6528|0.6496|0.6563|0.6578|0.6562|0.6540|
|ts-payment-service|119|null|null|null|null|null|null|null|
|ts-preserve-other-service|98|null|null|null|null|null|null|null|
|ts-preserve-service|376|0.8018|0.8095|0.7994|0.5759|0.5786|0.7963|0.8106|
|ts-price-service|119|null|null|null|null|null|null|null|
|ts-rebook-service|101|null|null|null|null|null|null|null|
|ts-route-service|119|null|null|null|null|null|null|null|
|ts-seat-service|119|null|null|null|null|null|null|null|
|ts-security-service|119|null|null|null|null|null|null|null|
|ts-station-service|119|null|null|null|null|null|null|null|
|ts-train-service|119|null|null|null|null|null|null|null|
|ts-travel-service|156|0.9994|1.0000|1.0000|0.9781|0.9752|0.9998|1.0000|
|ts-travel2-service|118|null|null|null|null|null|null|null|
|ts-user-service|191|0.7323|0.7045|0.7292|0.4097|0.4015|0.7310|0.6999|

逐seed主指标、计数、阈值和manifest hash见同目录`route-a-results.json`；逐seed服务/Case计数、分量分布与配置保留在对应远端运行目录，可用汇总脚本重建。
