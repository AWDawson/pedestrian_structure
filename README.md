## 环境
- paddle2.2

## 预训练模型
- r18vd的预训练：wget 10.12.72.157:8222/ResNet18_vd_pretrained.pdparams
- 其他resnet系列的预训练使用paddleclas官方提供：https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNetx_vd_ssld_pretrained.pdparams
- 预训练模型放到 /root/.paddleclas/weights 路径下

## 训练
- 将数据集中的all文件夹mv至dataset中
- 训练属性分类分支，修改/yaml/resnet.yaml中的backbone为期望的骨干网络，修改continue_train为False，Unfreeze_Epoch为10，task为cls，save_dir为模型保存路径
- 配置train.sh，bash train.sh，得到模型M
- 联合训练属性分类和人头检测，修改/yaml/resnet.yaml中的continue_train为True，last_params为M路径，Unfreeze_Epoch为50，task为cls_det，save_dir为新的模型保存路径
- 配置train.sh，bash train.sh，得到模型M’

## 测试
- 修改test_cls.py和test_det.py的model_path为M’路径，backbone为期望的骨干网络
- bash test.sh，得到评测指标

## Export
- demo: python export_model_pede.py -m resnet18_vd -p work/r18_trained/Epoch60-Total_Loss0.2959-Val_Loss0.7655 -o output_path
- code_infer.ipynb 看可视化
# pedestrian_structure
