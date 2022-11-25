# pedestrian_structure
## 环境
- paddle2.2

## 训练
- 准备好数据集，在/yaml/resnet_18.yaml中指定对应annotation_train_path和annotation_val_path的路径
- 训练属性分类分支，修改/yaml/resnet_18.yaml中的backbone为期望的骨干网络，修改continue_train为False，Unfreeze_Epoch为10，task为cls，save_dir为模型保存路径
- 配置train.sh，bash train.sh，得到模型M
- 联合训练属性分类和人头检测，修改/yaml/resnet_18.yaml中的continue_train为True，last_params为M路径，Unfreeze_Epoch为50，task为cls_det，save_dir为新的模型保存路径
- 配置train.sh，bash train.sh，得到模型M’

## 测试
- 修改test_cls.py和test_det.py的model_path为M’路径，backbone为期望的骨干网络
- bash test.sh，得到评测指标

## 导出
- demo: python export_model_pede.py -m resnet18_vd -p <model_save_path> -o output_path
- code_infer.ipynb 看可视化
