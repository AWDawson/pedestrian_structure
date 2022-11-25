rm -rf work/cls_pred.txt
rm -rf work/test_det_results
python test_det.py
python test_cls.py
python work/get_pred.py
python work/pr.py
python work/cls_acc.py
