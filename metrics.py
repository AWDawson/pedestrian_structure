from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


def check_metric_vaild(y_pred, y_true):
    if y_true.min() == y_true.max() == 0:   # precision
        return False
    if y_pred.min() == y_pred.max() == 0:   # recall
        return False
    return True


def GetIndexes(imgnames):
    f = open('./dataset/test_data.txt', 'r') # eval_imgname_QAv2    eval_imgname_QAv2_0711
    dic = {}
    for line in f.readlines():
        line = line.strip()
        name, attri = line.split(' ')
        name = name.split('/')[-1]
        dic[name] = attri
    
    indexes = {}
    for index, i in enumerate(imgnames):
        i = i.split('/')[-1]
        if indexes.get(dic[i], -1) == -1:
            indexes[dic[i]] = [index]
        else: indexes[dic[i]].append(index)
    return indexes


def attris_eval(labels_tensor, preds_tensor, imgnames):
        # attribute_list = [
        # '朝向：正', '朝向：背', '朝向：左', '朝向：右',
        # '性别：男', '性别：女', '性别：不确定',
        # '年龄：0-18', '年龄：18-55','年龄：>55', '年龄：不确定',
        # '上身颜色：其他', '上身颜色：黑', '上身颜色：白', '上身颜色：灰', '上身颜色：红',
        #         '上身颜色：黄', '上身颜色：绿', '上身颜色：蓝', '上身颜色：紫',
        #         '上身颜色：棕', '上身颜色：粉', '上身颜色：橙',
        # '下身颜色：其他', '下身颜色：黑', '下身颜色：白', '下身颜色：灰', '下身颜色：红',
        #         '下身颜色：黄', '下身颜色：绿', '下身颜色：蓝', '下身颜色：紫',
        #         '下身颜色：棕', '下身颜色：粉', '下身颜色：橙'
        # ]
        attribute_list = [
            '性别: 女', '年龄：>60', '年龄：18-60', '年龄：<18',
            '朝向：前', '朝向：侧', '朝向：背',
            'hat', 'glasses', 'HandBag', 'ShoulderBag', 'BackPack',
            'HoldObjectsInFront', 'ShortSleeve', 'LongSleeve',
            'UpperStride', 'UpperLogo', 'UpperPlaid', 'UpperSplice',
            'LowerStripe', 'LowerPattern', 'LongCoat', 'Trousers',
            'Shorts', 'Skirt&Dress', 'boots'
        ]
        #print(attribute_list)
        # indexes = GetIndexes(imgnames)
        # print(imgnames)

        preds_tensor = np.where(preds_tensor > 0.5, 1, 0)
        # Evaluation.
        accuracy_list = []
        precision_list = []
        recall_list = []
        f1_score_list = []
        average_precision = 0.0
        average_recall = 0.0
        average_f1score = 0.0
        valid_count = 0
        # for i, name in enumerate(attribute_list):
            # if '朝向' in name:
            #     ind = indexes['direction']
            #     # ind = indexes['gender']
            #     print(ind)
            # elif '性别' in name:
            #     ind = indexes['gender']
            # elif '年龄' in name:
            #     ind = indexes['age']
            #     # ind = indexes['gender']
            # elif '上身颜色' in name:
            #     ind = indexes['color_top']
            # elif '下身颜色' in name:
            #     ind = indexes['color_bottom']
                # ind = indexes['gender']
            # y_true, y_pred = labels_tensor[ind, i], preds_tensor[ind, i]
        for i in range(26):
            y_true, y_pred = labels_tensor[:, i], preds_tensor[:, i]
            accuracy_list.append(accuracy_score(y_true, y_pred))
            if check_metric_vaild(y_pred, y_true):    # exclude ill-defined cases
                precision_list.append(precision_score(y_true, y_pred, average='binary'))
                recall_list.append(recall_score(y_true, y_pred, average='binary'))
                f1_score_list.append(f1_score(y_true, y_pred, average='binary'))
                average_precision += precision_list[-1]
                average_recall += recall_list[-1]
                average_f1score += f1_score_list[-1]
                valid_count += 1
            else:
                precision_list.append(-1)
                recall_list.append(-1)
                f1_score_list.append(-1)
            # precision_list.append(precision_score(y_true, y_pred, average='binary'))
            # recall_list.append(recall_score(y_true, y_pred, average='binary'))
            # f1_score_list.append(f1_score(y_true, y_pred, average='binary'))
            # average_precision += precision_list[-1]
            # average_recall += recall_list[-1]
            # average_f1score += f1_score_list[-1]
            # valid_count += 1

        average_acc = np.mean(accuracy_list)
        average_precision = average_precision / valid_count
        average_recall = average_recall / valid_count
        average_f1score = average_f1score / valid_count

        ######################################################################
        # Print
        # ---------
        print("\n"
            "The Precision, Recall and F-score are ignored for some ill-defined cases."
            "\n")

        if True:
            from prettytable import PrettyTable
            table = PrettyTable(['attribute', 'accuracy', 'precision', 'recall', 'f1 score'])
            for i, name in enumerate(attribute_list):
                # if i in [4,5,12,13,14,15,16,17,18,19,20,21,22]:
                if True:
                    table.add_row([name,
                        '%.3f' % accuracy_list[i],
                        '%.3f' % precision_list[i] if precision_list[i] >= 0.0 else '-',
                        '%.3f' % recall_list[i] if recall_list[i] >= 0.0 else '-',
                        '%.3f' % f1_score_list[i] if f1_score_list[i] >= 0.0 else '-',
                        ])
            print(table)


        print('Average accuracy: {:.4f}'.format(average_acc))
        # print('Average precision: {:.4f}'.format(average_precision))
        # print('Average recall: {:.4f}'.format(average_recall))
        print('Average f1 score: {:.4f}'.format(average_f1score))
