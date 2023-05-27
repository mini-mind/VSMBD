from collections import defaultdict
from typing import Dict

import numpy as np
import torch
import torchmetrics
from sklearn.metrics import (
    average_precision_score,
    recall_score,
    roc_auc_score,
)

import torch
from torch.nn import functional as F
    
# wrap metrics
class Metrics(torchmetrics.Metric):
    def __init__(self, vno_shot_bound):
        super().__init__(compute_on_step=False, dist_sync_on_step=True)
        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("tn", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("probs", default=[])
        self.add_state("gts", default=[])
        self.add_state("vnos", default=[])
        self.add_state("sids", default=[])

        self.eps = 1e-5
        self.threshold = 0.5
        self.shot_bound = vno_shot_bound
        self.f1_metric = torchmetrics.classification.F1Score(compute_on_step=False, dist_sync_on_step=True)
        
    def update(self, vnos, sids, outputs, gts):
        probs = F.softmax(outputs.cpu(), dim=1)[:, 1]
        gt_one = gts == 1
        gt_zero = gts == 0
        pred_one = probs >= self.threshold
        pred_zero = probs < self.threshold
        self.tp += (gt_one * pred_one).sum()
        self.fp += (gt_zero * pred_one).sum()
        self.tn += (gt_zero * pred_zero).sum()
        self.fn += (gt_one * pred_zero).sum()
        for vno, sid, prob, gt in zip(vnos, sids, probs, gts):
            self.vnos.append(vno.detach().clone())
            self.sids.append(sid.detach().clone())
            self.probs.append(prob)
            self.gts.append(gt)
        self.f1_metric.update(probs, gts)
        
    def compute(self):
        score = {}
        vnos = torch.stack(self.vnos)
        probs = torch.stack(self.probs).cpu().numpy()
        gts = torch.stack(self.gts).cpu().numpy()
        # acc
        score["acc1"] = 100.0 * self.tp / (self.tp + self.fn + self.eps)
        score["acc0"] = 100.0 * self.tn / (self.fp + self.tn + self.eps)
        score["acc"] = 100.0 * (self.tp + self.tn) / (self.tp + self.fn + self.fp + self.tn + self.eps)
        # mAP
        all_vnos = list(set(list(vnos.numpy())))
        mAP = []
        for vno in all_vnos:
            index = np.where(vnos.numpy()==vno)[0]
            ap = average_precision_score(np.nan_to_num(gts[index]), np.nan_to_num(probs[index]))
            mAP.append(round(ap,2))
        mAP = np.mean(mAP)
        score["mAP"] = mAP * 100.0
        # ap, auc, f1
        ap = average_precision_score(np.nan_to_num(gts), np.nan_to_num(probs))
        auc = roc_auc_score(np.nan_to_num(gts), np.nan_to_num(probs))
        f1 = self.f1_metric.compute()
        score["ap"] = ap * 100.0
        score["auc"] = auc * 100.0
        score["f1"] = f1 * 100.0
        # r, miou
        recall, recall_at_3s, miou = self.compute_movienet()
        score["recall"] = recall * 100.0
        score["recall_at_3s"] = recall_at_3s * 100.0
        score["mIoU"] = miou * 100.0
        score = {k:(v.item() if isinstance(v, torch.Tensor) else v) for k,v in score.items()}
        return score
    
    def reset(self):
        super().reset()
        self.f1_metric.reset()
        
    def compute_movienet(self) -> torch.Tensor:
        vnos = torch.stack(self.vnos)
        sids = torch.stack(self.sids)
        probs = torch.stack(self.probs)
        gts = torch.stack(self.gts)

        result = defaultdict(dict)
        for vnox, sid, prob, gt in zip(vnos, sids, probs, gts):
            result[vnox.item()][sid.item()] = {
                "pred": 1 if prob>self.threshold else 0,
                "gt": gt.item(),
            }

        # compute exact recall
        recall = self._compute_exact_recall(result)
        recall_at_second = self._compute_recall_at_second(result)
        miou = self._compute_mIoU(result)

        del result  # recall, recall_one, pred, gt, preds, gts
        # torch.cuda.empty_cache()
        return recall, recall_at_second, miou

    def _compute_exact_recall(self, result):
        recall = []
        for _, result_dict_one in result.items():
            preds, gts = [], []
            for _, item in result_dict_one.items():
                pred = int(item.get("pred"))
                gt = int(item.get("gt"))
                preds.append(pred)
                gts.append(gt)
            recall_one = recall_score(gts, preds, average="binary")
            recall.append(recall_one)
        recall = np.mean(recall)
        return recall

    def _compute_recall_at_second(self, result, num_neighbor_shot=5, threshold=3):
        recall = []
        for vno, result_dict_one in result.items():
            shot_list = self.shot_bound[vno]

            cont_one, total_one = 0, 0
            for shotid, item in result_dict_one.items():
                gt = int(item.get("gt"))
                shot_time = int(shot_list[int(shotid)][1])
                if gt != 1:
                    continue

                total_one += 1
                for ind in range(0 - num_neighbor_shot, 1 + num_neighbor_shot):
                    shotid_cp = shotid + ind
                    if shotid_cp < 0 or (shotid_cp >= len(shot_list)):
                        continue
                    shot_time_cp = int(shot_list[shotid_cp][1])
                    item_cp = result_dict_one.get(shotid_cp)
                    if item_cp is None:
                        continue
                    else:
                        pred = item_cp.get("pred")
                        # FPS == 24
                        gap_time = np.abs(shot_time_cp - shot_time) / 24
                        if gt == pred and gap_time < threshold:
                            cont_one += 1
                            break
                            
            recall_one = cont_one / (total_one + self.eps)
            recall.append(recall_one)
        recall = np.mean(recall)
        return recall

    def _compute_mIoU(self, result):
        mious = []
        for vno, result_dict_one in result.items():
            shot_list = self.shot_bound[vno]
            gt_dict_one, pred_dict_one = {}, {}
            for shotid, item in result_dict_one.items():
                gt_dict_one.update({shotid: item.get("gt")})
                pred_dict_one.update({shotid: item.get("pred")})
            gt_pair_list = self.get_pair_list(gt_dict_one)
            pred_pair_list = self.get_pair_list(pred_dict_one)
            if pred_pair_list is None:
                mious.append(0)
                continue
            gt_scene_list = self.get_scene_list(gt_pair_list, shot_list)
            pred_scene_list = self.get_scene_list(pred_pair_list, shot_list)
            if gt_scene_list is None or pred_scene_list is None:
                return None
            miou1 = self.cal_miou(gt_scene_list, pred_scene_list)
            miou2 = self.cal_miou(pred_scene_list, gt_scene_list)
            mious.append(np.mean([miou1, miou2]))

        mious = np.mean(mious)
        return mious

    def get_scene_list(self, pair_list, shot_list):
        scene_list = []
        if pair_list is None:
            return None
        for item in pair_list:
            start = int(shot_list[int(item[0])][0])
            end = int(shot_list[int(item[-1])][1])
            scene_list.append((start, end))
        return scene_list

    def cal_miou(self, gt_scene_list, pred_scene_list):
        mious = []
        for gt_scene_item in gt_scene_list:
            rats = []
            for pred_scene_item in pred_scene_list:
                rat = self.getRatio(pred_scene_item, gt_scene_item)
                rats.append(rat)
            mious.append(np.max(rats))
        miou = np.mean(mious)
        return miou

    def getRatio(self, interval_1, interval_2):
        interaction = self.getIntersection(interval_1, interval_2)
        if interaction == 0:
            return 0
        else:
            return interaction / self.getUnion(interval_1, interval_2)

    def getIntersection(self, interval_1, interval_2):
        assert interval_1[0] < interval_1[1], "start frame is bigger than end frame."
        assert interval_2[0] < interval_2[1], "start frame is bigger than end frame."
        start = max(interval_1[0], interval_2[0])
        end = min(interval_1[1], interval_2[1])
        if start < end:
            return end - start
        return 0

    def getUnion(self, interval_1, interval_2):
        assert interval_1[0] < interval_1[1], "start frame is bigger than end frame."
        assert interval_2[0] < interval_2[1], "start frame is bigger than end frame."
        start = min(interval_1[0], interval_2[0])
        end = max(interval_1[1], interval_2[1])
        return end - start

    def get_pair_list(self, anno_dict):
        sort_anno_dict_key = sorted(anno_dict.keys())
        tmp = 0
        tmp_list = []
        tmp_label_list = []
        anno_list = []
        anno_label_list = []
        for key in sort_anno_dict_key:
            value = anno_dict.get(key)
            tmp += value
            tmp_list.append(key)
            tmp_label_list.append(value)
            if tmp == 1:
                anno_list.append(tmp_list)
                anno_label_list.append(tmp_label_list)
                tmp = 0
                tmp_list = []
                tmp_label_list = []
                continue
        if len(anno_list) == 0:
            return None
        while [] in anno_list:
            anno_list.remove([])
        tmp_anno_list = [anno_list[0]]
        pair_list = []
        for ind in range(len(anno_list) - 1):
            cont_count = int(anno_list[ind + 1][0]) - int(anno_list[ind][-1])
            if cont_count > 1:
                pair_list.extend(tmp_anno_list)
                tmp_anno_list = [anno_list[ind + 1]]
                continue
            tmp_anno_list.append(anno_list[ind + 1])
        pair_list.extend(tmp_anno_list)
        return pair_list