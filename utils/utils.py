import random
import numpy as np
import torch
import torch.nn.functional as F
import json


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class MetricsTracker:
    def __init__(self, args):
        self.samples = 0.0
        self.changed = 0.0
    
        self.corrects_base = 0.0
        self.corrects_tpt = 0.0

        self.top5_corrects_base = 0.0
        self.top5_corrects_tpt = 0.0
    
        self.cumulative_crossEntropy_base = 0.0
        self.cumulative_crossEntropy_tpt = 0.0
        
        self.crossEntropy = torch.nn.CrossEntropyLoss()

        self.sample_statistics = [
        {"target" : target, "tpt_improved_samples": [], "tpt_worsened_samples": [], "n_samples": 0}
            for target in range(args.nclasses)
        ]


    def update(self, i, view_img, output_base, output_tpt, target, writer, args):
        target_class = target.item()

        pred_base_conf, pred_base_class = torch.softmax(output_base, dim=1).max(1)
        pred_base_probs = torch.softmax(output_base, dim=1)
        base_crossEntropy = self.crossEntropy(output_base, target)

        pred_tpt_conf, pred_tpt_class = torch.softmax(output_tpt, dim=1).max(1)
        pred_tpt_probs = torch.softmax(output_tpt, dim=1)
        tpt_crossEntropy = self.crossEntropy(output_tpt, target)

        _, top5_pred_base = torch.topk(output_base, 5, dim=1)
        _, top5_pred_tpt = torch.topk(output_tpt, 5, dim=1)

        writer.add_scalar("confidence/Base", pred_base_conf, i)
        writer.add_scalar("confidence/TPT_coop", pred_tpt_conf, i)
        writer.add_scalar("confidence/difference(TPT-base)", pred_tpt_conf-pred_base_conf, i)

        self.cumulative_crossEntropy_base += base_crossEntropy
        self.cumulative_crossEntropy_tpt += tpt_crossEntropy
        writer.add_scalar("crossEntropy/Base", base_crossEntropy, i)
        writer.add_scalar("crossEntropy/TPT_coop", tpt_crossEntropy, i)
        writer.add_scalar("crossEntropy/difference(base-TPT)", base_crossEntropy-tpt_crossEntropy, i)

        true_dist = F.one_hot(target, num_classes=args.nclasses).float().to(args.device)
        kl_base = F.kl_div(pred_base_probs.log(), true_dist, reduction='batchmean')
        kl_tpt = F.kl_div(pred_tpt_probs.log(), true_dist, reduction='batchmean')
        writer.add_scalar("KL/Base_vs_True", kl_base.item(), i)
        writer.add_scalar("KL/TPT_vs_True", kl_tpt.item(), i)
        writer.add_scalar("KL/Base-TPT", kl_base.item()-kl_tpt.item(), i)

        self.corrects_base += pred_base_class.eq(target).sum().item()
        self.corrects_tpt += pred_tpt_class.eq(target).sum().item()
        
        self.top5_corrects_base += torch.sum(top5_pred_base == target.view(-1, 1)).float().item()
        self.top5_corrects_tpt += torch.sum(top5_pred_tpt == target.view(-1, 1)).float().item()

        self.samples += 1

        # Calculate current accuracy
        curr_base_acc = self.get_accuracy_base()
        curr_TPTcoop_acc = self.get_accuracy_tpt()
        curr_base_top5_acc = self.get_top5_accuracy_base()
        curr_TPTcoop_top5_acc = self.get_top5_accuracy_tpt()

        writer.add_scalar("AccuracyTOP1/Base", curr_base_acc, i)
        writer.add_scalar("AccuracyTOP1/TPT_coop", curr_TPTcoop_acc, i)
        writer.add_scalar("AccuracyTOP1/difference(TPT-base)", curr_TPTcoop_acc-curr_base_acc, i)

        writer.add_scalar("AccuracyTOP5/Base_Top5", curr_base_top5_acc, i)
        writer.add_scalar("AccuracyTOP5/TPT_coop_Top5", curr_TPTcoop_top5_acc, i)
        writer.add_scalar("AccuracyTOP5/difference_Top5(TPT-base)", curr_TPTcoop_top5_acc-curr_base_top5_acc, i)

        improvement = (
            pred_tpt_class.eq(target).sum().item()
            - pred_base_class.eq(target).sum().item()
        )
        writer.add_scalar("Samples/TPTimprovement", improvement, i)

        self.changed += improvement
        writer.add_scalar("Samples/TPTchanged", improvement, i)

        if improvement > 0:
            if args.save_imgs:
                writer.add_image(
                    f"Improved_Images/img_{i}:{target_class}-{args.classnames[target_class]}", view_img.squeeze(0), target_class, i
                )
            self.sample_statistics[target]["tpt_improved_samples"].append(i)
        elif improvement < 0:
            if args.save_imgs:
                writer.add_image(
                    f"Worsened_Images/img_{i}:{target_class}-{args.classnames[target_class]}", view_img.squeeze(0), target_class, i
                )
            self.sample_statistics[target]["tpt_worsened_samples"].append(i)

        self.sample_statistics[target]["n_samples"] += 1

    def write_info(self, writer, args):
        global_stat =  {
            "tpt_improved_samples": [],
            "tpt_worsened_samples": [],
            "n_samples": 0,
        }
        for target in range(args.nclasses):
            ntpt_improved = len(self.sample_statistics[target]["tpt_improved_samples"]) 
            ntpt_worsened = len(self.sample_statistics[target]["tpt_worsened_samples"]) 
            self.sample_statistics[target]["ntpt_improved"] = ntpt_improved
            self.sample_statistics[target]["ntpt_worsened"] = ntpt_worsened

            writer.add_scalar("Samples-PerClass/NImproved", ntpt_improved, target)
            writer.add_scalar("Samples-PerClass/NWorsened", ntpt_worsened, target)
            writer.add_scalar("Samples-PerClass/Nsamples", self.sample_statistics[target]["n_samples"], target) 
            writer.add_scalar("Samples-PerClass/ImprovedRatio", ntpt_improved/self.sample_statistics[target]["n_samples"] if self.sample_statistics[target]["n_samples"]!=0 else 0, target)
            writer.add_scalar("Samples-PerClass/WorsenedRatio", ntpt_worsened/self.sample_statistics[target]["n_samples"] if self.sample_statistics[target]["n_samples"]!=0 else 0, target)

            global_stat["tpt_improved_samples"] += self.sample_statistics[target]["tpt_improved_samples"]
            global_stat["tpt_worsened_samples"] += self.sample_statistics[target]["tpt_worsened_samples"]
            global_stat["n_samples"] += self.sample_statistics[target]["n_samples"]

        global_stat["ntpt_improved"] = len(global_stat["tpt_improved_samples"]) 
        global_stat["ntpt_worsened"] =  len(global_stat["tpt_worsened_samples"]) 
        writer.add_scalar("Samples-Global/NImproved", global_stat["ntpt_improved"])
        writer.add_scalar("Samples-Global/NWorsened", global_stat["ntpt_worsened"])
        writer.add_scalar("Samples-Global/ImprovedRatio", global_stat["ntpt_improved"]/global_stat["n_samples"] if global_stat["n_samples"]!=0 else 0 )
        writer.add_scalar("Samples-Global/WorsenedRatio", global_stat["ntpt_worsened"]/global_stat["n_samples"] if global_stat["n_samples"]!=0 else 0 )

        writer.add_text("statistics/perclass", json.dumps(self.sample_statistics, indent=4))
        writer.add_text("statistics/global", json.dumps(global_stat, indent=4))


        final_accuracy_base = self.get_accuracy_base()
        final_accuracy_tpt = self.get_accuracy_tpt()
        final_accuracy_base_top5 = self.get_top5_accuracy_base()
        final_accuracy_tpt_top5 = self.get_top5_accuracy_tpt()

        writer.add_scalar("AccuracyTOP1/Final_Base", final_accuracy_base, 0)
        writer.add_scalar("AccuracyTOP1/Final_TPT_coop", final_accuracy_tpt, 0) 
        
        writer.add_scalar("AccuracyTOP5/Final_Base_Top5", final_accuracy_base_top5, 0)
        writer.add_scalar("AccuracyTOP5/Final_TPT_coop_Top5", final_accuracy_tpt_top5, 0)

        writer.add_scalar("crossEntropy/Final_Base", self.cumulative_crossEntropy_base, 0)
        writer.add_scalar("crossEntropy/Final_TPT_coop", self.cumulative_crossEntropy_tpt, 0)



    def get_accuracy_base(self):
        return self.corrects_base / self.samples if self.samples > 0 else 0

    def get_accuracy_tpt(self):
        return self.corrects_tpt / self.samples if self.samples > 0 else 0
    
    def get_top5_accuracy_base(self):
        return self.top5_corrects_base / self.samples if self.samples > 0 else 0
        
    def get_top5_accuracy_tpt(self):
        return self.top5_corrects_tpt / self.samples if self.samples > 0 else 0