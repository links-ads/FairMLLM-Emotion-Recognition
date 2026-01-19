import numpy as np

class FairnessMetrics(object):

    def __init__(
            self,
            targets: np.ndarray,
            predictions: np.ndarray,
            sens_attr: str,
            sens_attr_values: np.ndarray,
        ):

        valid_mask = predictions != "Unknown"
        targets = targets[valid_mask]
        predictions = predictions[valid_mask]
        sens_attr_values = sens_attr_values[valid_mask]
        if len(targets) == 0:
            raise ValueError("Fairness - No valid predictions after filtering 'Unknown'")

        self.unique_classes = sorted(set(targets))
        class_to_idx = {cls: idx for idx, cls in enumerate(self.unique_classes)}
        
        unique_sens = sorted(set(sens_attr_values))
        sens_to_idx = {s: idx for idx, s in enumerate(unique_sens)}
        
        self.true_y = np.array([class_to_idx[y] for y in targets])
        self.pred_y = np.array([class_to_idx[y] for y in predictions])
        self.sens_attr = sens_attr
        self.sens_attr_values = np.array([sens_to_idx[s] for s in sens_attr_values])

        self.class_range = list(range(len(self.unique_classes)))
        self.y_hat = []
        self.yneq_hat = []
        for y_hat_idx in self.class_range:
            self.y_hat.append(self.pred_y == y_hat_idx) # class i
            self.yneq_hat.append(self.pred_y != y_hat_idx) # not class i
            
        self.sens_attr_range = list(set(self.sens_attr_values))
        self.s = []
        for s_idx in self.sens_attr_range:
            self.s.append(self.sens_attr_values == s_idx)

        self.y_s = []
        self.yneq_s = []
        for y_idx in self.class_range:
            self.y_s.append([])
            self.yneq_s.append([])
            for s_idx in self.sens_attr_range:
                self.y_s[y_idx].append(np.bitwise_and(self.true_y == y_idx, self.s[s_idx]))
                self.yneq_s[y_idx].append(np.bitwise_and(self.true_y != y_idx, self.s[s_idx]))
        self.y_s = np.array(self.y_s)
        self.yneq_s = np.array(self.yneq_s)

    
    def statistical_parity(self):
        """
        P(y^=0|s=0) = P(y^=0|s=1) = ... = P(y^=0|s=N)
        [...]
        P(y^=M|s=0) = P(y^=M|s=1) = ... = P(y^=M|s=N)
        """
        stat_parity = []
        for y_hat_idx in self.class_range:
            stat_parity.append([])
            for s_idx in self.sens_attr_range:
                stat_parity[y_hat_idx].append(
                    float(sum(np.bitwise_and(self.y_hat[y_hat_idx], self.s[s_idx])) /
                    sum(self.s[s_idx]))
                )
        return stat_parity


    def equal_opportunity(self):
        """
        P(y^=0|y=0,s=0) = P(y^=0|y=0,s=1) = ... = P(y^=0|y=0,s=N)
        [...]
        P(y^=M|y=M,s=0) = P(y^=M|y=M,s=1) = ... = P(y^=M|y=M,s=N)
        """
        equal_opp = []
        for y_hat_idx in self.class_range:
            equal_opp.append([])
            for s_idx in self.sens_attr_range:
                denominator = sum(self.y_s[y_hat_idx][s_idx])
                if denominator == 0:
                    equal_opp[y_hat_idx].append(None)  # or 0.0, or np.nan explicitly
                else:
                    equal_opp[y_hat_idx].append(
                        float(sum(np.bitwise_and(self.y_hat[y_hat_idx], self.y_s[y_hat_idx][s_idx])) /
                        denominator)
                    )
        return equal_opp


    def overall_accuracy_equality(self):
        ''' P(y^=0|y=0,s=0) + ... + P(y^=M|y=M,s=0) = ... = P(y^=0|y=0,s=N) + ... + P(y^=M|y=M,s=N) '''
        oae_s = []
        for s_idx in self.sens_attr_range:
            oae_temp = 0.0
            count = 0
            for y_hat_idx in self.class_range:
                denominator = sum(self.y_s[y_hat_idx][s_idx])
                if denominator > 0:
                    oae_temp += float(
                        sum(np.bitwise_and(self.y_hat[y_hat_idx], self.y_s[y_hat_idx][s_idx])) /
                        denominator
                    )
                    count += 1
            # If no classes had samples for this sensitive attribute group
            oae_s.append(oae_temp if count > 0 else None)
        return oae_s


    def treatment_equality(self):
        """
        P(y^=0|y/=0,s=0) / P(y^/=0|y=0,s=0) = ... = P(y^=0|y/=0,s=N) / P(y^/=0|y=0,s=N)
        [...]
        P(y^=M|y/=M,s=0) / P(y^/=M|y=M,s=0) = ... = P(y^=M|y/=M,s=N) / P(y^/M|y=M,s=N)
        """
        te_fp_fn = []
        te_fn_fp = []
        te = []
        for y_hat_idx in self.class_range:
            te_fp_fn.append([])
            te_fn_fp.append([])
            abs_te_fp_fn = 0.0
            abs_te_fn_fp = 0.0
            te.append([])
            for s_idx in self.sens_attr_range:
                try:
                    te_fp_fn[y_hat_idx].append(
                        float((sum(np.bitwise_and(self.y_hat[y_hat_idx], self.yneq_s[y_hat_idx][s_idx])) / sum(self.yneq_s[y_hat_idx][s_idx])) /
                        (sum(np.bitwise_and(self.yneq_hat[y_hat_idx], self.y_s[y_hat_idx][s_idx])) / sum(self.y_s[y_hat_idx][s_idx])))
                    )
                except ZeroDivisionError:
                    te_fp_fn[y_hat_idx].append(100.0)
                
                try:
                    te_fn_fp[y_hat_idx].append(
                        float((sum(np.bitwise_and(self.yneq_hat[y_hat_idx], self.y_s[y_hat_idx][s_idx])) / sum(self.y_s[y_hat_idx][s_idx])) /
                        (sum(np.bitwise_and(self.y_hat[y_hat_idx], self.yneq_s[y_hat_idx][s_idx])) / sum(self.yneq_s[y_hat_idx][s_idx])))
                    )
                except ZeroDivisionError:
                    te_fn_fp[y_hat_idx].append(100.0)

                abs_te_fp_fn += abs(te_fp_fn[y_hat_idx][s_idx])
                abs_te_fn_fp += abs(te_fn_fp[y_hat_idx][s_idx])
        
                if abs_te_fp_fn < abs_te_fn_fp:
                    te[y_hat_idx].append(te_fp_fn[y_hat_idx][s_idx])
                else:
                    te[y_hat_idx].append(te_fn_fp[y_hat_idx][s_idx])
        return te
    

    def compute(self):
        """
        Returns a flattened dictionary with keys in the format:
        {sens_attr}_{emotion}_{metric_name}
        
        For each emotion and metric, computes the mean absolute difference
        among all pairwise combinations of sensitive attribute values.
        Also computes the mean across all emotions for each metric.
        
        For example:
        - gender_emotion0_statistical_parity: mean(|P(y^=0|s=0) - P(y^=0|s=1)|, ...)
        - gender_mean_statistical_parity: mean over all emotions
        """
        sp = self.statistical_parity()
        eo = self.equal_opportunity()
        oae = self.overall_accuracy_equality()
        # te = self.treatment_equality()
        
        fairness_dict = {}
        
        # Statistical Parity: for each emotion, compute mean absolute difference
        sp_values = []
        for emotion_idx in self.class_range:
            emotion_label = self.unique_classes[emotion_idx]
            probs = [sp[emotion_idx][sens_idx] for sens_idx in self.sens_attr_range]
            abs_diffs = []
            for i in range(len(probs)):
                for j in range(i + 1, len(probs)):
                    abs_diffs.append(abs(probs[i] - probs[j]))
            
            mean_abs_diff = np.mean(abs_diffs) if abs_diffs else 0.0
            key = f"{emotion_label}_statistical_parity"
            fairness_dict[key] = mean_abs_diff
            sp_values.append(mean_abs_diff)
        fairness_dict["statistical_parity"] = np.mean(sp_values) if sp_values else 0.0
        
        # Equal Opportunity: for each emotion
        eo_values = []
        for emotion_idx in self.class_range:
            emotion_label = self.unique_classes[emotion_idx]
            probs = [eo[emotion_idx][sens_idx] for sens_idx in self.sens_attr_range]
            probs = [p for p in probs if p is not None]
            
            abs_diffs = []
            for i in range(len(probs)):
                for j in range(i + 1, len(probs)):
                    abs_diffs.append(abs(probs[i] - probs[j]))
            
            mean_abs_diff = np.mean(abs_diffs) if abs_diffs else None
            key = f"{emotion_label}_equal_opportunity"
            fairness_dict[key] = mean_abs_diff
            if mean_abs_diff is not None:
                eo_values.append(mean_abs_diff)
        fairness_dict["equal_opportunity"] = np.mean(eo_values) if eo_values else None
        
        # Overall Accuracy Equality (already a single value across emotions)
        probs = [oae[sens_idx] for sens_idx in self.sens_attr_range]
        probs = [p for p in probs if p is not None]
        
        abs_diffs = []
        for i in range(len(probs)):
            for j in range(i + 1, len(probs)):
                abs_diffs.append(abs(probs[i] - probs[j]))
        
        mean_abs_diff = np.mean(abs_diffs) if abs_diffs else None
        key = "overall_accuracy_equality"
        fairness_dict[key] = mean_abs_diff
        
        return fairness_dict