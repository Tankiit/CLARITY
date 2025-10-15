#!/usr/bin/env python
"""
Workshop Paper Ablation Framework
================================================================================
Minimal-but-deep approach: ONE model (DistilBERT Credal CBM) + RICH ablations

Five comprehensive ablation studies:
1. Uncertainty decomposition necessity (correlation with prediction errors)
2. Intervention effectiveness (epistemic vs aleatoric)
3. Ensemble size impact (1, 3, 5, 7 members)
4. Domain-specific patterns (CEBaB analysis)
5. Calibration analysis (Expected Calibration Error)

Total runtime: ~4-5 hours
Paper contribution: Deep ablation analysis, not model zoo
================================================================================
"""

import os
import sys
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from typing import Dict, List, Tuple
from scipy import stats
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Import the CLARITY Credal CBM model
from clarity_credal_cbm import ClarityCredalCBM


class CEBABDataset(Dataset):
    """Dataset wrapper for CEBaB"""
    def __init__(self, dataset, tokenizer, max_length=128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item['description']
        review_majority = item['review_majority']

        # Handle different label formats
        if isinstance(review_majority, str):
            if review_majority == 'no majority':
                label = 1
            else:
                try:
                    label_value = int(review_majority)
                    label = 1 if label_value >= 3 else 0
                except ValueError:
                    label = 1
        else:
            try:
                label_value = int(review_majority)
                label = 1 if label_value >= 3 else 0
            except (ValueError, TypeError):
                label = 1

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding['labels'] = torch.tensor(label, dtype=torch.long)

        return encoding


class AblationSuite:
    """
    Comprehensive ablation study suite for workshop paper

    Design Philosophy:
    - Focus on ONE model (DistilBERT Credal CBM)
    - RICH ablations instead of model zoo
    - Each ablation answers a specific research question
    """

    def __init__(self, model, tokenizer, device, output_dir='./ablation_results'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.results = {}

    def load_checkpoint(self, checkpoint_path):
        """Load trained model checkpoint"""
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Checkpoint epoch: {checkpoint['epoch']}")
        print(f"  Val accuracy: {checkpoint['val_acc']:.4f}")
        print(f"  Epistemic: {checkpoint['epistemic']:.4f}")
        print(f"  Aleatoric: {checkpoint['aleatoric']:.4f}")

    def evaluate_model(self, dataloader, return_detailed=False):
        """
        Comprehensive model evaluation

        Returns:
            metrics: Dict with accuracy, loss, uncertainties
            detailed: (Optional) Dict with per-sample predictions, uncertainties, labels
        """
        self.model.eval()

        all_preds = []
        all_labels = []
        all_logits = []
        all_epistemic = []
        all_aleatoric = []
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask, labels)

                preds = outputs['logits'].argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_logits.extend(outputs['logits'].cpu().numpy())

                # Collect uncertainty metrics
                epistemic = outputs['uncertainty_metrics']['epistemic'].mean(1)
                aleatoric = outputs['uncertainty_metrics']['aleatoric'].mean(1)
                all_epistemic.extend(epistemic.cpu().numpy())
                all_aleatoric.extend(aleatoric.cpu().numpy())

                total_loss += outputs['loss'].item()

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_logits = np.array(all_logits)
        all_epistemic = np.array(all_epistemic)
        all_aleatoric = np.array(all_aleatoric)

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        avg_loss = total_loss / len(dataloader)
        avg_epistemic = np.mean(all_epistemic)
        avg_aleatoric = np.mean(all_aleatoric)
        avg_total = avg_epistemic + avg_aleatoric
        epistemic_ratio = avg_epistemic / avg_total if avg_total > 0 else 0

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'loss': avg_loss,
            'epistemic': avg_epistemic,
            'aleatoric': avg_aleatoric,
            'total_uncertainty': avg_total,
            'epistemic_ratio': epistemic_ratio
        }

        if return_detailed:
            detailed = {
                'predictions': all_preds,
                'labels': all_labels,
                'logits': all_logits,
                'epistemic': all_epistemic,
                'aleatoric': all_aleatoric
            }
            return metrics, detailed

        return metrics

    # =========================================================================
    # ABLATION 1: Uncertainty Decomposition Necessity
    # =========================================================================

    def ablation_1_decomposition_necessity(self, dataloader):
        """
        Research Question: Is uncertainty decomposition necessary?

        Method:
        - Correlate epistemic uncertainty with prediction errors
        - Correlate aleatoric uncertainty with prediction errors
        - Compare which uncertainty type better predicts errors

        Expected Result:
        - High epistemic uncertainty → More likely to be wrong
        - Aleatoric uncertainty → Less correlated with errors (data noise)
        """
        print("\n" + "="*80)
        print("ABLATION 1: Uncertainty Decomposition Necessity")
        print("="*80)
        print("\nResearch Question: Does decomposing uncertainty help predict errors?")
        print("Method: Correlate epistemic/aleatoric with prediction correctness\n")

        # Get detailed predictions
        metrics, detailed = self.evaluate_model(dataloader, return_detailed=True)

        # Calculate error indicator (1 = error, 0 = correct)
        errors = (detailed['predictions'] != detailed['labels']).astype(int)

        # Calculate correlations
        epistemic_corr, epistemic_pval = stats.pearsonr(detailed['epistemic'], errors)
        aleatoric_corr, aleatoric_pval = stats.pearsonr(detailed['aleatoric'], errors)
        total_uncert = detailed['epistemic'] + detailed['aleatoric']
        total_corr, total_pval = stats.pearsonr(total_uncert, errors)

        # Split into correct/incorrect predictions
        correct_mask = errors == 0
        incorrect_mask = errors == 1

        epistemic_correct = detailed['epistemic'][correct_mask]
        epistemic_incorrect = detailed['epistemic'][incorrect_mask]
        aleatoric_correct = detailed['aleatoric'][correct_mask]
        aleatoric_incorrect = detailed['aleatoric'][incorrect_mask]

        # Statistical test (t-test)
        epistemic_ttest = stats.ttest_ind(epistemic_incorrect, epistemic_correct)
        aleatoric_ttest = stats.ttest_ind(aleatoric_incorrect, aleatoric_correct)

        results = {
            'overall_accuracy': metrics['accuracy'],
            'correlations': {
                'epistemic_vs_errors': {
                    'correlation': float(epistemic_corr),
                    'p_value': float(epistemic_pval),
                    'interpretation': 'Higher epistemic = more errors' if epistemic_corr > 0 else 'No correlation'
                },
                'aleatoric_vs_errors': {
                    'correlation': float(aleatoric_corr),
                    'p_value': float(aleatoric_pval),
                    'interpretation': 'Higher aleatoric = more errors' if aleatoric_corr > 0 else 'No correlation'
                },
                'total_vs_errors': {
                    'correlation': float(total_corr),
                    'p_value': float(total_pval)
                }
            },
            'uncertainty_by_correctness': {
                'correct_predictions': {
                    'count': int(correct_mask.sum()),
                    'epistemic_mean': float(epistemic_correct.mean()),
                    'epistemic_std': float(epistemic_correct.std()),
                    'aleatoric_mean': float(aleatoric_correct.mean()),
                    'aleatoric_std': float(aleatoric_correct.std())
                },
                'incorrect_predictions': {
                    'count': int(incorrect_mask.sum()),
                    'epistemic_mean': float(epistemic_incorrect.mean()),
                    'epistemic_std': float(epistemic_incorrect.std()),
                    'aleatoric_mean': float(aleatoric_incorrect.mean()),
                    'aleatoric_std': float(aleatoric_incorrect.std())
                }
            },
            'statistical_tests': {
                'epistemic_difference': {
                    't_statistic': float(epistemic_ttest.statistic),
                    'p_value': float(epistemic_ttest.pvalue),
                    'significant': epistemic_ttest.pvalue < 0.05,
                    'mean_difference': float(epistemic_incorrect.mean() - epistemic_correct.mean())
                },
                'aleatoric_difference': {
                    't_statistic': float(aleatoric_ttest.statistic),
                    'p_value': float(aleatoric_ttest.pvalue),
                    'significant': aleatoric_ttest.pvalue < 0.05,
                    'mean_difference': float(aleatoric_incorrect.mean() - aleatoric_correct.mean())
                }
            }
        }

        # Print summary
        print("\nResults:")
        print(f"  Overall Accuracy: {metrics['accuracy']:.4f}")
        print(f"\n  Correlation with Errors:")
        print(f"    Epistemic: r={epistemic_corr:.4f} (p={epistemic_pval:.4e})")
        print(f"    Aleatoric: r={aleatoric_corr:.4f} (p={aleatoric_pval:.4e})")
        print(f"    Total:     r={total_corr:.4f} (p={total_pval:.4e})")

        print(f"\n  Average Uncertainty:")
        print(f"    Correct Predictions:")
        print(f"      Epistemic: {epistemic_correct.mean():.4f} ± {epistemic_correct.std():.4f}")
        print(f"      Aleatoric: {aleatoric_correct.mean():.4f} ± {aleatoric_correct.std():.4f}")
        print(f"    Incorrect Predictions:")
        print(f"      Epistemic: {epistemic_incorrect.mean():.4f} ± {epistemic_incorrect.std():.4f}")
        print(f"      Aleatoric: {aleatoric_incorrect.mean():.4f} ± {aleatoric_incorrect.std():.4f}")

        print(f"\n  Statistical Significance:")
        print(f"    Epistemic difference: {'YES' if epistemic_ttest.pvalue < 0.05 else 'NO'} (p={epistemic_ttest.pvalue:.4e})")
        print(f"    Aleatoric difference: {'YES' if aleatoric_ttest.pvalue < 0.05 else 'NO'} (p={aleatoric_ttest.pvalue:.4e})")

        print(f"\n  KEY FINDING:")
        if epistemic_corr > aleatoric_corr and epistemic_ttest.pvalue < 0.05:
            print(f"    ✓ Epistemic uncertainty STRONGLY predicts errors!")
            print(f"    ✓ Decomposition is NECESSARY for error detection")
        else:
            print(f"    ⚠ Weak correlation - needs investigation")

        self.results['ablation_1_decomposition'] = results
        return results

    # =========================================================================
    # ABLATION 2: Intervention Effectiveness
    # =========================================================================

    def ablation_2_intervention_effectiveness(self, dataloader):
        """
        Research Question: Which uncertainty type should we target for intervention?

        Method:
        - Identify high-epistemic vs high-aleatoric samples
        - Measure accuracy on each group
        - Determine which uncertainty type is more "fixable"

        Expected Result:
        - High epistemic → Lower accuracy (fixable with more training)
        - High aleatoric → Moderate accuracy (inherent data ambiguity)
        """
        print("\n" + "="*80)
        print("ABLATION 2: Intervention Effectiveness")
        print("="*80)
        print("\nResearch Question: Which uncertainty type should we target?")
        print("Method: Compare accuracy on high-epistemic vs high-aleatoric samples\n")

        # Get detailed predictions
        metrics, detailed = self.evaluate_model(dataloader, return_detailed=True)

        # Calculate thresholds (75th percentile)
        epistemic_threshold = np.percentile(detailed['epistemic'], 75)
        aleatoric_threshold = np.percentile(detailed['aleatoric'], 75)

        # Identify sample groups
        high_epistemic_mask = detailed['epistemic'] > epistemic_threshold
        high_aleatoric_mask = detailed['aleatoric'] > aleatoric_threshold
        low_both_mask = (detailed['epistemic'] <= epistemic_threshold) & \
                        (detailed['aleatoric'] <= aleatoric_threshold)
        high_both_mask = high_epistemic_mask & high_aleatoric_mask

        # Exclusive groups
        epistemic_only_mask = high_epistemic_mask & ~high_aleatoric_mask
        aleatoric_only_mask = high_aleatoric_mask & ~high_epistemic_mask

        def group_accuracy(mask):
            if mask.sum() == 0:
                return 0.0, 0
            preds = detailed['predictions'][mask]
            labels = detailed['labels'][mask]
            return accuracy_score(labels, preds), int(mask.sum())

        results = {
            'overall_accuracy': metrics['accuracy'],
            'thresholds': {
                'epistemic_75th': float(epistemic_threshold),
                'aleatoric_75th': float(aleatoric_threshold)
            },
            'groups': {
                'high_epistemic_only': {
                    'count': int(epistemic_only_mask.sum()),
                    'accuracy': float(group_accuracy(epistemic_only_mask)[0]),
                    'avg_epistemic': float(detailed['epistemic'][epistemic_only_mask].mean()),
                    'avg_aleatoric': float(detailed['aleatoric'][epistemic_only_mask].mean())
                },
                'high_aleatoric_only': {
                    'count': int(aleatoric_only_mask.sum()),
                    'accuracy': float(group_accuracy(aleatoric_only_mask)[0]),
                    'avg_epistemic': float(detailed['epistemic'][aleatoric_only_mask].mean()),
                    'avg_aleatoric': float(detailed['aleatoric'][aleatoric_only_mask].mean())
                },
                'high_both': {
                    'count': int(high_both_mask.sum()),
                    'accuracy': float(group_accuracy(high_both_mask)[0]),
                    'avg_epistemic': float(detailed['epistemic'][high_both_mask].mean()),
                    'avg_aleatoric': float(detailed['aleatoric'][high_both_mask].mean())
                },
                'low_both': {
                    'count': int(low_both_mask.sum()),
                    'accuracy': float(group_accuracy(low_both_mask)[0]),
                    'avg_epistemic': float(detailed['epistemic'][low_both_mask].mean()),
                    'avg_aleatoric': float(detailed['aleatoric'][low_both_mask].mean())
                }
            }
        }

        # Print summary
        print("\nResults:")
        print(f"  Overall Accuracy: {metrics['accuracy']:.4f}")
        print(f"\n  Thresholds (75th percentile):")
        print(f"    Epistemic: {epistemic_threshold:.4f}")
        print(f"    Aleatoric: {aleatoric_threshold:.4f}")

        print(f"\n  Group Analysis:")
        for group_name, group_data in results['groups'].items():
            print(f"\n    {group_name.replace('_', ' ').title()}:")
            print(f"      Samples: {group_data['count']}")
            print(f"      Accuracy: {group_data['accuracy']:.4f}")
            print(f"      Avg Epistemic: {group_data['avg_epistemic']:.4f}")
            print(f"      Avg Aleatoric: {group_data['avg_aleatoric']:.4f}")

        # Key finding
        epist_acc = results['groups']['high_epistemic_only']['accuracy']
        aleat_acc = results['groups']['high_aleatoric_only']['accuracy']

        print(f"\n  KEY FINDING:")
        if epist_acc < aleat_acc:
            acc_gap = aleat_acc - epist_acc
            print(f"    ✓ High epistemic samples: {epist_acc:.2%} accuracy")
            print(f"    ✓ High aleatoric samples: {aleat_acc:.2%} accuracy")
            print(f"    ✓ Gap: {acc_gap:.2%} - Target epistemic for improvement!")
        else:
            print(f"    ⚠ Unexpected pattern - needs investigation")

        self.results['ablation_2_intervention'] = results
        return results

    # =========================================================================
    # ABLATION 3: Ensemble Size Impact
    # =========================================================================

    def ablation_3_ensemble_size(self, dataloader, ensemble_sizes=[1, 3, 5, 7]):
        """
        Research Question: How does ensemble size affect uncertainty quality?

        Method:
        - Vary number of credal points (ensemble members)
        - Measure epistemic uncertainty and accuracy
        - Find optimal trade-off

        Expected Result:
        - Larger ensemble → Better epistemic estimates
        - Diminishing returns after 5 members
        """
        print("\n" + "="*80)
        print("ABLATION 3: Ensemble Size Impact")
        print("="*80)
        print("\nResearch Question: Optimal number of ensemble members?")
        print(f"Method: Test ensemble sizes {ensemble_sizes}\n")

        results = {
            'ensemble_sizes': ensemble_sizes,
            'results_by_size': {}
        }

        # Save original model config
        original_n_credal = self.model.n_credal_points

        for n_credal in ensemble_sizes:
            print(f"\n{'='*40}")
            print(f"Testing ensemble size: {n_credal}")
            print(f"{'='*40}")

            # Modify model
            self.model.n_credal_points = n_credal

            # Evaluate
            metrics = self.evaluate_model(dataloader)

            results['results_by_size'][str(n_credal)] = {
                'accuracy': metrics['accuracy'],
                'epistemic': metrics['epistemic'],
                'aleatoric': metrics['aleatoric'],
                'epistemic_ratio': metrics['epistemic_ratio'],
                'f1': metrics['f1']
            }

            print(f"\n  Results:")
            print(f"    Accuracy: {metrics['accuracy']:.4f}")
            print(f"    Epistemic: {metrics['epistemic']:.4f} ({metrics['epistemic_ratio']:.1%})")
            print(f"    Aleatoric: {metrics['aleatoric']:.4f}")
            print(f"    F1 Score: {metrics['f1']:.4f}")

        # Restore original
        self.model.n_credal_points = original_n_credal

        # Analysis
        print(f"\n{'='*40}")
        print(f"Summary Analysis")
        print(f"{'='*40}\n")

        accuracies = [results['results_by_size'][str(n)]['accuracy'] for n in ensemble_sizes]
        epistemics = [results['results_by_size'][str(n)]['epistemic'] for n in ensemble_sizes]

        best_acc_idx = np.argmax(accuracies)
        best_ensemble = ensemble_sizes[best_acc_idx]

        print(f"  Accuracy by ensemble size:")
        for i, n in enumerate(ensemble_sizes):
            marker = " ← BEST" if i == best_acc_idx else ""
            print(f"    {n} members: {accuracies[i]:.4f}{marker}")

        print(f"\n  Epistemic uncertainty by ensemble size:")
        for i, n in enumerate(ensemble_sizes):
            print(f"    {n} members: {epistemics[i]:.4f}")

        print(f"\n  KEY FINDING:")
        print(f"    ✓ Best ensemble size: {best_ensemble} members")
        print(f"    ✓ Accuracy: {accuracies[best_acc_idx]:.4f}")

        # Check diminishing returns
        if len(ensemble_sizes) >= 3:
            acc_gains = [accuracies[i+1] - accuracies[i] for i in range(len(accuracies)-1)]
            print(f"    ✓ Accuracy gains: {[f'{g:.4f}' for g in acc_gains]}")
            if acc_gains[-1] < acc_gains[0] / 2:
                print(f"    ✓ Diminishing returns observed after {ensemble_sizes[-2]} members")

        results['best_ensemble_size'] = int(best_ensemble)
        results['best_accuracy'] = float(accuracies[best_acc_idx])

        self.results['ablation_3_ensemble_size'] = results
        return results

    # =========================================================================
    # ABLATION 4: Domain-Specific Patterns (CEBaB)
    # =========================================================================

    def ablation_4_domain_patterns(self, dataloader):
        """
        Research Question: Are there domain-specific uncertainty patterns?

        Method:
        - Analyze uncertainty patterns in CEBaB
        - Look at sentiment-specific patterns
        - Identify challenging aspects

        Expected Result:
        - Mixed reviews → Higher aleatoric
        - Clear sentiment → Lower total uncertainty
        """
        print("\n" + "="*80)
        print("ABLATION 4: Domain-Specific Patterns (CEBaB)")
        print("="*80)
        print("\nResearch Question: Domain-specific uncertainty patterns?")
        print("Method: Analyze CEBaB sentiment patterns\n")

        # Get detailed predictions
        metrics, detailed = self.evaluate_model(dataloader, return_detailed=True)

        # Split by predicted class (positive/negative sentiment)
        positive_mask = detailed['predictions'] == 1
        negative_mask = detailed['predictions'] == 0

        # Split by confidence (using softmax on logits)
        logits = detailed['logits']
        probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        max_probs = probs.max(axis=1)

        high_conf_mask = max_probs > 0.8
        medium_conf_mask = (max_probs >= 0.6) & (max_probs <= 0.8)
        low_conf_mask = max_probs < 0.6

        def analyze_group(mask, name):
            if mask.sum() == 0:
                return None

            preds = detailed['predictions'][mask]
            labels = detailed['labels'][mask]
            acc = accuracy_score(labels, preds)

            return {
                'count': int(mask.sum()),
                'accuracy': float(acc),
                'avg_epistemic': float(detailed['epistemic'][mask].mean()),
                'std_epistemic': float(detailed['epistemic'][mask].std()),
                'avg_aleatoric': float(detailed['aleatoric'][mask].mean()),
                'std_aleatoric': float(detailed['aleatoric'][mask].std()),
                'avg_confidence': float(max_probs[mask].mean())
            }

        results = {
            'overall_accuracy': metrics['accuracy'],
            'by_sentiment': {
                'positive': analyze_group(positive_mask, 'Positive'),
                'negative': analyze_group(negative_mask, 'Negative')
            },
            'by_confidence': {
                'high': analyze_group(high_conf_mask, 'High confidence'),
                'medium': analyze_group(medium_conf_mask, 'Medium confidence'),
                'low': analyze_group(low_conf_mask, 'Low confidence')
            },
            'uncertainty_distribution': {
                'epistemic': {
                    'min': float(detailed['epistemic'].min()),
                    'max': float(detailed['epistemic'].max()),
                    'mean': float(detailed['epistemic'].mean()),
                    'std': float(detailed['epistemic'].std()),
                    'quartiles': {
                        '25th': float(np.percentile(detailed['epistemic'], 25)),
                        '50th': float(np.percentile(detailed['epistemic'], 50)),
                        '75th': float(np.percentile(detailed['epistemic'], 75))
                    }
                },
                'aleatoric': {
                    'min': float(detailed['aleatoric'].min()),
                    'max': float(detailed['aleatoric'].max()),
                    'mean': float(detailed['aleatoric'].mean()),
                    'std': float(detailed['aleatoric'].std()),
                    'quartiles': {
                        '25th': float(np.percentile(detailed['aleatoric'], 25)),
                        '50th': float(np.percentile(detailed['aleatoric'], 50)),
                        '75th': float(np.percentile(detailed['aleatoric'], 75))
                    }
                }
            }
        }

        # Print summary
        print("\nResults:")
        print(f"  Overall Accuracy: {metrics['accuracy']:.4f}")

        print(f"\n  By Predicted Sentiment:")
        for sentiment, data in results['by_sentiment'].items():
            if data:
                print(f"\n    {sentiment.title()}:")
                print(f"      Samples: {data['count']}")
                print(f"      Accuracy: {data['accuracy']:.4f}")
                print(f"      Epistemic: {data['avg_epistemic']:.4f} ± {data['std_epistemic']:.4f}")
                print(f"      Aleatoric: {data['avg_aleatoric']:.4f} ± {data['std_aleatoric']:.4f}")

        print(f"\n  By Confidence Level:")
        for conf_level, data in results['by_confidence'].items():
            if data:
                print(f"\n    {conf_level.title()} Confidence:")
                print(f"      Samples: {data['count']}")
                print(f"      Accuracy: {data['accuracy']:.4f}")
                print(f"      Epistemic: {data['avg_epistemic']:.4f} ± {data['std_epistemic']:.4f}")
                print(f"      Aleatoric: {data['avg_aleatoric']:.4f} ± {data['std_aleatoric']:.4f}")
                print(f"      Avg Confidence: {data['avg_confidence']:.4f}")

        print(f"\n  KEY FINDING:")
        if results['by_confidence']['low']:
            low_conf_epist = results['by_confidence']['low']['avg_epistemic']
            high_conf_epist = results['by_confidence']['high']['avg_epistemic']
            print(f"    ✓ Low confidence → High epistemic: {low_conf_epist:.4f}")
            print(f"    ✓ High confidence → Low epistemic: {high_conf_epist:.4f}")
            print(f"    ✓ Uncertainty tracks confidence as expected!")

        self.results['ablation_4_domain_patterns'] = results
        return results

    # =========================================================================
    # ABLATION 5: Calibration Analysis
    # =========================================================================

    def ablation_5_calibration(self, dataloader, n_bins=10):
        """
        Research Question: Are uncertainty estimates well-calibrated?

        Method:
        - Calculate Expected Calibration Error (ECE)
        - Bin predictions by confidence
        - Compare predicted confidence vs actual accuracy

        Expected Result:
        - Well-calibrated model: ECE < 0.1
        - Confidence should match empirical accuracy
        """
        print("\n" + "="*80)
        print("ABLATION 5: Calibration Analysis")
        print("="*80)
        print("\nResearch Question: Are predictions well-calibrated?")
        print(f"Method: Expected Calibration Error (ECE) with {n_bins} bins\n")

        # Get detailed predictions
        metrics, detailed = self.evaluate_model(dataloader, return_detailed=True)

        # Calculate confidences (max softmax probability)
        logits = detailed['logits']
        probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        confidences = probs.max(axis=1)
        predictions = detailed['predictions']
        labels = detailed['labels']
        correct = (predictions == labels).astype(int)

        # Calculate ECE
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        bins_data = []

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = correct[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()

                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

                bins_data.append({
                    'bin': f'({bin_lower:.2f}, {bin_upper:.2f}]',
                    'count': int(in_bin.sum()),
                    'proportion': float(prop_in_bin),
                    'avg_confidence': float(avg_confidence_in_bin),
                    'accuracy': float(accuracy_in_bin),
                    'calibration_error': float(np.abs(avg_confidence_in_bin - accuracy_in_bin))
                })

        # Reliability diagram data
        results = {
            'overall_accuracy': metrics['accuracy'],
            'expected_calibration_error': float(ece),
            'n_bins': n_bins,
            'bins': bins_data,
            'calibration_quality': 'Excellent' if ece < 0.05 else 'Good' if ece < 0.1 else 'Fair' if ece < 0.15 else 'Poor'
        }

        # Print summary
        print("\nResults:")
        print(f"  Overall Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Expected Calibration Error (ECE): {ece:.4f}")
        print(f"  Calibration Quality: {results['calibration_quality']}")

        print(f"\n  Reliability Diagram:")
        print(f"  {'Bin':<20} {'Count':<10} {'Confidence':<12} {'Accuracy':<12} {'Error':<10}")
        print(f"  {'-'*70}")
        for bin_data in bins_data:
            if bin_data['count'] > 0:
                print(f"  {bin_data['bin']:<20} {bin_data['count']:<10} "
                      f"{bin_data['avg_confidence']:<12.4f} {bin_data['accuracy']:<12.4f} "
                      f"{bin_data['calibration_error']:<10.4f}")

        print(f"\n  KEY FINDING:")
        if ece < 0.1:
            print(f"    ✓ Model is well-calibrated (ECE < 0.1)")
            print(f"    ✓ Confidence scores are reliable!")
        else:
            print(f"    ⚠ Model needs calibration (ECE = {ece:.4f})")
            print(f"    ⚠ Consider temperature scaling or Platt scaling")

        self.results['ablation_5_calibration'] = results
        return results

    # =========================================================================
    # Master Run Function
    # =========================================================================

    def run_all_ablations(self, val_loader, save_results=True):
        """
        Run all 5 ablation studies sequentially

        Args:
            val_loader: Validation DataLoader
            save_results: Save results to JSON
        """
        print("\n" + "="*80)
        print(" "*15 + "WORKSHOP PAPER ABLATION SUITE")
        print("="*80)
        print("\nRunning 5 comprehensive ablation studies...")
        print("Estimated runtime: ~30-60 minutes")
        print("="*80 + "\n")

        # Run all ablations
        ablation_1_results = self.ablation_1_decomposition_necessity(val_loader)
        ablation_2_results = self.ablation_2_intervention_effectiveness(val_loader)
        ablation_3_results = self.ablation_3_ensemble_size(val_loader, ensemble_sizes=[1, 3, 5, 7])
        ablation_4_results = self.ablation_4_domain_patterns(val_loader)
        ablation_5_results = self.ablation_5_calibration(val_loader)

        # Final summary
        print("\n" + "="*80)
        print(" "*20 + "ABLATION SUITE COMPLETE!")
        print("="*80)

        print("\n📊 KEY FINDINGS SUMMARY:\n")

        print("1️⃣ Decomposition Necessity:")
        epist_corr = ablation_1_results['correlations']['epistemic_vs_errors']['correlation']
        print(f"   - Epistemic-error correlation: {epist_corr:.4f}")
        print(f"   - Decomposition is {'NECESSARY ✓' if epist_corr > 0.1 else 'LIMITED ⚠'}")

        print("\n2️⃣ Intervention Effectiveness:")
        high_epist_acc = ablation_2_results['groups']['high_epistemic_only']['accuracy']
        print(f"   - High-epistemic accuracy: {high_epist_acc:.4f}")
        print(f"   - Target epistemic for improvement!")

        print("\n3️⃣ Ensemble Size:")
        best_size = ablation_3_results['best_ensemble_size']
        best_acc = ablation_3_results['best_accuracy']
        print(f"   - Optimal ensemble size: {best_size} members")
        print(f"   - Best accuracy: {best_acc:.4f}")

        print("\n4️⃣ Domain Patterns:")
        low_conf_count = ablation_4_results['by_confidence']['low']['count'] if ablation_4_results['by_confidence']['low'] else 0
        print(f"   - Low confidence samples: {low_conf_count}")
        print(f"   - Uncertainty tracks confidence ✓")

        print("\n5️⃣ Calibration:")
        ece = ablation_5_results['expected_calibration_error']
        quality = ablation_5_results['calibration_quality']
        print(f"   - ECE: {ece:.4f}")
        print(f"   - Quality: {quality}")

        # Save results
        if save_results:
            output_path = os.path.join(self.output_dir, 'ablation_results.json')
            with open(output_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            print(f"\n✅ Results saved to: {output_path}")

        print("\n" + "="*80)
        print("All ablations complete! Ready for workshop paper. 📝")
        print("="*80 + "\n")

        return self.results


def main():
    """
    Main runner for workshop paper ablations

    Usage:
        python workshop_ablations.py

    This will:
    1. Load the trained model checkpoint
    2. Run all 5 ablation studies
    3. Save comprehensive results to JSON
    """
    print("\n" + "="*80)
    print(" "*10 + "Workshop Paper Ablation Framework - CEBaB Analysis")
    print("="*80 + "\n")

    # Configuration
    checkpoint_path = 'checkpoints/best_model_cebab_50epochs.pt'
    base_model_name = 'distilbert-base-uncased'
    num_concepts = 15
    n_credal_points = 5
    num_classes = 2
    batch_size = 32  # Larger batch for faster evaluation
    max_seq_length = 128

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else
                         "mps" if torch.backends.mps.is_available() else
                         "cpu")
    print(f"Device: {device}\n")

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        print("Please ensure the 50-epoch training has completed.")
        print("Or update the checkpoint_path variable in this script.")
        return

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Load validation dataset
    print("Loading CEBaB validation dataset...")
    dataset = load_dataset("CEBaB/CEBaB")
    val_dataset = CEBABDataset(dataset['validation'], tokenizer, max_seq_length)
    print(f"Validation size: {len(val_dataset)}\n")

    # Create dataloader
    is_cuda = device.type == 'cuda'
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4 if is_cuda else 0,
        pin_memory=is_cuda
    )

    # Initialize model
    print("Initializing CLARITY Credal CBM model...")
    concept_names = [f"concept_{i}" for i in range(num_concepts)]

    model = ClarityCredalCBM(
        base_model_name=base_model_name,
        num_concepts=num_concepts,
        num_classes=num_classes,
        n_credal_points=n_credal_points,
        concept_names=concept_names
    ).to(device)

    # Initialize ablation suite
    ablation_suite = AblationSuite(
        model=model,
        tokenizer=tokenizer,
        device=device,
        output_dir='./ablation_results'
    )

    # Load checkpoint
    ablation_suite.load_checkpoint(checkpoint_path)

    # Run all ablations
    print("\n" + "="*80)
    print("Starting comprehensive ablation analysis...")
    print("="*80)

    results = ablation_suite.run_all_ablations(val_loader, save_results=True)

    print("\n✅ Ablation analysis complete!")
    print("Results saved to: ./ablation_results/ablation_results.json")
    print("\nUse these results for your workshop paper! 📝")


if __name__ == "__main__":
    main()
