import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from tqdm import tqdm
import os
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
EXTERNAL_TEST_DIR = r"F:\jihad\Breast GAN types\breast\breast\test1"
BATCH_SIZE = 32
NUM_CLASSES = 2
IMG_SIZE = 224
IMG_SIZE_INCEPTION = 299

# Find the most recent best model file
def find_best_model():
    model_files = glob.glob('best_model_*.pth')
    if not model_files:
        raise FileNotFoundError("No best model file found. Please ensure the model has been trained first.")
    # Get the most recent file
    latest_model = max(model_files, key=os.path.getctime)
    return latest_model

# Data transforms for standard models
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Data transforms for Inception v3
test_transform_inception = transforms.Compose([
    transforms.Resize((IMG_SIZE_INCEPTION, IMG_SIZE_INCEPTION)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Model loading function
def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model_name = checkpoint['model_name']
    class_names = checkpoint['class_names']
    
    print(f"Loading model: {model_name}")
    print(f"Classes: {class_names}")
    print(f"Previous F1-Score: {checkpoint['f1_score']:.4f}")
    
    # Initialize model architecture
    if model_name == 'efficientnet_v2_l':
        model = models.efficientnet_v2_l(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    
    elif model_name == 'resnet50':
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, NUM_CLASSES)
    
    elif model_name == 'densenet121':
        model = models.densenet121(weights=None)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, NUM_CLASSES)
    
    elif model_name == 'inception_v3':
        model = models.inception_v3(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, NUM_CLASSES)
        model.aux_logits = False
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, model_name, class_names

# Evaluation function
def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_paths = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating External Dataset'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

# Calculate metrics
def calculate_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.0
        print("Warning: AUC-ROC could not be calculated")
    
    cm = confusion_matrix(y_true, y_pred)
    
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'confusion_matrix': cm
    }

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Find and load the best model
    try:
        model_path = find_best_model()
        print(f"Found model: {model_path}\n")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    model, model_name, class_names = load_model(model_path, device)
    
    # Determine which transform to use
    if model_name == 'inception_v3':
        transform = test_transform_inception
        print(f"Using Inception v3 transform (299x299)\n")
    else:
        transform = test_transform
        print(f"Using standard transform (224x224)\n")
    
    # Load external test dataset
    print(f"Loading external test dataset from: {EXTERNAL_TEST_DIR}")
    
    if not os.path.exists(EXTERNAL_TEST_DIR):
        print(f"Error: Directory not found: {EXTERNAL_TEST_DIR}")
        return
    
    try:
        external_dataset = datasets.ImageFolder(EXTERNAL_TEST_DIR, transform=transform)
        external_loader = DataLoader(external_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        print(f"External test samples: {len(external_dataset)}")
        print(f"External test classes: {external_dataset.classes}")
        print(f"Class to index mapping: {external_dataset.class_to_idx}\n")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Evaluate on external dataset
    print(f"{'='*60}")
    print(f"Evaluating {model_name} on External Test Set")
    print(f"{'='*60}\n")
    
    y_pred, y_true, y_prob = evaluate_model(model, external_loader, device)
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    
    # Display results
    print(f"\n{'='*60}")
    print(f"External Test Results for {model_name}")
    print(f"{'='*60}")
    print(f"Accuracy:     {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision:    {metrics['precision']:.4f}")
    print(f"Recall:       {metrics['recall']:.4f}")
    print(f"F1-Score:     {metrics['f1_score']:.4f}")
    print(f"AUC-ROC:      {metrics['auc_roc']:.4f}")
    print(f"Sensitivity:  {metrics['sensitivity']:.4f}")
    print(f"Specificity:  {metrics['specificity']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {metrics['true_negatives']:4d}  FP: {metrics['false_positives']:4d}")
    print(f"  FN: {metrics['false_negatives']:4d}  TP: {metrics['true_positives']:4d}")
    print(f"{'='*60}\n")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Overall metrics for external test
    external_metrics_df = pd.DataFrame([{
        'model_name': model_name,
        'test_dataset': 'external',
        'test_samples': len(external_dataset),
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1_score': metrics['f1_score'],
        'auc_roc': metrics['auc_roc'],
        'sensitivity': metrics['sensitivity'],
        'specificity': metrics['specificity'],
        'TPR': metrics['sensitivity'],  # TPR is same as sensitivity/recall
        'TNR': metrics['specificity'],  # TNR is same as specificity
        'true_negatives': metrics['true_negatives'],
        'false_positives': metrics['false_positives'],
        'false_negatives': metrics['false_negatives'],
        'true_positives': metrics['true_positives']
    }])
    
    external_metrics_file = f'external_test_metrics_{model_name}_{timestamp}.csv'
    external_metrics_df.to_csv(external_metrics_file, index=False)
    print(f"Saved: {external_metrics_file}")
    
    # 2. Detailed predictions for external test
    detailed_predictions = []
    for i in range(len(y_true)):
        detailed_predictions.append({
            'sample_index': i,
            'true_label': int(y_true[i]),
            'true_class': class_names[y_true[i]],
            'predicted_label': int(y_pred[i]),
            'predicted_class': class_names[y_pred[i]],
            'probability_class_0': float(1 - y_prob[i]),
            'probability_class_1': float(y_prob[i]),
            'correct': int(y_true[i] == y_pred[i])
        })
    
    predictions_df = pd.DataFrame(detailed_predictions)
    predictions_file = f'external_test_predictions_{model_name}_{timestamp}.csv'
    predictions_df.to_csv(predictions_file, index=False)
    print(f"Saved: {predictions_file}")
    
    # 3. Confusion matrix
    cm_df = pd.DataFrame(
        metrics['confusion_matrix'],
        index=[f'True_{name}' for name in class_names],
        columns=[f'Pred_{name}' for name in class_names]
    )
    cm_file = f'external_test_confusion_matrix_{model_name}_{timestamp}.csv'
    cm_df.to_csv(cm_file)
    print(f"Saved: {cm_file}")
    
    print("\nExternal test evaluation completed successfully!")
    print("\nGenerated files:")
    print(f"  1. {external_metrics_file}")
    print(f"  2. {predictions_file}")
    print(f"  3. {cm_file}")
    
    # Summary statistics
    correct_predictions = (y_true == y_pred).sum()
    total_predictions = len(y_true)
    print(f"\nSummary:")
    print(f"  Total samples: {total_predictions}")
    print(f"  Correct predictions: {correct_predictions}")
    print(f"  Incorrect predictions: {total_predictions - correct_predictions}")
    print(f"  Accuracy: {correct_predictions/total_predictions*100:.2f}%")


if __name__ == '__main__':
    main()