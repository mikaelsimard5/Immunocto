import pandas as pd


def get_metrics(csv_path, prediction_columns = None):

    df = pd.read_csv(csv_path)
    
    # Identify the predicted class by selecting the column with the highest value for each row
    df['predicted'] = df[prediction_columns].idxmax(axis=1)

    # Initialize dictionaries for precision and recall
    precision_per_class = {}
    recall_per_class = {}

    for cls in prediction_columns:  # Loop through each class
        # True Positives (TP): Correctly predicted as the class
        TP = df[(df['label'] == cls) & (df['predicted'] == cls)].shape[0]
        
        # False Positives (FP): Predicted as the class but actually not
        FP = df[(df['label'] != cls) & (df['predicted'] == cls)].shape[0]
        
        # False Negatives (FN): Actually the class but predicted as not
        FN = df[(df['label'] == cls) & (df['predicted'] != cls)].shape[0]
        
        # Precision: TP / (TP + FP)
        precision_per_class[cls] = TP / (TP + FP) if (TP + FP) > 0 else 0
    
        # Recall: TP / (TP + FN)
        recall_per_class[cls] = TP / (TP + FN) if (TP + FN) > 0 else 0
    

    metrics = pd.DataFrame({'Precision': precision_per_class,
                            'Recall': recall_per_class}).fillna(0)

    print(metrics)
    return 


# Read datasets
lizard_on_lizard = "./Analysis/inference_results/lizard_convnet_tested_on_lizard.csv"
lizard_on_segpath = "./Analysis/inference_results/lizard_convnet_tested_on_SegPath.csv"
lizard_on_immunocto = "./Analysis/inference_results/lizard_convnet_tested_on_Immunocto.csv"

lizard_prediction_columns = ['connective', 'eosinophil', 'epithelial', 'lymphocyte', 'neutrophil', 'plasma']

print('Train Lizard on Test lizard:')
get_metrics(lizard_on_lizard, prediction_columns = lizard_prediction_columns)

print('Train Lizard on Test SegPath:')
get_metrics(lizard_on_segpath, prediction_columns = lizard_prediction_columns)

print('Train Lizard on Test Immunocto:')
get_metrics(lizard_on_immunocto, prediction_columns = lizard_prediction_columns)