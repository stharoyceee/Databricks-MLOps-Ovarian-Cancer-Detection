# Databricks-MLOps-Ovarian-Cancer-Detection
An enterprise-grade, reproducible computer vision MLOps pipeline deployed on Databricks Lakehouse using Unity Catalog, MLflow, and Databricks Workflows (Asset Bundles).

Raw Histopathology Images (Unity Catalog Volume)
                    │
                    ▼
          [ 01_Data_Cleaning ]
    (Tissue Masking & Delta Table Cataloging)
                    │
                    ▼
           [ 02_Preprocessing ]
   (Stratified Splitting & MLflow Dataset Lineage)
                    │
                    ▼
     [ 03_Feature_Extraction_Selection ]
    (ResNet-50 2048-D -> Variance Drop -> Colinearity Filter -> SelectKBest 200-D)
                    │
                    ▼
          [ 04_Model_Training ]
    (Tournament: Logistic Regression vs Random Forest vs XGBoost via Nested MLflow Runs)
                    │
                    ▼
      [ 05_Champion_Model_Registration ]
   (End-to-End MLflow PyFunc Wrapper + Automated Quality Gate & Alias Assignment)
                    │
                    ▼
         [ 06_Champion_Model_Testing ]
 (Holdout Batch Inference & Delta Diagnostics: 86.75% Test Accuracy / 0.8674 Macro F1)
