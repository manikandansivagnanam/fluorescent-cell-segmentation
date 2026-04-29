\## 🧬 Project Overview



This repository contains a modular, end-to-end deep learning pipeline for the \*\*image segmentation and object detection of fluorescent neuronal cells\*\*. Leveraging pretrained neural network architectures, this project aims to accurately identify and segment cellular structures from high-resolution microscopy datasets, facilitating automated quantitative analysis in neurobiology.



\---



\## 📁 Repository Architecture



The codebase is structured for scalability, reproducibility, and clear separation of concerns.



\### Data Processing \& Loading



| Script | Description |

|---|---|

| `prepare\_dataset.py` | Handles raw data ingestion and standardises image formats |

| `create\_splits.py` | Partitions the dataset into robust training, validation, and testing sets |

| `dataset.py` | Defines the custom PyTorch `Dataset` class and handles image–mask pairings |

| `dataloaders.py` | Configures batching, shuffling, and data loading mechanisms |



\### Model Architecture \& Training



| Script | Description |

|---|---|

| `model.py` | Defines the deep learning architecture, integrating pretrained weights for feature extraction |

| `train.py` | Contains the main training loop, including loss calculation, backpropagation, and optimisation |

| `metrics.py` | Defines evaluation metrics such as Intersection over Union (IoU) and Dice Coefficient |

| `run\_all.py` | Orchestration script to execute the entire pipeline end-to-end |

| `classical\_baseline.py` | Implements a traditional computer vision baseline for comparison against the deep learning approach |



\### Inference \& Evaluation



| Script | Description |

|---|---|

| `infer.py` | Executes the trained model on unseen test data to generate segmentation masks |

| `summarise\_results.py` | Aggregates performance metrics across the test set |

| `merge\_history.py` | Consolidates training logs and metric histories for plotting |



\### Testing \& Visualisation Utilities



| Script | Description |

|---|---|

| `visualize\_dataset.py` | Generates exploratory data analysis (EDA) plots to inspect raw images and ground truth masks |

| `check\_mask.py` | Verifies the integrity and binary/categorical mapping of segmentation masks |

| `test\_dataset.py` / `testdataloaders\_onesplit.py` | Unit tests to ensure data is loaded and augmented correctly before training |

| `test\_gpu.py` | Verifies hardware acceleration and GPU allocation |



\---



\## 🚀 Getting Started



\### 1. Environment Setup



It is recommended to use a virtual environment. Install the required dependencies using the provided requirements file:



```bash

pip install -r requirements.txt

```



\### 2. Verify Hardware



Ensure your environment is configured for GPU acceleration:



```bash

python test\_gpu.py

```



\### 3. Execution Pipeline



To run the full pipeline from data preparation to inference, use the main orchestration script:



```bash

python run\_all.py

```



Alternatively, execute the pipeline step-by-step:



```bash

\# Step 1: Prepare data

python prepare\_dataset.py



\# Step 2: Create data splits

python create\_splits.py



\# Step 3: Train the model

python train.py



\# Step 4: Run inference

python infer.py

```



\---



Results are aggregated and reported via `summarise\_results.py`.



\---



\## 🗂️ Project Structure



```

.

├── prepare\_dataset.py

├── create\_splits.py

├── dataset.py

├── dataloaders.py

├── model.py

├── train.py

├── metrics.py

├── run\_all.py

├── classical\_baseline.py

├── infer.py

├── summarise\_results.py

├── merge\_history.py

├── visualize\_dataset.py

├── check\_mask.py

├── test\_dataset.py

├── testdataloaders\_onesplit.py

├── test\_gpu.py

└── requirements.txt

```







