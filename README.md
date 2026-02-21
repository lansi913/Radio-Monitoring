# Radio Monitoring of Cardiac Activity with Pre-trained and Fine-tuned Model for Bedridden Patients

The data and code of the paper titled Radio Monitoring of Cardiac Activity with Pre-trained and Fine-tuned Model for Bedridden Patients.

---

# ➡️ Folder Structure Overview

The folder consists of three parts:

## 📊 data

* The radio displacement samples from two hospital cohorts of patients with cardiac abnormalities comprise 172 patients in the 'Ruijin' folder and 259 patients in the 'Xinhua' folder.
* Each CSV file containing radio displacement samples includes 30-second sample segments at a sampling rate of 50 Hz.
* A file named 'disp_mask.ipynb' demonstrates the process of masking displacement waveforms.

## 🕹️ code
* In 'Data' folder, the fintuning dataset from two hostiptals are listed and the pretraining dataset from Ruijin Hospital can be downloaded from the link in the 'pretraining_data.txt' file.
* In 'Module' folder, some basic python files as network modules are listed.
* In 'base_model.py' file, the network architecture is defined.
* **Run 'Pre_traing.py' file, you can get a pre-trained model** after putting the right pretraining dataset into the corresponding file path.
* **Run 'fine_tune_LoRA.py' file, you can get the finetuned results** regarding to the arrhythmia of premature ventricular contraction. You can also try other arrhythmias in the fintuning dataset. A pre-trained model weight file for testing purposes can be downloaded from the link in the file 'Model_weights.txt'. 
* It is recommended to change your execution path to the current system directory of code. This allows you to avoid modifying the relative import paths of these files when running Python scripts. 

## 🔧 Installation

Install Python 3.9 or higher version and necessary dependencies.

`
pip install -r requirements.txt 
`

---

# 🔥 Pre-traing and Fine-tunning neraul network

## 1️⃣ Wash the unlabelled data as the pre-traing dataset. 

## 2️⃣ Train the pre-traing model.

## 3️⃣ Fine-tune the base model to downstream tasks.

---

# ❤️ Acknowledgments

For more insteresting information, welcome refer to our lab website: https://changzhan.sjtu.edu.cn/


