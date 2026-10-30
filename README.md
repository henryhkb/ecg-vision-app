# ECG-Vision App

Detecting **atrial fibrillation (AF)** from images of paper ECGs using computer vision.

> **Status:** a learning and portfolio project. A first model reaches test F1 0.967 on clean, rendered ECG images, but a robustness test showed it fails on photo-like images (blur, tilt, lighting, perspective). Retraining with data augmentation is next.

---

## Why this project

Paper ECGs are still common in many hospitals, but the specialists who can read them are scarce. In Ghana, the Ghana Society of Cardiology reported only about 20 cardiologists nationwide in 2020, with more than half concentrated in Accra and Kumasi and several regions having none ([source](https://www.peacefmonline.com/pages/local/health/202002/402094.php)).

Many district hospitals have an ECG machine but no expert nearby to interpret the trace. This project explores whether a model can read a **photo of a printed ECG** and flag atrial fibrillation, one of the most common heart rhythm problems and a major risk factor for stroke.

## The task

**Binary image classification:** given an image of a 12-lead ECG, predict **Normal (0)** or **Atrial fibrillation (1)**.

Atrial fibrillation shows three key signs on an ECG:
1. Irregular R–R intervals with no pattern
2. No clear P waves
3. A wobbly, fibrillatory baseline between beats

## Data

This project uses **PTB-XL** (v1.0.3), a public dataset of 21,799 clinical 12-lead ECGs (10 seconds each) from 18,869 patients, annotated by cardiologists.

- Source: [PhysioNet – PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/)
- Licence: [Creative Commons Attribution 4.0](https://physionet.org/content/ptb-xl/view-license/1.0.3/)
- Version used: 100 Hz recordings (`records100`)

The dataset is **not included** in this repository. Download it from PhysioNet (the notebooks use the faster AWS mirror listed on the PhysioNet page).

### Label definitions

| Class | Rule (from `scp_codes`) | Count |
|---|---|---|
| Normal (0) | `NORM` **and** `SR` (sinus rhythm) **and not** `AFIB` | 8,204 |
| AF (1) | `AFIB` present | 1,514 |
| **Total** | | **9,718** |

**Note:** in PTB-XL, `NORM` describes wave *shape* (a diagnostic statement), while `AFIB` describes *rhythm*. Some ECGs carry both labels (37 records), so `NORM` alone does not guarantee a normal rhythm. That is why the normal class also requires `SR` and excludes `AFIB`.

### Train / validation / test split

Uses the dataset authors' recommended stratified folds (patients never appear in more than one split):

| Split | Folds | Normal | AF | Total |
|---|---|---|---|---|
| Train | 1–8 | 6,548 | 1,211 | 7,759 |
| Validation | 9 | 841 | 151 | 992 |
| Test | 10 | 815 | 152 | 967 |

Folds 9 and 10 contain only human-validated ECGs, giving the most reliable evaluation labels. The AF share is about 15% in every split, so the classes are **imbalanced** (roughly 5.4 normal : 1 AF). Evaluation will therefore focus on sensitivity, specificity and F1 rather than accuracy alone.

## Notebooks

| Notebook | Purpose | Status |
|---|---|---|
| `ECG_Vision_01_Explore_PTBXL.ipynb` | Download PTB-XL, explore metadata, build NORM vs. AFIB labels and splits, plot raw waveforms with `wfdb`, and render all 9,718 ECGs as standard 12-lead printout images | ✅ Done |
| `ECG_Vision_02_Train_Model.ipynb` | Fine-tune ResNet-18 on the ECG images, analyse validation errors, choose the decision threshold, and evaluate once on the test set | ✅ Done |
| `ECG_Vision_03_Robustness.ipynb` | Test the trained model on photo-like distortions, then retrain with data augmentation | 🔄 In progress |

Built and run in **Google Colab**, with data stored in Google Drive.

## Roadmap

- [x] Learn ECG fundamentals (waves, grid, heart rate, rhythm, AF)
- [x] Define the task: Normal vs. AF
- [x] Download, explore and label PTB-XL
- [x] Plot real ECG waveforms
- [x] Render 9,718 ECGs as standard 12-lead printout images (3×4 leads + lead II rhythm strip, 25 mm/s, 10 mm/mV)
- [ ] Add more realistic image artefacts (creases, shadows, handwriting), e.g. with [ECG-Image-Kit](https://github.com/alphanumericslab/ecg-image-kit)
- [x] Simple baselines (always-Normal and an R–R irregularity rule)
- [x] Train a CNN image classifier (ResNet-18, transfer learning)
- [x] Choose the decision threshold on the validation set
- [x] Evaluate on the held-out test set (sensitivity, specificity, F1)
- [x] Error analysis of validation mistakes
- [x] Robustness test on photo-like distortions (model failed: many false alarms)
- [ ] Retrain with photo-like data augmentation and repeat the robustness test
- [x] Blind cardiologist review of the validation disagreements (agreed with the model in 6 of 7)
- [ ] Test on real phone photos of printed ECGs
- [ ] Prototype a mobile app

## ECG images

Each ECG is drawn from its raw 100 Hz signal as a standard 12-lead printout on ECG paper:

- Rows 1–3: leads I, aVR, V1, V4 / II, aVL, V2, V5 / III, aVF, V3, V6, each showing 2.5 seconds
- Row 4: lead II rhythm strip across the full 10 seconds
- Standard scale: 25 mm/s horizontally, 10 mm/mV vertically, with 1 mm and 5 mm grid boxes

Images are saved as `<ecg_id>.png` (about 100 KB each, roughly 1 GB in total) and are not included in this repository.

## Model

- **Architecture:** ResNet-18 pretrained on ImageNet (transfer learning), with the final layer replaced by a single output (the AF score)
- **Input:** each ECG image resized to 240 × 496 pixels and normalised with ImageNet statistics
- **Loss:** `BCEWithLogitsLoss` with `pos_weight = 5.41` (normal ÷ AF in the training set), so mistakes on the rarer AF class count more
- **Optimiser:** Adam, learning rate 1e-4, batch size 32, 5 epochs on a Colab T4 GPU (about 130 s per epoch)
- **Model selection:** the epoch with the best validation F1 was kept (epoch 3)
- **Threshold:** chosen on the validation set only. Thresholds from 0.05 to 0.95 were compared; **0.30** was chosen because missing AF is the more harmful error for a screening tool (validation: 3 missed AF, 6 false alarms, F1 0.970). The best-F1 threshold, 0.60, missed 5 AF with 1 false alarm.

## Results

### Baselines (validation set)

| Method | Sensitivity | Specificity | F1 |
|---|---|---|---|
| Always predict Normal | 0.000 | 1.000 | 0.000 (accuracy 0.848) |
| R–R irregularity rule (coefficient of variation > 0.110, learned on train) | 0.901 | 0.937 | 0.800 |

### CNN on the held-out test set (fold 10, evaluated once, threshold 0.30)

| | Predicted Normal | Predicted AF |
|---|---|---|
| **True Normal (815)** | 811 | 4 |
| **True AF (152)** | 6 | 146 |

| Sensitivity | Specificity | F1 | Accuracy |
|---|---|---|---|
| **0.961** | **0.995** | **0.967** | 0.990 |

The test F1 (0.967) is close to the validation F1 at the same threshold (0.970), so the model generalises to unseen patients.

## Error analysis (validation set, threshold 0.5)

The model made 7 validation errors: 5 missed AF and 2 false alarms. Reviewing the images and the original PTB-XL reports showed:

- **The model relies heavily on rhythm irregularity.** A normal ECG with premature ventricular contractions (PVCs), which make the rhythm irregular, was called AF with probability 0.96. AF ECGs whose rhythm looked fairly regular were called normal.
- **Some AF labels are ambiguous.** 4 of the 5 missed AF ECGs have the German report *"Vorhofflimmern/-flattern"* (atrial fibrillation **or** flutter). Atrial flutter is often regular, so some of these may not be fibrillation.
- **One missed case was reported as intermittent AF,** which may not be present during the 10-second recording.
- **The Normal class includes ECGs with extra beats** (such as PVCs), because the label rule (`NORM` + `SR` + no `AFIB`) does not exclude them.

### Blind review by a cardiologist

The 7 ECGs were sent to a cardiologist, Dr. Stephen Okyere Mensah, as images labelled only A to G, without the PTB-XL label or the model's prediction, so the reading would not be influenced.

| ECG | Cardiologist | PTB-XL label | Model (threshold 0.5) | Agrees with cardiologist |
|---|---|---|---|---|
| A | Not AF | AF | Normal | Model |
| B | Not AF | Normal | AF (0.96) | Label |
| C | Not AF | AF | Normal | Model |
| D | Not AF | AF | Normal | Model |
| E | AF | Normal | AF (0.57) | Model |
| F | Not AF | AF | Normal | Model |
| G | Not AF | AF | Normal | Model |

**The cardiologist agreed with the model in 6 of the 7 cases** and with the PTB-XL label in 1. This suggests most of these validation "errors" are **label errors** in PTB-XL's AF class, consistent with the ambiguous "fibrillation/flutter" reports. The one clear model error (B) is a normal ECG with premature ventricular contractions, which supports the finding that irregular rhythm alone can mislead the model.

**Caveats:** one reader and 7 ECGs, selected because the model disagreed with the label. The review shows that label noise exists but not how much; ECGs where the model and label agreed were not reviewed. No labels in the training, validation or test sets were changed based on this review.

## Robustness to photo-like images

Real use means phone photos of printed ECGs, not clean digital renders. To check this, the trained model (unchanged, threshold 0.30) was run on the 992 validation ECGs with one distortion applied at a time, using `torchvision` transforms:

| Distortion | Imitates | Missed AF (of 151) | False alarms (of 841) | Sensitivity | Specificity | F1 |
|---|---|---|---|---|---|---|
| None (clean) | | 3 | 6 | 0.980 | 0.993 | 0.970 |
| Rotation (up to ±8°) | Phone held at a tilt | 2 | 304 | 0.987 | 0.639 | 0.493 |
| Brightness/contrast (±50%) | Dim room, flash, uneven light | 2 | 303 | 0.987 | 0.640 | 0.494 |
| Perspective (scale 0.3) | Photo taken at an angle | 0 | 764 | 1.000 | 0.092 | 0.283 |
| Gaussian blur (σ = 3) | Out of focus, shaky hand | 0 | 841 | 1.000 | 0.000 | 0.264 |

**Finding:** under any distortion, the model labels many normal ECGs as AF, and under blur it labels every ECG as AF. The high sensitivity is not a strength here; it comes from predicting AF for almost everything. Because every normal ECG in training was perfectly clean, the model appears to have learned "not perfectly clean" as a shortcut for AF. This is a **distribution shift** problem, and it means the current model would not work on phone photos.

**Next step:** retrain with random photo-like augmentations applied to the training images only (rotation, perspective, blur and lighting, each applied with 50% probability), then repeat this table. Validation and test images stay unaugmented for the main scores.

## Limitations

- The model was trained and tested on clean, computer-rendered ECGs. On photo-like distortions its specificity collapses (see Robustness above), so the reported test scores apply to clean digital images only.
- Labels come from PTB-XL as published. A blind cardiologist review of the 7 validation disagreements found the label likely wrong in 6, so the reported scores are affected by label noise (in either direction).
- Only two classes (Normal vs. AF); other arrhythmias were excluded from the data.
- Not validated clinically; see the disclaimer below.

## How to run

1. Open the notebooks in Google Colab.
2. Run the setup cell to mount Google Drive and set project paths.
3. On first use, run the download cells in notebook 01 to fetch PTB-XL and save a copy to Drive.
4. In later sessions, run only the setup cell; it restores the dataset from Drive and loads the saved labels.
5. In notebook 02, choose a GPU first (**Runtime → Change runtime type → T4 GPU**), then run the cells in order. Training takes about 11 minutes for 5 epochs. The trained weights are saved to `ECG_Vision_App/models/resnet18_best.pt` on Drive (not included in this repository).

## Related work

- [George B. Moody PhysioNet Challenge 2024](https://moody-challenge.physionet.org/2024/): digitisation and classification of ECG images
- [PMcardio](https://www.powerfulmedical.com/): a commercial, certified smartphone app for AI ECG interpretation

## Acknowledgements

Thanks to **Dr. Stephen Okyere Mensah** for the blind review of the misclassified ECGs.

## Disclaimer

This is an educational research project. It is **not a medical device** and must not be used for clinical diagnosis or treatment decisions.

## Citation

If you use PTB-XL, please cite:

> Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., & Schaeffter, T. (2022). PTB-XL, a large publicly available electrocardiography dataset (version 1.0.3). *PhysioNet*. https://doi.org/10.13026/kfzx-aw45

> Wagner, P., Strodthoff, N., Bousseljot, R.-D., Kreiseler, D., Lunze, F. I., Samek, W., & Schaeffter, T. (2020). PTB-XL, a large publicly available ECG dataset. *Scientific Data*, 7, 154. https://doi.org/10.1038/s41597-020-0495-6

## Author

**Henry Kofi Brandoh**, MPhil candidate, Intelligent Computing Systems, Ashesi University
