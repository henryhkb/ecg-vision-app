# ECG-Vision App

Detecting **atrial fibrillation (AF)** from images of paper ECGs using computer vision.

> **Status:** a learning and portfolio project. The current model, retrained with photo-like data augmentation, reaches test F1 0.954 on clean ECG images and 0.92–0.96 on blurred, tilted, angled and re-lit versions (the first model scored 0.27–0.52 on those). A first real-camera test on 36 phone photos (of ECGs shown on a laptop screen) gave F1 0.94–0.97 for both models; testing on photos of printed ECGs under harder conditions is next.

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
| `ECG_Vision_03_Robustness.ipynb` | Test the trained model on photo-like distortions, retrain with data augmentation, choose a new threshold and evaluate once on the test set | ✅ Done |
| `ECG_Vision_04_Phone_Photos.ipynb` | Select 40 test ECGs, photograph them with an iPhone, match photos to ECGs and score both models | ✅ Done |

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
- [x] Retrain with photo-like data augmentation and repeat the robustness test (test F1 on distorted images: 0.27–0.52 → 0.92–0.96)
- [x] Blind cardiologist review of the validation disagreements (agreed with the model in 6 of 7)
- [x] First real-camera test: 36 iPhone photos of ECGs on a laptop screen (F1 0.94–0.97)
- [ ] Test on phone photos of printed ECGs under harder conditions (dim light, angles, shadows)
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

### Fix: retraining with data augmentation

A fresh pretrained ResNet-18 was trained with the same settings as the first model (loss, class weight, optimiser, learning rate, clean validation, best-epoch saving), with one change: every training image was randomly altered each time it was used. Each alteration was applied with 50% probability:

- rotation up to ±8°
- perspective (distortion scale 0.3)
- Gaussian blur (σ between 0.1 and 3.0)
- brightness and contrast (±50%)

Validation and test images were not augmented. Training ran for 8 epochs (about 7 minutes each on a Colab T4, as augmentation runs on the CPU); validation F1 peaked at **epoch 2 (0.967)**, which was kept. Later epochs lowered the training loss without improving validation F1.

**Threshold:** chosen again on the validation set, scoring every threshold on the clean and the four distorted versions. **0.45** was chosen: on clean validation images it matches the first model (3 missed AF, 6 false alarms, F1 0.970), its average F1 across the five versions (0.949) is within 0.008 of the best (0.70), and it misses fewer AF cases than higher thresholds, which matters for screening.

### Final comparison on the test set (fold 10, evaluated once)

| Distortion | First model F1 (threshold 0.30) | **Augmented model F1 (threshold 0.45)** | Augmented: missed AF (of 152) | Augmented: false alarms (of 815) |
|---|---|---|---|---|
| None (clean) | 0.967 | 0.954 | 6 | 8 |
| Rotation | 0.505 | **0.961** | 6 | 6 |
| Perspective | 0.292 | **0.945** | 15 | 1 |
| Blur | 0.272 | **0.916** | 4 | 23 |
| Brightness/contrast | 0.518 | **0.930** | 5 | 17 |

**Result:** augmentation raised test F1 on distorted images from 0.27–0.52 to 0.92–0.96. Under blur, the first model called all 815 normal test ECGs AF; the augmented model calls 23. The cost is small on clean images (F1 0.967 → 0.954, false alarms 4 → 8).

**Remaining weaknesses:** perspective distortion hides some AF (15 of 152 missed, sensitivity 0.901), and blur and lighting changes still cause 17–23 false alarms.

## First real-camera test: phone photos of a screen

**Setup.** 40 ECGs were drawn at random from the test set (20 AF, 20 normal, fixed seed), shown full-screen on a laptop and photographed with an iPhone in ordinary room light, holding the phone naturally. Each photo was cropped to the ECG only (no straightening, filters or editing). No printer was available, so these are photos of a **screen**, not of paper.

**Matching photos to ECGs.** The cropping removed the code printed on each page, and some photos were retakes. Photos were matched back to their ECGs automatically by comparing the trace pattern (dark pixels only) of each photo with the 40 originals, using the order in which the photos were taken. Retakes were dropped (the better-matching photo kept), two photos did not sync to Colab, and two pairs that looked mismatched on visual inspection were excluded. **36 photos were scored (17 AF, 19 normal).** Each model used the threshold chosen earlier on validation.

| Model | Missed AF (of 17) | False alarms (of 19) | Sensitivity | Specificity | F1 |
|---|---|---|---|---|---|
| First model (clean training, threshold 0.30) | 0 | 1 | 1.000 | 0.947 | 0.971 |
| Augmented model (threshold 0.45) | 2 | 0 | 0.882 | 1.000 | 0.938 |

**Interpretation.** Both models worked well on these photos, unlike the first model's collapse under simulated distortions. The photos were cropped, sharp, fairly straight and well lit, so they were milder than the simulated blur, perspective and lighting changes; this test did not include the conditions where augmentation helps most. The augmented model gave very confident predictions on normal photos (probability ≤ 0.01 for all 19) but missed two AF photos that the first model caught with moderate confidence. With 36 photos, the two models differ by 3 photos in total, which is too few to say either is better.

**Limitations of this test:** photos of a screen rather than paper (no paper texture, creases or printer colours; screens can add moiré patterns); a small sample; one phone and one photographer; and matching done by image similarity rather than a visible code.

## Limitations

- Apart from the 36 screen photos, all images are simulated from digital ECG signals. Performance on phone photos of **printed** ECGs (shadows, creases, handwriting, different printers) is untested.
- The augmented model still misses more AF on photos taken at an angle (perspective: sensitivity 0.901).
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
