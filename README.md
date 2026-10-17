# ECG-Vision App

Detecting **atrial fibrillation (AF)** from images of paper ECGs using computer vision.

> **Status:** early stage, a learning and portfolio project. Data preparation is complete; image generation and model training are next.

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
| `ECG_Vision_01_Explore_PTBXL.ipynb` | Download PTB-XL, explore metadata, build NORM vs. AFIB labels and splits, then load raw waveforms with `wfdb` and plot normal vs. AF on ECG-style grids | ✅ Done |

Built and run in **Google Colab**, with data stored in Google Drive.

## Roadmap

- [x] Learn ECG fundamentals (waves, grid, heart rate, rhythm, AF)
- [x] Define the task: Normal vs. AF
- [x] Download, explore and label PTB-XL
- [x] Plot real ECG waveforms
- [ ] Generate realistic ECG-paper images (e.g. with [ECG-Image-Kit](https://github.com/alphanumericslab/ecg-image-kit))
- [ ] Train a baseline CNN image classifier (transfer learning)
- [ ] Evaluate on the held-out test set (sensitivity, specificity, F1)
- [ ] Error analysis with a clinician
- [ ] Test on real phone photos of printed ECGs
- [ ] Prototype a mobile app

## How to run

1. Open the notebooks in Google Colab.
2. Run the setup cell to mount Google Drive and set project paths.
3. On first use, run the download cells in notebook 01 to fetch PTB-XL and save a copy to Drive.
4. In later sessions, run only the setup cell; it restores the dataset from Drive and loads the saved labels.

## Related work

- [George B. Moody PhysioNet Challenge 2024](https://moody-challenge.physionet.org/2024/): digitisation and classification of ECG images
- [PMcardio](https://www.powerfulmedical.com/): a commercial, certified smartphone app for AI ECG interpretation

## Disclaimer

This is an educational research project. It is **not a medical device** and must not be used for clinical diagnosis or treatment decisions.

## Citation

If you use PTB-XL, please cite:

> Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., & Schaeffter, T. (2022). PTB-XL, a large publicly available electrocardiography dataset (version 1.0.3). *PhysioNet*. https://doi.org/10.13026/kfzx-aw45

> Wagner, P., Strodthoff, N., Bousseljot, R.-D., Kreiseler, D., Lunze, F. I., Samek, W., & Schaeffter, T. (2020). PTB-XL, a large publicly available ECG dataset. *Scientific Data*, 7, 154. https://doi.org/10.1038/s41597-020-0495-6

## Author

**Henry Kofi Brandoh**, MPhil candidate, Intelligent Computing Systems, Ashesi University
