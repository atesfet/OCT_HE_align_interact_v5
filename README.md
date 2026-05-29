# OCT_HE_align_interact_v5

Interactive OCT/HE image alignment app, version 5.

Author: Ates Fettahoglu

This app lets you load an OCT image and an H&E image, preprocess them, remove/edit tissue masks, run automatic registration, manually fine-tune the alignment, and save the registered outputs locally.

The app runs on your own computer. Your images are not uploaded to a website or cloud server.

## What The App Produces

For each OCT/HE pair, the app creates an output folder named after the sample files, for example:

```text
coregistration_outputs/interactive_app/sample_name__he_section_name/
```

If the same sample is run more than once, the app adds a suffix such as `_02`.

Important output files include:

```text
he_registered.tiff
oct_registered.tiff
registered_mask.tiff
alignment_summary.json
```

The app also saves preview images for masks, overlays, and QC checks.

## Recommended Installation For Non-Coders

These instructions use Miniconda because it keeps the app separate from the rest of your computer.

### Step 1. Install Miniconda

Download and install Miniconda from:

https://docs.conda.io/en/latest/miniconda.html

Choose the installer for your operating system.

### Step 2. Download This Repository

On GitHub, click:

```text
Code -> Download ZIP
```

Then unzip the downloaded file.

You should have a folder named:

```text
OCT_HE_align_interact_v5
```

### Step 3. Open A Terminal In The Folder

On macOS:

1. Open Terminal.
2. Type `cd ` with a space after it.
3. Drag the `OCT_HE_align_interact_v5` folder into the Terminal window.
4. Press Enter.

### Step 4. Create The App Environment

Run this command once:

```bash
conda env create -f environment.yml
```

This may take several minutes. The first time the app removes backgrounds, `rembg` may also download a model file automatically.

On macOS, after installing Miniconda, you can alternatively double-click `setup_environment.command`.

### Step 5. Start The App

Run:

```bash
conda activate oct_he_align_v5
python src/coregistration_app.py --host 127.0.0.1 --port 8766
```

Then open this address in your web browser:

```text
http://127.0.0.1:8766/
```

Keep the Terminal window open while using the app. Closing the Terminal stops the app.

## Easier macOS Launch After Setup

After the conda environment has been created, macOS users can also run:

```bash
chmod +x run_app.command
```

Then double-click `run_app.command` to start the app.

If macOS blocks the file because it was downloaded from the internet, right-click the file, choose Open, and confirm that you want to open it.

## How To Use The App

1. Load images: choose an OCT image and an HE image by file path or upload.
2. Preprocess: run modality-specific preprocessing for OCT and HE.
3. Remove background: generate OCT and HE tissue masks.
4. Edit masks: add or erase tissue regions directly on the overlays if needed.
5. Auto-register: run automatic HE-to-OCT alignment.
6. Manually adjust: fine-tune scale, rotation, translation, and HE opacity if needed.
7. Save: save the final registered OCT, registered HE, and overlap mask.

## Notes About File Sizes

Microscopy images can be very large. If possible, use file paths instead of browser upload. Path-based loading avoids copying large files into the app folder.

## Troubleshooting

If the browser page does not load, make sure the Terminal says something like:

```text
Interactive coregistration app: http://127.0.0.1:8766
```

If port `8766` is already in use, start the app on another port:

```bash
python src/coregistration_app.py --host 127.0.0.1 --port 8770
```

Then open:

```text
http://127.0.0.1:8770/
```

If installation fails, confirm that you are using the conda environment with Python 3.12. Version 5 includes a recovered v3 registration reference that expects Python 3.12.

## For GitHub Maintainers

Suggested repository name:

```text
OCT_HE_align_interact_v5
```

To push this folder to GitHub:

```bash
cd OCT_HE_align_interact_v5
git init
git add .
git commit -m "Initial OCT HE interactive alignment v5 release"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/OCT_HE_align_interact_v5.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.
