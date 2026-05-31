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

The `coregistration_outputs` folder is created inside the same local folder where this repository is stored. For example, if you place the app at:

```text
/Users/yourname/Desktop/OCT_HE_align_interact_v5
```

the outputs will be written to:

```text
/Users/yourname/Desktop/OCT_HE_align_interact_v5/coregistration_outputs
```

No hard-coded computer-specific path is required.

If the same sample is run more than once, the app adds a suffix such as `_02`.

Important output files include:

```text
he_registered.tiff
oct_registered.tiff
registered_mask.tiff
alignment_summary.json
```

The app also saves preview images for masks, overlays, and QC checks.

The clean final files are also copied into a simple `output` folder inside each sample folder:

```text
output/he_registered.tiff
output/oct_registered.tiff
output/registered_mask.tiff
```

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
conda activate alignment_v5_env
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

1. Load images: choose an OCT image and an HE image by file path or upload. You can optionally specify an output folder for that sample.
2. Preprocess: run modality-specific preprocessing for OCT and HE, or click `Run All Processing And Save` to run all remaining steps automatically.
3. Remove background: generate OCT and HE tissue masks.
4. Edit masks: add or erase tissue regions directly on the overlays if needed.
5. Auto-register: run automatic HE-to-OCT alignment.
6. Manually adjust: fine-tune scale, rotation, translation, and HE opacity if needed.
7. Save: save the final registered OCT, registered HE, and overlap mask.

## Loading A Previously Processed Sample

Use the `Load Processed Output` section to reopen a case that was already processed by this app.

1. Enter the pipeline output directory, usually something inside `coregistration_outputs/interactive_app` or a batch output folder.
2. Click `Scan Output Directory`.
3. Pick the processed sample from the dropdown menu.
4. Click `Load Selected Sample`.

The app will automatically fill all available previews, masks, registration overlays, save links, and manual adjustment controls. If the processed sample includes `transform_state.json`, you can make manual alignment adjustments and save again.

## Batch Processing

The app also has a `Batch Process` section for running many samples without manual interaction.

1. Enter the input folder that contains multiple OCT/HE samples.
2. Optionally enter a batch output folder. If left blank, outputs go inside this app's `coregistration_outputs/interactive_app` folder.
3. Enter how many samples should run in parallel.
4. If you want to rerun samples that already have output folders, check `overwrite already processed samples`.
5. Click `Run Batch Registration`.
6. Wait until the batch status says complete.
7. Review the overlay preview shown for each sample.
8. Leave `keep` checked for samples you want to keep.
9. Uncheck `keep` for samples you want to discard.
10. Click `Delete Unchecked Outputs`.

When overwrite is enabled, the app scans the selected output folder using the sample IDs and replaces matching sample output folders instead of creating `_02`, `_03`, and so on.

By default, every sample is marked as `keep`, so nothing is deleted unless you uncheck it.

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
