# Racial Disparities in Automated Speech Recognition

Files and code to reproduce results found in our PNAS paper can be found here.

## Setup
Install the necessary python modules
```bash
pip3 install -r requirements.txt
```

We assume prior generation of audio snippets for both VOC and CORAAL; this process is provided in src/utils/snippet_generation.py with CORAAL mp3 snippets contained in input/CORAAL_audio.

## Clean and standardize all (ground truth and ASR) transcriptions and calculate WER on all snippets
(Only CORAAL data are displayed, using the ground-truth and ASR transcripts contained in input/CORAAL_transcripts.csv)
```bash
python3 src/clean_WER.py
```

## Match audio between black and white speakers, and perform analyses
Note that input files include: VOC_WER.csv (which contains VOC error rates for all 5 ASRs, without transcriptions given privacy constraints), and DDM.csv (which contains the random sampling of 150 CORAAL snippets for DDM encoding).  The full R code is provided in src/analysis.Rmd, which compiles to src/analysis.html.
```bash
Rscript src/analysis.R
```

### Additional analyses are provided in src/utils, including n-gram matched samples, lexicon share, and language modeling