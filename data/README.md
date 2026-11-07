# DataSEC Dataset

The DataSEC audio dataset is not stored in this GitHub repository because the complete archive is approximately 4.6 GB.

## Official Source

Dataset: DataSEC - Dataset for Sound Event Classification of environmental noise

Zenodo record:
https://zenodo.org/records/23097290

Version: v4

Archive: DATASEC.zip

DOI:
https://doi.org/10.5281/zenodo.23097290

## Download

From the project root, run:

```bash
python download_datasec.py
```

The script downloads the official archive and verifies its MD5 checksum.

Alternatively, download DATASEC.zip directly from the Zenodo record.

## Expected Local Arrangement

Keep the dataset outside the GitHub repository.

For example:

```text
AI/
├── Assignment/
│   ├── src/
│   ├── app/
│   ├── data/
│   │   └── README.md
│   └── README.md
│
└── DATASEC/
    ├── DATASEC.zip
    └── DATASEC/
        ├── Bells/
        ├── Birds/
        ├── Cat fights and moans/
        └── ...
```

The exact extracted directory structure should be checked against the downloaded archive.

## Dataset Characteristics

- 4,292 mono-channel WAV recordings
- 44.1 kHz sampling rate
- 22 main sound classes
- 28 subclasses
- Approximately 18 hours and 26 minutes of audio

## Important Licence Notice

DataSEC contains audio collected from different sources and has mixed licensing conditions.

Users must consult the original Zenodo record and the relevant source licences before using, sharing or redistributing the audio files.

This repository does not redistribute the dataset.

## GitHub Notice

Do not commit any of the following:

- DATASEC.zip
- Extracted WAV files
- Copies of the original audio
- Other large dataset archives

Only this README and other appropriate documentation should be stored in the repository's `data/` directory.
