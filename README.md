# XRA ViSNR

### Description
this is the repository for ViSNR


# contents


`dataloaders.py`        -- custom pytorch dataloaders for loading data
`exploratory.py`        -- exploratory analysis
`mask.py`               -- generate gt masks from data [adapted from original source](https://github.com/cmctec/ARCADE/tree/main/useful%20scripts)
`mask2.py`              -- alternative variation
`trainer.py`            -- defines training protocols

`model_weights.pt`      -- state dict file for model parameters
`requirements.txt`       -- virtual environment config.

`vit_attn.py`           -- code for first draft of snr attention with vit

`vitbase.py`            -- base model full script
`visnr.py`              -- final visnr model full script
`vit_base.pdf`          -- base model notebook
`visnr.pdf`             -- final visnr notebook


`test_model.py`         -- run final model on test set

`references.md`         -- includes links to code references used (also 
included in individual scripts)



``

***

The dataset and model weights are too large to upload to github, it can be found from google drive link below:
[Link to Google Drive](https://drive.google.com/drive/folders/1nmKbwY2FyOaELEWV-X4qf6JuzIj5otJw?usp=share_link)
`model_weights`         -- contains trained weights from our ViSNR model
`syntax`                -- full data set
`test_data`             -- directory of test data needed to run this mode.


# main frameworks used
[Pytorch](https://docs.pytorch.org/docs/stable/index.html)
[HuggingFace](https://huggingface.co/docs/transformers/en/model_doc/vit)
[Monai Core](https://docs.monai.io/en/stable/)


# instructions to run

1. clone this repo
2. cd into repo dir
3. create conda virtual env with dependencies

```zsh
conda env create -f environment.yml
conda activate {environment name}
```
4. change PATH variable for data
4. run `visnr.py`

note: path of data directory may need to be modified to run depending on local file paths.






