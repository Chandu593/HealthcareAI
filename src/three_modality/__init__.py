"""Three-modality experiment suite (GRU + autoencoder + unified TCD embedding).

Additive sibling of src/paper and the legacy pipeline: nothing in those packages is modified by this one,
and everything written here goes under <run.output_dir>/three_modality/ -- never into outputs/paper,
outputs/ablation, or the legacy label files.

Stage 1 (this file set): the INCIDENT sepsis-proxy label (fixed 24-96 h event window, time-resolved evidence).
"""
