# Credits

## Nigerian English speech data — OpenSLR SLR70

Nigerian English speech data from OpenSLR SLR70 (https://www.openslr.org/70/),
© its contributors, used under Creative Commons Attribution-ShareAlike 4.0
International (https://creativecommons.org/licenses/by-sa/4.0/).

The fine-tuning set (`voice_dataset.zip`, gitignored, not distributed here)
is derived from the SLR70 male archive: 300 utterances of speaker `07508`,
resampled to 16 kHz mono, loudness-matched to −20 LUFS, with transcripts
normalized for speech. Raw SLR70 audio lives in `data/slr70/` (gitignored);
see `data/slr70_audit.md` for the full audit.

ShareAlike note: distributing the fine-tuned weights (or the dataset) must
carry the same CC BY-SA 4.0 terms with attribution. Private training runs
do not trigger the share-alike distribution clause.
