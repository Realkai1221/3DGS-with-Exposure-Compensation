# Baselines trained for comparison

`v1_original/` is the first version of the exposure code, unchanged:
`trainv1.py` and the four files of its `exposure` package are copies of the
files in `exposure-3dgs-latex/code/`.  It is kept apart from the `exposure`
package of code-v3 because the two have the same name.

`v1_original/render_v1.py` is new.  It renders the test views with the same
exposure adjustment trainv1.py applies during training, so that v1 is evaluated
on the picture it was trained to match (see the comment at its top).

The Colab notebook runs v1 in its own copy of the 2023 gaussian-splatting
repository.  The only change it makes to trainv1.py is one added line that also
saves the exposure values (`exposure_model.pt`), which the evaluation needs.

The 2024 version of 3DGS is not stored here: the notebook downloads it from the
official repository at commit 54c035f (30 Oct 2024) and trains it twice.
`3dgs2024` uses its default options; with those the exposure compensation of
the 2024 code is not applied (the code only uses it with `--train_test_exp`).
`3dgs2024_exp` switches it on with the options given in its README.  That
procedure trains on the left half of every test image and evaluates the right
half, so its numbers are for right halves only.
