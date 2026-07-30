# Release Notes

## Nightly

### Cramér--von Mises test for IDF curves

* Add the `CvMDistribution` type.
* Add a bootstrap calibration procedure for the scaling test that accounts for inter-duration dependence.
* Compute test-statistic p-values using a Davies-type inversion method with numerical integration provided by `QuadGK.jl`.
* Add documentation for the training-validation goodness-of-fit test applied to IDF data.
