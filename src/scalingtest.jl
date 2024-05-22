"""
scalingtest(fd::Type{<:SimpleScaling}, data::IDFdata;
    d_out::Real = minimum(values(getduration(data))), q::Integer = 100)

Performs the testing procedure in order to state if the model fd may be rejected considering the data. It returns the p-value of the test.
d_out is the duration to be put in the validation set. By default it will be set to the smallest duration in the data.
q is the number of eigenvalues to compute when using the Zolotarev approximation for the p-value.
"""
function scalingtest(pd_type::Type{<:SimpleScaling}, data::IDFdata;
                    d_out::Real = minimum(values(getduration(data))), q::Integer = 100)

    # First step : parameter estimation
    train_data = excludeduration(data, d_out)
    fitted_model = fit_mle(pd_type, train_data, d_out)

    # Fisher information matrix(-ces)

    return d_out
end