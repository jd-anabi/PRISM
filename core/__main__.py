import core.app as app

if __name__ == '__main__':
    inits_dict, params_dict, rescale_params, force_params_dict, units_dict, si_factors, model, labels, state_dep_drift = app.setup()
    app.run(inits_dict, params_dict, rescale_params, force_params_dict, units_dict, si_factors, model, labels, state_dep_drift)