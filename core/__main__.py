import core.app as app

if __name__ == '__main__':
    inits_dict, params_dict, force_params_dict, units_dict, si_factors, model, state_dep_drift = app.setup()
    app.run(inits_dict, params_dict, force_params_dict, units_dict, si_factors, model, state_dep_drift)