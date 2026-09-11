"""Worker-callable inference runners: module-level so a Worker can call them with an injected
fig_sink, and free of Qt so they stay testable headless."""
from core import cli, orchestrator


# ── worker-callable runners (module-level so a Worker can call them with an injected fig_sink) ─────
def _run_simulated_inference(cfg, posterior, cell_path, T_obs_s, *, gt_dicts=None, prior=None,
                             fig_sink=None):
    """Mirror orchestrator.run's simulated branch: inject GT + T_obs, simulate, show GT trace + infer.

    ``gt_dicts`` is the hand-entered alternative to ``cell_path``: an (inits, params, rescale, forcing)
    tuple in parse_values_file's shape. It goes through the SAME inject_ground_truth validation, so
    typed values are bounds-checked exactly like a file's. ``prior`` is a LoadedPrior."""
    ignored = (cfg.inject_ground_truth(*gt_dicts) if gt_dicts is not None
               else cli.load_and_validate_gt(cfg, cell_path))
    if ignored:
        print(f"Note: the bounds file does not declare {', '.join(ignored)} — those cell values were "
              f"ignored (the bounds file defines the inferred set).")
    cfg.T_obs = T_obs_s * cfg.get_unit_conversion_factor("s")
    # Is this observation actually in the region the network trained on? Bounds-checking cannot tell.
    if prior is not None:
        for msg in orchestrator.check_observation_in_distribution(cfg, prior.prior, prior.force_prior):
            print(f"WARNING: {msg}")
    obs = orchestrator.generate_observations(cfg, fig_sink=fig_sink)     # writes the artifact + the trace
    inf = orchestrator.infer_and_visualize(cfg, posterior, obs, fig_sink=fig_sink)
    return obs, inf


def _run_experimental_inference(cfg, posterior, rec, *, fig_sink=None):
    """Any bench recording set (passive, driven, chi): the stage checks and hashes the files first."""
    obs = orchestrator.build_experiment_observation(cfg, rec, fig_sink=fig_sink)
    inf = orchestrator.infer_and_visualize(cfg, posterior, obs, fig_sink=fig_sink)
    return obs, inf


def _run_tsnpe_round(cfg, posterior, prior, obs_ref, n_directions, level, num_runs, run_size_cap, *,
                     fig_sink=None):
    """One TSNPE round: region from the posterior around a RECORDED observation -> prior RESTRICTED to
    it -> simulate -> retrain. The proposal is the truncated prior and never the posterior (the rule
    lives in core/SBI/truncate.py and is pinned by tests/test_conditioning_repair.py). Nothing here
    reimplements it; this function only carries the GUI's choices into orchestrator. ``prior`` is a
    LoadedPrior; ``posterior`` is the LoadedPosterior the region is drawn from (session.posterior).

    :return: the LoadedPosterior build_posterior just wrote and read back -- its own
             ``.posterior.truncation`` and ``.posterior.x_obs_digest`` carry the region and the
             observation the round is valid near, so nothing here needs to return them separately.
    """
    from core.artifacts import resolve_store
    obs = resolve_store(None).load_observation(cfg, obs_ref)
    # t_scale_idx: a direction that loads on t_scale is not truncated -- the per-batch override would
    # turn its box into a reweighting (D4) -- and the region records which ones were skipped.
    region = orchestrator.build_truncation_region(posterior, obs, n_directions=n_directions, level=level,
                                                  t_scale_idx=len(cfg.params_dict) + cfg.rescale_idx["t_scale"])
    print(f"[tsnpe] region from observation {obs.name or obs.id}: {region!r}", flush=True)
    return orchestrator.build_posterior(cfg, prior, None, True, fig_sink=fig_sink, num_runs=num_runs,
                                        run_size_cap=run_size_cap, truncation=region, observation=obs,
                                        parent_posterior=posterior)

