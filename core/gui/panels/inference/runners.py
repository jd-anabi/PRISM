"""Worker-callable inference runners: module-level so a Worker can call them with an injected
fig_sink, and free of Qt so they stay testable headless."""
from core import orchestrator


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

