def finalize_association_config(args):
    association_cfg = args.model.relations

    for branch_cfg in association_cfg.branches.values():
        if float(branch_cfg.coeff) == 0.0:
            branch_cfg.active = False

    if float(association_cfg.regularizer.coeff) == 0.0:
        association_cfg.regularizer.active = False

    if not association_cfg.weighting.active:
        association_cfg.weighting.mode = "disabled"

    return args
