"""JobManager process entry point; deliberately absent from the 2-D registry."""


def run(params, progress):
    from .pipeline import run_sample
    return run_sample(params["spec"], params["out_dir"], progress=progress)
