import typer

from wrf_ensembly.commands import experiment, preprocess

app = typer.Typer()
app.add_typer(experiment.app, name="experiment")
app.add_typer(preprocess.app, name="preprocess")

app()
