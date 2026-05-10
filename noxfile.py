import nox

nox.options.default_venv_backend = "none"  # reuse the Poetry-managed venv


@nox.session
def tests(session):
    session.run("poetry", "run", "pytest", "tests/", external=True)


@nox.session
def lint(session):
    session.run("poetry", "run", "ruff", "check", "stormvogel/", external=True)
    session.run("poetry", "run", "pyright", external=True)


@nox.session
def docs(session):
    session.run(
        "poetry",
        "run",
        "sphinx-build",
        "-b",
        "html",
        "docs/",
        "docs/_build/",
        external=True,
    )
