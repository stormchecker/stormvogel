import nox

nox.options.default_venv_backend = "none"  # reuse the uv-managed venv


@nox.session
def tests(session):
    session.run("uv", "run", "--locked", "pytest", "tests/", external=True)


@nox.session
def lint(session):
    session.run("uv", "run", "--locked", "ruff", "check", "stormvogel/", external=True)
    session.run("uv", "run", "--locked", "pyright", external=True)


@nox.session
def docs(session):
    session.run(
        "uv",
        "run",
        "--locked",
        "--all-extras",
        "sphinx-build",
        "-b",
        "html",
        "docs/",
        "docs/_build/",
        external=True,
    )
