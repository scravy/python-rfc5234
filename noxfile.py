import nox


@nox.session(python=["3.12", "3.13"])
def tests(session):
    session.run("uv", "-q", "pip", "install", ".", external=True)
    session.run("uv", "-q", "pip", "install", "pytest", "pytest-xdist", external=True)
    session.run("python", "--version")
    session.run("pytest", "-n", "auto")
