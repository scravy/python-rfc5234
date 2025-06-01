import nox


@nox.session(python=["3.12", "3.13"])
def tests(session):
    session.run("uv", "pip", "install", ".", external=True)
    session.run("uv", "pip", "install", "pytest", external=True)
    session.run("python", "--version")
    session.run("pytest")
