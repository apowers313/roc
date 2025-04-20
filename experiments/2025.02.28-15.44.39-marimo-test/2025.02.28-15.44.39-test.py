import marimo

__generated_with = "0.11.20"
app = marimo.App(width="medium", auto_download=["html", "ipynb"])


@app.cell
def _(mo):
    mo.md(
        r"""
        "__[test](https://github.com/apowers313/roc/blob/master/experiments/2025.02.28-15.44.39-test/2025.02.28-15.44.39-test.py)__"
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.__version__
    return


@app.cell
def _(mo):
    print("hi", mo.__version__)
    return


@app.cell
def _():
    import os

    print(os.getcwd())
    return (os,)


@app.cell
def _():
    import sys
    sys.path.append('/home/apowers/Projects/roc')
    return (sys,)


@app.cell
def _(sys):
    print(sys.path)
    return


@app.cell
def _():
    import roc
    return (roc,)


if __name__ == "__main__":
    app.run()
