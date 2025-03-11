import marimo

__generated_with = "0.10.16"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        "__[[FTName]](https://github.com/apowers313/roc/blob/master/experiments/[DATE_NOW('yyyy.MM.dd-HH.mm.ss')]-[FTName%paramcase]/[DATE_NOW('yyyy.MM.dd-HH.mm.ss')]-[FTName%paramcase].py)__"
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
