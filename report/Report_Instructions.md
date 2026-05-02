# REPORT

## Installation
### Windows
1. Install **TeX Live**. When the installer opens, choose **Custom Install**, then deselect language collections you do not need (like Cyrillic or Arabic). Ensure "Add to PATH" is checked.
2. Install [Strawberry Perl](https://strawberryperl.com/) (required by `latexmk` and `makeglossaries`).
3. Install [Python](https://www.python.org/downloads/) and then install Pygments (required by `minted` for code highlighting):
    ```
    pip install Pygments
    ```
4. In VS Code, install **LaTeX Workshop** (James Yu).
5. In VS Code, configure the `settings.json` file:
    * Press `Ctrl + Shift + P`.
    * Type "Open User Settings (JSON)" and click it.
    * Add (or merge) these lines into your JSON file:
    ```json
    {
        "latex-workshop.latex.recipe.default": "lastUsed",
        "latex-workshop.latex.autoBuild.run": "onSave",
        "latex-workshop.view.pdf.viewer": "tab",
        "latex-workshop.latex.tools": [
            {
                "name": "latexmk",
                "command": "latexmk",
                "args": [
                    "-synctex=1",
                    "-interaction=nonstopmode",
                    "-file-line-error",
                    "-pdf",
                    "-outdir=%OUTDIR%",
                    "%DOC%"
                ],
                "env": {}
            },
            {
                "name": "latexmk-xelatex",
                "command": "latexmk",
                "args": [
                    "-xelatex",
                    "-shell-escape",
                    "-synctex=1",
                    "-interaction=nonstopmode",
                    "-file-line-error",
                    "%DOC%"
                ],
                "env": {}
            }
        ],
        "latex-workshop.latex.recipes": [
            {
                "name": "latexmk",
                "tools": ["latexmk"]
            },
            {
                "name": "latexmk (xelatex)",
                "tools": ["latexmk-xelatex"]
            }
        ]
    }
    ```

## Run

### Basic report (`report/report.tex`)
1. Open the `.tex` file.
2. Press `Ctrl + S` (uses the default `latexmk` recipe with pdfLaTeX).

### Report template (`report/ReportMain.tex`)
1. Open `ReportMain.tex`.
2. Select the **latexmk (xelatex)** recipe:
    * Press `Ctrl + Shift + P`.
    * Type "LaTeX Workshop: Build with recipe" and click it.
    * Select **latexmk (xelatex)**.
3. After the first build, subsequent saves (`Ctrl + S`) will reuse this recipe automatically.