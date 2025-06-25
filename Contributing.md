# Contributing to AutoFair Explainability

We welcome contributions to the **AutoFair Explainability** repository! Whether you’re fixing bugs, adding features, improving documentation, or writing tutorials — your help is appreciated.

---

## Before You Start

Please read this guide to understand the contribution process and our expectations.

---

## Contribution Guidelines

### What You Can Contribute

- Fix bugs or errors

- Add new fairness metrics or explainability modules

- Refactor or optimize existing code

- Improve documentation and docstrings

- Add or enhance example notebooks

- Write tests to increase coverage
- Add you own explainability methods

## How to contribute

1. Fork the repository on GitHub

2. Create a new branch

3. Make your changes

4. Add or update documentation and tests as needed

5. Commit your changes with clear commit messages

6. Push your branch

7. Open a Pull Request from your branch to the main branch

## Issues
Issues should be created for bugs or feature suggestions. In most cases, a pull request should close one or more issues. If no issue exists, feel free to create an issue and corresponding PR at the same time. If you see an open issue you wish to contribute to, please leave a comment asking to be assigned to it so others know it is being worked on.

Bug reports should be accompanied by a trace showing the error message along with a short example reproducing the issue. If you are able to diagnose the problem and suggest a fix that is especially helpful. If you're willing to implement the change and submit a PR that is best!

If you're interested in contributing but don't know where to start, try filtering the issues by tag: "good first issue", "help wanted", "contribution welcome", or "easy".

## Documentation
Changes to documentation only (e.g., clarifying descriptions, improving the user guide) should be discussed in an issue first. A PR which fixes obvious typos does not require a corresponding issue.

All new functions or classes should include a docstring specifying inputs and outputs and their respective types, a short (and optionally, long) description, and an example snippet, if non-obvious. interconnect uses Google-style docstrings.

To generate the documentation using Sphinx, you will first need to install the [docs] extras if you have not already. Then, run:
```bash
make html
```
from the docs/ directory and review the resulting html files in docs/build/html/. In particular, make sure any new pages are rendered properly, have no broken links, and match existing pages in thoroughness and style.

## Example Notebooks
All new algorithms or metrics should be accompanied by an example notebook demonstrating a typical use case or reproducing experiments from the paper. The notebook should contain text explaining major steps (assume an educated but not expert audience). Results should be reliable and ideally reproducible via random seed and demonstrate a clear advantage. This should be more involved than unit tests and may require increased computational time. Datasets should be easy to load from the notebook without manually downloading and extracting from a link, if possible, and the license and any terms should be clearly presented.

Examples should be placed in the **examples/** directory. They should be named as **demo_<class_name>**.ipynb according to the feature they demonstrate.

## Tests
Tests should be fast (< 15s) and deterministic but avoid overfitting to a random seed. If the assertion result is not obvious, a comment should explain how it was calculated. Remember: these tests exist to catch implementation mistakes, test edge/corner cases, check expected error cases, validate inputs/outputs with other parts of the toolkit, etc. -- example notebooks should demonstrate efficacy and real-life use cases.

## PR Checklist
### Code
Remember to remove unnecessary imports, print statements, and commented code. If any code is copy-pasted from somewhere else, make sure to attribute the source. All added files should be human-readable (no binary files) except example notebooks/images. Any necessary pre-trained models or data should be downloaded from a (trusted) external source.

### Naming, description
Please be descriptive when creating a PR but also remember that the code should speak for itself -- it should be readable with good commenting and documentation. The description should explain the high-level changes, reference the inciting issue, mention the license of any new libraries/datasets, and note any compatibility issues that might arise. This is also a place to leave questions for discussion with the reviewer.

### Draft/WIP
For larger contributions, it may be useful to create a draft PR containing work-in-progress. In this case, please specify if you want feedback from the maintainers since by default they will only review PRs which are marked ready for review and have no merge issues.

### Testing, examples, documentation
Pull requests contributing new features (e.g., metrics, algorithms) must include unit tests. If an existing test is failing, the fix does not require any new tests but a bug not caught by any test should have a new test submitted along with the fix.

New features should also be accompanied by an example notebook.

Also remember to add a line to the corresponding .rst file in docs/source/modules/ so an autosummary will be generated and displayed in the documentation.

Link to issue, tag relevant maintainer
A PR should close at least one relevant issue. If no issue exists yet, just submit the issue and PR at the same time. PRs and issues may be linked by using closing keywords in the description or via the sidebar on the right.



## DCO
This repository requires a Developer's Certificate of Origin 1.1 signoff on every commit. A DCO provides your assurance to the community that you wrote the code you are contributing or have the right to pass on the code that you are contributing. It is generally used in place of a Contributor License Agreement (CLA). You can easily signoff a commit by using the -s or --signoff flag:

```bash
git commit -s -m 'This is my commit message'
```
If you are using the web interface, this should happen automatically. If you've already made a commit, you can fix it by amending the commit and force-pushing the change:
```bash
git commit --amend --no-edit --signoff
git push -f
```
This will only amend your most recent commit and will not affect the message. If there are multiple commits that need fixing, you can try:
```bash
git rebase --signoff HEAD~<n>
git push -f
```
where <n> is the number of commits missing signoffs.

