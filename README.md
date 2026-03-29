# cortical-remapping-rnn
this is for molding feasible models of the cirrus remapping 

state-bci-rnn/
├── core/
│   ├── __init__.py
│   └── vanilla_rnn.py
├── scripts/
│   └── test_rnn.py
├── tests/
├── notebooks/
├── requirements.txt
├── pyproject.toml
└── README.md


TODO: make the noise Poiss
TODO: analysis code codier that index of the snapshots is eqal to the trials it's very dengarus and can cause a lot of problems if the order of the trials changes. I should fix it so it fetch actual trial numbers instead of relying on the order of the snapshots
## Tip for commitng the notebook:
install nbstripout and have it run automatically whenever committing notebook changes to the (local) git repository. To install nbstripout for use in the repository, navigate to your local copy and call (again assuming the use of uv)
pip install nbstripout

Then install nbstripout as a git filter to your local repository using

nbstripout --install

You can check its installation by
nbstripout --status