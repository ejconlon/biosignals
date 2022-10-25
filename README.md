# biosignals

## Development

This project uses pip and virtualenv. To prepare the environment and run tests:

    make venv test

The dataset we use is stored in the gitignored `datasets` directory. You can download it with

    make download

Please make sure that all the checks in `make test` pass before committing.
