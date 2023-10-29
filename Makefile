.DEFAULT_GOAL := help

SHELL=/bin/bash

.PHONY: fmt
fmt:  ## Run autoformatting and linting
	@poetry run pre-commit run --all-files
