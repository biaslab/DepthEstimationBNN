SHELL = /bin/bash
.DEFAULT_GOAL = help

.PHONY: test

test: ## Run tests, use test_args="folder1:test1 folder2:test2" argument to run reduced testset, use dev=true to use `dev-ed` version of core packages
	julia -e '\
		import Pkg; \
		Pkg.activate("."); \
		Pkg.test(test_args = split("$(test_args)") .|> string); \
	'